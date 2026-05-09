"""Score de spécialisation inter-moteurs — Sprint 89 (A.II.8b).

Sprint 89 — A.II.8b du plan d'évolution 2026.

Pourquoi ce module
------------------
La matrice de divergence taxonomique (Sprint 35
``inter_engine.taxonomy_divergence_matrix``) répond à *« à quel
point ces moteurs se trompent-ils différemment ? »*.  Ce
sprint la transforme en un **score de spécialisation** lisible
et complète la lecture par :

- une **classification** discrète (similar / distinct /
  highly_specialized) que le chercheur peut consommer sans
  avoir à interpréter une distance ;
- un **top-N des paires** les plus spécialisées, qui répond
  directement à la question *« quels moteurs sont les meilleurs
  candidats pour un voting ensemble ? »*.

Ce module **ne recommande pas** de pipeline d'ensemble — il
fournit l'observation factuelle et laisse le chercheur arbitrer.

Convention de score
-------------------
On utilise la **Jensen-Shannon divergence** déjà calculée par
``inter_engine.jensen_shannon_divergence`` : elle est
symétrique, bornée dans [0, 1], et son interprétation est
intuitive :

- ≈ 0 → profils taxonomiques identiques
- 1 → distributions totalement disjointes

Dépendances
-----------
S'appuie strictement sur ``picarones.evaluation.metrics.inter_engine`` (Sprint
35) — pas de double calcul, pas de logique nouvelle de
divergence.
"""

from __future__ import annotations

import logging
from typing import Optional

from picarones.evaluation.metrics.inter_engine import jensen_shannon_divergence

logger = logging.getLogger(__name__)


# Seuils par convention éditoriale.  La roadmap ne fixe rien :
# ces seuils sont des **guides de lecture**, pas des verdicts.
# Le chercheur peut les surcharger via ``classify_specialization``.
DEFAULT_THRESHOLDS = (
    ("similar", 0.10),
    ("distinct", 0.30),
    ("highly_specialized", 1.01),  # tout score ≥ 0.30
)


def compute_specialization_score(
    taxonomy_a: dict[str, float],
    taxonomy_b: dict[str, float],
) -> float:
    """Score de spécialisation entre deux moteurs ∈ [0, 1].

    0 = mêmes erreurs, 1 = erreurs totalement disjointes.
    Délègue à ``jensen_shannon_divergence`` (Sprint 35).
    """
    return jensen_shannon_divergence(taxonomy_a, taxonomy_b)


def classify_specialization(
    score: float,
    thresholds: Optional[tuple[tuple[str, float], ...]] = None,
) -> str:
    """Classe le score en catégorie discrète.

    Convention :
    - score < 0.10 → ``similar``
    - 0.10 ≤ score < 0.30 → ``distinct``
    - score ≥ 0.30 → ``highly_specialized``

    L'utilisateur peut passer ses propres ``thresholds`` (liste
    triée par valeur croissante de tuples ``(label, max_score)``).
    """
    rules = thresholds or DEFAULT_THRESHOLDS
    for label, max_score in rules:
        if score < max_score:
            return label
    # Garde-fou : si aucun seuil ne match, dernière catégorie
    return rules[-1][0]


def compute_specialization_matrix(
    taxonomies: dict[str, dict[str, float]],
) -> Optional[dict]:
    """Matrice de spécialisation symétrique entre tous les moteurs.

    Parameters
    ----------
    taxonomies:
        Map ``{engine_name: {error_class: count_or_proportion}}``.

    Returns
    -------
    dict | None
        ``{
            "engines": list[str],
            "matrix": list[list[float]],       # carrée, symétrique
            "n_pairs": int,                     # paires distinctes
            "max_score": float,
            "max_pair": (str, str) | None,
        }`` ; ``None`` si moins de 2 moteurs.
    """
    if not taxonomies or len(taxonomies) < 2:
        return None
    engines = sorted(taxonomies.keys())
    n = len(engines)
    matrix = [[0.0] * n for _ in range(n)]
    n_pairs = 0
    max_score = 0.0
    max_pair: Optional[tuple[str, str]] = None
    for i in range(n):
        for j in range(i + 1, n):
            score = compute_specialization_score(
                taxonomies[engines[i]], taxonomies[engines[j]],
            )
            matrix[i][j] = score
            matrix[j][i] = score
            n_pairs += 1
            if score > max_score:
                max_score = score
                max_pair = (engines[i], engines[j])
    return {
        "engines": engines,
        "matrix": matrix,
        "n_pairs": n_pairs,
        "max_score": max_score,
        "max_pair": max_pair,
    }


def top_specialized_pairs(
    matrix_data: Optional[dict],
    n: int = 5,
    *,
    min_score: float = 0.0,
) -> list[dict]:
    """Top-N paires de moteurs triées par score décroissant.

    Returns
    -------
    list[dict]
        Une liste de ``{
            "engine_a": str, "engine_b": str,
            "score": float, "category": str,
        }`` triée par score décroissant.  Liste vide si
        ``matrix_data`` est ``None`` ou que toutes les paires
        sont sous ``min_score``.
    """
    if not matrix_data:
        return []
    engines = matrix_data["engines"]
    matrix = matrix_data["matrix"]
    pairs: list[dict] = []
    for i, engine_a in enumerate(engines):
        for j in range(i + 1, len(engines)):
            score = matrix[i][j]
            if score < min_score:
                continue
            pairs.append({
                "engine_a": engine_a,
                "engine_b": engines[j],
                "score": score,
                "category": classify_specialization(score),
            })
    pairs.sort(key=lambda p: -p["score"])
    return pairs[:n]


__all__ = [
    "DEFAULT_THRESHOLDS",
    "compute_specialization_score",
    "classify_specialization",
    "compute_specialization_matrix",
    "top_specialized_pairs",
]
