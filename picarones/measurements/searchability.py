"""Recherchabilité fuzzy — Sprint 84 (A.II.5).

Sprint 84 — A.II.5 du plan d'évolution 2026.

Pourquoi ce module
------------------
Le CER mesure les erreurs caractère par caractère.  Mais pour
un usage *recherche plein-texte* (ce que font Elastic, Solr en
mode fuzzy, ou la recherche full-text de Gallica), la question
réelle est :

    *« Combien de mots de ma GT sont retrouvables dans la
    sortie OCR, à orthographe approchée près ? »*

Un CER de 8 % peut donner 95 % de findability si les erreurs
sont concentrées sur des caractères non-significatifs ou sur
quelques mots aberrants ; à l'inverse, 4 % de CER mais
distribué sur tous les noms propres rend le corpus inutilisable
pour l'indexation prosopographique.

Méthode
-------
Pour chaque token GT, on regarde s'il existe au moins un token
hypothèse à distance de Levenshtein ≤ ``max_distance`` (défaut
2, valeur Elastic ``fuzziness: AUTO`` standard pour mots ≥ 5
caractères).  Le **rappel** est la proportion de tokens GT
ainsi retrouvés.

Multiplicité
------------
Si la GT contient *« le »* deux fois et l'hypothèse une fois,
seul un token GT est compté comme retrouvé (alignement
multi-set, comme ``rare_token_recall`` Sprint 71).

Sortie
------
``compute_searchability(reference, hypothesis)`` retourne
``{n_gt_tokens, n_searchable, recall, missed_tokens}``.

Limites documentées
-------------------
- Tokenisation par split sur whitespace (cohérent avec le reste
  du codebase).  Pas de stemming ni de lemmatisation.
- Levenshtein non pondéré — substitution = insertion = suppression
  = 1.  Pour un poids différent (par ex. faute classique
  diacritique = 0,5), passer une fonction custom.
- Pas de sémantique : *« roi »* ≠ *« souverain »*.  Pour la
  similarité sémantique, voir des modules futurs (BERTScore).
"""

from __future__ import annotations

import logging
from typing import Optional

from picarones.core.metric_registry import register_metric
from picarones.core.modules import ArtifactType

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Tokenisation et distance d'édition
# ──────────────────────────────────────────────────────────────────────────


def _split_words(text: Optional[str]) -> list[str]:
    """Tokenisation par whitespace — cohérent avec
    ``lexical_modernization.py``, ``rare_tokens.py``, etc."""
    if not text:
        return []
    return text.split()


def levenshtein_distance(a: str, b: str) -> int:
    """Distance de Levenshtein (substitution=insertion=suppression=1).

    Implémentation DP O(|a|·|b|) en mémoire O(min(|a|,|b|)).
    """
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    # |a| ≥ |b|
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current[j] = min(
                current[j - 1] + 1,        # insertion
                previous[j] + 1,           # suppression
                previous[j - 1] + cost,    # substitution
            )
        previous = current
    return previous[-1]


# ──────────────────────────────────────────────────────────────────────────
# Calcul principal
# ──────────────────────────────────────────────────────────────────────────


def compute_searchability(
    reference: Optional[str],
    hypothesis: Optional[str],
    *,
    max_distance: int = 2,
    case_sensitive: bool = False,
) -> dict:
    """Recherchabilité fuzzy de ``reference`` dans ``hypothesis``.

    Parameters
    ----------
    reference, hypothesis:
        Transcriptions GT et OCR.
    max_distance:
        Seuil de distance de Levenshtein (≤ pour considérer un
        token comme retrouvé).  Défaut 2 — convention
        ``fuzziness: AUTO`` d'Elastic pour mots ≥ 5 caractères.
    case_sensitive:
        Si False (défaut), casse insensible côté match — la
        sortie ``missed_tokens`` reste avec la casse GT
        originale.

    Returns
    -------
    dict
        ``{
            "n_gt_tokens": int,
            "n_searchable": int,
            "recall": float | None,    # None si n_gt_tokens == 0
            "missed_tokens": list[str],
            "max_distance": int,
        }``
    """
    if max_distance < 0:
        raise ValueError(f"max_distance doit être ≥ 0, reçu {max_distance}")
    gt_tokens = _split_words(reference)
    hyp_tokens = _split_words(hypothesis)
    n_gt = len(gt_tokens)
    if n_gt == 0:
        return {
            "n_gt_tokens": 0,
            "n_searchable": 0,
            "recall": None,
            "missed_tokens": [],
            "max_distance": max_distance,
        }
    # Multi-set : un token hypothèse ne peut servir qu'une fois.
    # Tri par longueur croissante pour matcher d'abord les
    # tokens GT les plus courts (où ε-fautes sont plus rares).
    if case_sensitive:
        gt_for_match = list(gt_tokens)
        hyp_for_match = list(hyp_tokens)
    else:
        gt_for_match = [t.lower() for t in gt_tokens]
        hyp_for_match = [t.lower() for t in hyp_tokens]

    hyp_used = [False] * len(hyp_for_match)
    n_searchable = 0
    missed: list[str] = []
    for gi, gt_match in enumerate(gt_for_match):
        # Court-circuit si match exact disponible
        best_idx = -1
        best_dist = max_distance + 1
        for hi, used in enumerate(hyp_used):
            if used:
                continue
            hyp_match = hyp_for_match[hi]
            # Court-circuit longueur (Levenshtein ≥ |Δlen|)
            if abs(len(hyp_match) - len(gt_match)) > max_distance:
                continue
            d = levenshtein_distance(gt_match, hyp_match)
            if d < best_dist:
                best_dist = d
                best_idx = hi
                if d == 0:
                    break  # match exact, inutile de chercher mieux
        if best_idx >= 0 and best_dist <= max_distance:
            hyp_used[best_idx] = True
            n_searchable += 1
        else:
            missed.append(gt_tokens[gi])
    recall = n_searchable / n_gt
    return {
        "n_gt_tokens": n_gt,
        "n_searchable": n_searchable,
        "recall": recall,
        "missed_tokens": missed,
        "max_distance": max_distance,
    }


# ──────────────────────────────────────────────────────────────────────────
# Enregistrement registre typé (Sprint 34)
# ──────────────────────────────────────────────────────────────────────────


@register_metric(
    name="searchability_recall",
    input_types=(ArtifactType.TEXT, ArtifactType.TEXT),
    description=(
        "Recherchabilité fuzzy : proportion de tokens GT retrouvés "
        "dans l'OCR à distance de Levenshtein ≤ 2. Proxy direct de "
        "la qualité pour la recherche plein-texte (Elastic, Solr)."
    ),
)
def searchability_recall_metric(reference: str, hypothesis: str) -> float:
    """Variante scalaire pour le registre typé : retourne le
    rappel en [0, 1], ou ``0.0`` si la GT est vide (convention
    cohérente avec rare_token_recall Sprint 71).
    """
    result = compute_searchability(reference, hypothesis)
    recall = result.get("recall")
    return 0.0 if recall is None else recall


__all__ = [
    "levenshtein_distance",
    "compute_searchability",
    "searchability_recall_metric",
]
