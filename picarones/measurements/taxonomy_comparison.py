"""Taxonomie comparative entre deux moteurs — Sprint 77 (A.I.4 chantier 3).

Sprint 77 — A.I.4 chantier 3 du plan d'évolution 2026 (clôture A.I.4).

Pourquoi ce module
------------------
Le détecteur narratif ``error_profile_outlier`` (Sprint 19) signale
qu'un moteur a un profil taxonomique éloigné de ses concurrents,
mais le rapport n'expose pas cette différence visuellement.  Ce
sprint répond à *« deux moteurs ont le même CER global, mais lequel
fait des erreurs plus récupérables ? »*.

Lecture concrète
----------------
- Moteur A : 80 % d'erreurs ``case_error`` → toutes corrigeables
  par un post-processing trivial (récupérables).
- Moteur B : 80 % d'erreurs ``lacuna`` (mots manquants) →
  irrécupérables sans relire l'image.

À CER égal, A est massivement préférable pour un workflow
d'édition critique.  Cette vue rend la différence visible.

Catégorisation des classes
--------------------------
On annote chaque classe d'erreur d'un degré de **récupérabilité**
(critère éditorial pragmatique, pas verdict imposé) :

- ``recoverable`` : récupérable par post-processing trivial
  (case_error, ligature_error, abbreviation_error)
- ``difficult`` : récupérable au prix d'un effort
  (diacritic_error, visual_confusion, hapax)
- ``irrecoverable`` : impossible à corriger sans l'image
  (lacuna, oov_character, segmentation_error)

L'utilisateur consulte ces catégories comme un guide, pas un
verdict — c'est lui qui juge selon ses besoins éditoriaux.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Classification éditoriale.  Documentée dans la docstring.
RECOVERABILITY: dict[str, str] = {
    "case_error":         "recoverable",
    "ligature_error":     "recoverable",
    "abbreviation_error": "recoverable",
    "diacritic_error":    "difficult",
    "visual_confusion":   "difficult",
    "hapax":              "difficult",
    "lacuna":             "irrecoverable",
    "oov_character":      "irrecoverable",
    "segmentation_error": "irrecoverable",
}


def _normalize_counts(counts: dict[str, int]) -> dict[str, float]:
    """Convertit un dict de comptes en proportions [0, 1]."""
    total = sum(counts.values())
    if total <= 0:
        return {k: 0.0 for k in counts}
    return {k: v / total for k, v in counts.items()}


def compare_taxonomies(
    engine_a_name: str,
    engine_a_counts: dict[str, int],
    engine_b_name: str,
    engine_b_counts: dict[str, int],
) -> Optional[dict]:
    """Compare deux profils taxonomiques.

    Parameters
    ----------
    engine_a_name, engine_b_name:
        Noms d'identification des moteurs (utilisés dans le rendu).
    engine_a_counts, engine_b_counts:
        Maps ``{class_name: count}`` produites par
        ``aggregate_taxonomy``.

    Returns
    -------
    Optional[dict]
        ``{
            "engine_a": str, "engine_b": str,
            "total_a": int, "total_b": int,
            "classes": list[str],     # classes apparaissant chez A ou B
            "proportions_a": dict[str, float],
            "proportions_b": dict[str, float],
            "deltas": dict[str, float],   # prop_b - prop_a (signé)
            "recoverability": dict[str, str],  # mapping class → niveau
            "totals_by_recoverability": {
                "recoverable":   {"a": float, "b": float},
                "difficult":     {"a": float, "b": float},
                "irrecoverable": {"a": float, "b": float},
            },
        }``
        Ou ``None`` si les deux moteurs ont 0 erreur chacun.
    """
    if engine_a_name == engine_b_name:
        # On accepte des comparaisons même si les noms sont
        # identiques (cas tests), mais on émet un warning.
        logger.warning(
            "[taxonomy_comparison] engine_a et engine_b ont le même nom : %s",
            engine_a_name,
        )

    total_a = sum(engine_a_counts.values()) if engine_a_counts else 0
    total_b = sum(engine_b_counts.values()) if engine_b_counts else 0
    if total_a == 0 and total_b == 0:
        return None

    classes = sorted(set(engine_a_counts) | set(engine_b_counts))
    if not classes:
        return None

    prop_a = _normalize_counts(
        {c: engine_a_counts.get(c, 0) for c in classes},
    )
    prop_b = _normalize_counts(
        {c: engine_b_counts.get(c, 0) for c in classes},
    )
    deltas = {c: prop_b[c] - prop_a[c] for c in classes}

    # Agrégat par récupérabilité (utile pour la lecture rapide)
    totals_recov: dict[str, dict[str, float]] = {
        "recoverable":   {"a": 0.0, "b": 0.0},
        "difficult":     {"a": 0.0, "b": 0.0},
        "irrecoverable": {"a": 0.0, "b": 0.0},
    }
    for cls in classes:
        level = RECOVERABILITY.get(cls, "difficult")
        if level not in totals_recov:
            level = "difficult"
        totals_recov[level]["a"] += prop_a[cls]
        totals_recov[level]["b"] += prop_b[cls]

    return {
        "engine_a": engine_a_name,
        "engine_b": engine_b_name,
        "total_a": total_a,
        "total_b": total_b,
        "classes": classes,
        "proportions_a": prop_a,
        "proportions_b": prop_b,
        "deltas": deltas,
        "recoverability": {
            cls: RECOVERABILITY.get(cls, "difficult") for cls in classes
        },
        "totals_by_recoverability": totals_recov,
    }


__all__ = [
    "RECOVERABILITY",
    "compare_taxonomies",
]
