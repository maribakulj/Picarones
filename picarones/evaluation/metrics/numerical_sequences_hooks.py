"""Câblage runner des séquences numériques

A.II.5b (vue HTML + câblage runner).

Le module ``picarones/core/numerical_sequences.py``
a livré la couche de calcul.  Ce helper prépare la donnée
adaptative pour le runner et agrège les compteurs par moteur.

Adaptive masking
----------------
On ne stocke le résultat que si la GT contient au moins une
séquence numérique détectée — sinon le module n'apparaît pas
dans le rapport.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

from picarones.evaluation.metrics.numerical_sequences import (
    CATEGORIES,
    compute_numerical_sequence_metrics,
)

logger = logging.getLogger(__name__)


def compute_numerical_sequence_metrics_adaptive(
    reference: Optional[str],
    hypothesis: Optional[str],
) -> Optional[dict]:
    """Calcule les métriques séquences numériques avec masquage
    adaptatif : retourne ``None`` si la GT n'en contient
    aucune."""
    if not reference:
        return None
    result = compute_numerical_sequence_metrics(reference, hypothesis or "")
    if (result.get("n_total") or 0) == 0:
        return None
    return result


def aggregate_numerical_sequence_metrics(
    per_doc: Iterable[Optional[dict]],
) -> Optional[dict]:
    """Agrège par moteur : somme les compteurs par catégorie et
    recalcule les scores globaux et per-category.

    Format de sortie identique à ``compute_numerical_sequence_metrics``
    pour faciliter le rendu HTML symétrique.
    """
    docs = [d for d in per_doc if d]
    if not docs:
        return None
    total_n = 0
    total_strict = 0
    total_value = 0
    per_cat: dict[str, dict] = {}
    for cat in CATEGORIES:
        per_cat[cat] = {
            "n_total": 0,
            "strict": 0,
            "value": 0,
            "lost_items": [],
        }
    for d in docs:
        for cat in CATEGORIES:
            cat_data = (d.get("per_category") or {}).get(cat) or {}
            per_cat[cat]["n_total"] += int(cat_data.get("n_total") or 0)
            per_cat[cat]["strict"] += int(cat_data.get("strict") or 0)
            per_cat[cat]["value"] += int(cat_data.get("value") or 0)
            per_cat[cat]["lost_items"].extend(
                cat_data.get("lost_items") or [],
            )
        total_n += int(d.get("n_total") or 0)
    # Recalcul des scores
    for cat, slot in per_cat.items():
        n = slot["n_total"]
        slot["strict_score"] = slot["strict"] / n if n else 0.0
        slot["value_score"] = slot["value"] / n if n else 0.0
        # Cap des lost_items à 50 par catégorie
        slot["lost_items"] = slot["lost_items"][:50]
        total_strict += slot["strict"]
        total_value += slot["value"]
    return {
        "n_docs": len(docs),
        "n_total": total_n,
        "global_strict_score": (
            total_strict / total_n if total_n else 0.0
        ),
        "global_value_score": (
            total_value / total_n if total_n else 0.0
        ),
        "per_category": per_cat,
    }


__all__ = [
    "compute_numerical_sequence_metrics_adaptive",
    "aggregate_numerical_sequence_metrics",
]
