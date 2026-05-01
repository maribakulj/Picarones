"""Câblage runner de la recherchabilité (Sprint 86).

Sprint 86 — A.II.5a (vue HTML + câblage runner).

Le module ``picarones/core/searchability.py`` (Sprint 84) a livré
la couche de calcul.  Ce helper prépare la donnée pour le runner
historique et l'agrégation par moteur.

Adaptive masking
----------------
Comme pour les modules philologiques (Sprint 61), on ne calcule
le rappel que si la GT contient au moins un token —  pas de
calcul vide qui produirait du bruit dans le rapport.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

from picarones.measurements.searchability import (
    _split_words,
    compute_searchability,
)

logger = logging.getLogger(__name__)


def compute_searchability_metrics(
    reference: Optional[str],
    hypothesis: Optional[str],
    *,
    max_distance: int = 2,
) -> Optional[dict]:
    """Recherchabilité d'un document (adaptive).

    Retourne ``None`` si la GT est vide ou ne contient aucun
    token — ce qui déclenche l'adaptive masking côté HTML.
    """
    if not reference or not _split_words(reference):
        return None
    return compute_searchability(
        reference, hypothesis or "", max_distance=max_distance,
    )


def aggregate_searchability_metrics(
    per_doc: Iterable[Optional[dict]],
) -> Optional[dict]:
    """Agrège les métriques par-doc en un score corpus-wide.

    Convention : on somme les ``n_gt_tokens`` et ``n_searchable``
    et on recalcule un rappel **micro** (cohérent avec ECE/MCE
    Sprint 39 et NER Sprint 38).
    """
    docs = [d for d in per_doc if d]
    if not docs:
        return None
    n_gt = sum(int(d.get("n_gt_tokens") or 0) for d in docs)
    n_search = sum(int(d.get("n_searchable") or 0) for d in docs)
    if n_gt == 0:
        return None
    # On garde l'union des missed_tokens (capped pour ne pas
    # exploser le JSON sur de gros corpus)
    missed: list[str] = []
    for d in docs:
        missed.extend(d.get("missed_tokens") or [])
    return {
        "n_docs": len(docs),
        "n_gt_tokens": n_gt,
        "n_searchable": n_search,
        "recall": n_search / n_gt,
        "missed_tokens_sample": missed[:50],
        "max_distance": docs[0].get("max_distance", 2),
    }


__all__ = [
    "compute_searchability_metrics",
    "aggregate_searchability_metrics",
]
