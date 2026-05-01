"""Câblage runner du delta Flesch (Sprint 87 — A.II.2).

Sprint 87 — A.II.2 (vue HTML + câblage runner du delta Flesch
livré par le Sprint 52).

Pourquoi ce module
------------------
Le ``flesch_delta`` mesure la différence de lisibilité entre la
GT et la sortie OCR.  Un score positif signale une *over-
normalisation* typique des LLM/VLM qui modernisent un texte
ancien (le Flesch monte parce que les mots sont plus simples) ;
un score négatif signale une dégradation OCR brutale.

Cette métrique est calculée **automatiquement** par le runner
sur chaque document, agrégée par moteur, et présentée dans le
rapport.

Adaptive masking
----------------
On ne calcule que si la GT contient ≥ 5 mots — en dessous, le
Flesch est trop instable pour être informatif.

Langue
------
Lecture depuis ``corpus.metadata.get("language", "fr")``.  Pour
les corpus mixtes, l'utilisateur peut passer une langue
explicite à l'orchestrateur.
"""

from __future__ import annotations

import logging
import statistics
from typing import Iterable, Optional

from picarones.measurements.readability import (
    Language,
    count_words,
    flesch_delta,
    flesch_score,
)

logger = logging.getLogger(__name__)


_MIN_WORDS_FOR_FLESCH = 5


def compute_readability_metrics(
    reference: Optional[str],
    hypothesis: Optional[str],
    *,
    lang: Language = "fr",
) -> Optional[dict]:
    """Calcule le delta Flesch d'un document avec adaptive masking.

    Retourne ``None`` si la GT contient moins de
    ``_MIN_WORDS_FOR_FLESCH`` mots.
    """
    ref = reference or ""
    n_ref_words = count_words(ref)
    if n_ref_words < _MIN_WORDS_FOR_FLESCH:
        return None
    hyp = hypothesis or ""
    flesch_ref = flesch_score(ref, lang=lang)
    flesch_hyp = flesch_score(hyp, lang=lang) if hyp else None
    delta = (
        flesch_delta(ref, hyp, lang=lang) if hyp else None
    )
    return {
        "lang": lang,
        "flesch_reference": flesch_ref,
        "flesch_hypothesis": flesch_hyp,
        "flesch_delta": delta,
        "n_words_reference": n_ref_words,
    }


def aggregate_readability_metrics(
    per_doc: Iterable[Optional[dict]],
) -> Optional[dict]:
    """Agrège : moyenne/médiane des deltas + part de docs
    « over-normalisés » (delta > +5 points).
    """
    docs = [d for d in per_doc if d]
    if not docs:
        return None
    deltas = [
        float(d["flesch_delta"]) for d in docs
        if isinstance(d.get("flesch_delta"), (int, float))
    ]
    if not deltas:
        return None
    over_norm = sum(1 for d in deltas if d > 5.0)
    under_norm = sum(1 for d in deltas if d < -5.0)
    lang = docs[0].get("lang") or "fr"
    return {
        "lang": lang,
        "n_docs": len(docs),
        "n_docs_with_delta": len(deltas),
        "delta_mean": statistics.fmean(deltas),
        "delta_median": statistics.median(deltas),
        "delta_min": min(deltas),
        "delta_max": max(deltas),
        "n_over_normalized": over_norm,
        "n_under_normalized": under_norm,
        "over_normalized_rate": over_norm / len(deltas),
    }


__all__ = [
    "compute_readability_metrics",
    "aggregate_readability_metrics",
]
