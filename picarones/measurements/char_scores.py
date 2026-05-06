"""``picarones.measurements.char_scores`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.evaluation.metrics.char_scores`.
"""

from __future__ import annotations

import warnings

from picarones.evaluation.metrics.char_scores import (  # noqa: F401
    LIGATURE_TABLE,
    DIACRITIC_MAP,
    LigatureScore,
    DiacriticScore,
    compute_ligature_score,
    compute_diacritic_score,
    aggregate_ligature_scores,
    aggregate_diacritic_scores,
    _ALL_LIGATURES,
    _SEQ_TO_LIGATURE,
    _build_diacritic_map,
    _ALL_DIACRITICS,
    _LIGATURE_SET,
    _check_char_at_context,
)

warnings.warn(
    "picarones.measurements.char_scores is deprecated and will be removed in 2.0.  "
    "Import from picarones.evaluation.metrics.char_scores instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ['LIGATURE_TABLE', 'DIACRITIC_MAP', 'LigatureScore', 'DiacriticScore', 'compute_ligature_score', 'compute_diacritic_score', 'aggregate_ligature_scores', 'aggregate_diacritic_scores']
