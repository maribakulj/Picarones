"""``picarones.measurements.difficulty`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.evaluation.metrics.difficulty`.
"""

from __future__ import annotations

import warnings

from picarones.evaluation.metrics.difficulty import (  # noqa: F401
    DifficultyScore,
    compute_difficulty_score,
    compute_all_difficulties,
    difficulty_label,
    _W_VARIANCE,
    _W_QUALITY,
    _W_DENSITY,
    _SPECIAL_CHARS_RE,
    _special_char_density,
    _variance,
)

warnings.warn(
    "picarones.measurements.difficulty is deprecated and will be removed in 2.0.  "
    "Import from picarones.evaluation.metrics.difficulty instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ['DifficultyScore', 'compute_difficulty_score', 'compute_all_difficulties', 'difficulty_label']
