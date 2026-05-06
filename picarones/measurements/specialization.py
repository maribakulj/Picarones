"""``picarones.measurements.specialization`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.evaluation.metrics.specialization`.
"""

from __future__ import annotations

import warnings

from picarones.evaluation.metrics.specialization import (  # noqa: F401
    DEFAULT_THRESHOLDS,
    compute_specialization_score,
    classify_specialization,
    compute_specialization_matrix,
    top_specialized_pairs,
)

warnings.warn(
    "picarones.measurements.specialization is deprecated and will be removed in 2.0.  "
    "Import from picarones.evaluation.metrics.specialization instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ['DEFAULT_THRESHOLDS', 'compute_specialization_score', 'classify_specialization', 'compute_specialization_matrix', 'top_specialized_pairs']
