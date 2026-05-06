"""``picarones.measurements.statistics.wilcoxon`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.evaluation.statistics.wilcoxon`.  Migration ::

    from picarones.evaluation.statistics import ...
"""

from __future__ import annotations

import warnings

from picarones.evaluation.statistics.wilcoxon import (
    compute_pairwise_stats,
    wilcoxon_test,
    _SCIPY_AVAILABLE,
    _normal_sf,
)

warnings.warn(
    "picarones.measurements.statistics.wilcoxon is deprecated and will be "
    "removed in 2.0.  Import from picarones.evaluation.statistics instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ['compute_pairwise_stats', 'wilcoxon_test', '_SCIPY_AVAILABLE', '_normal_sf']
