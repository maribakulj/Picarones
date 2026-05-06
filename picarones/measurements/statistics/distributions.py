"""``picarones.measurements.statistics.distributions`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.evaluation.statistics.distributions`.  Migration ::

    from picarones.evaluation.statistics import ...
"""

from __future__ import annotations

import warnings

from picarones.evaluation.statistics.distributions import (
    compute_reliability_curve,
    compute_venn_data,
)

warnings.warn(
    "picarones.measurements.statistics.distributions is deprecated and will be "
    "removed in 2.0.  Import from picarones.evaluation.statistics instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ['compute_reliability_curve', 'compute_venn_data']
