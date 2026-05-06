"""``picarones.measurements.statistics.correlation`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.evaluation.statistics.correlation`.  Migration ::

    from picarones.evaluation.statistics import ...
"""

from __future__ import annotations

import warnings

from picarones.evaluation.statistics.correlation import (
    compute_correlation_matrix,
)

warnings.warn(
    "picarones.measurements.statistics.correlation is deprecated and will be "
    "removed in 2.0.  Import from picarones.evaluation.statistics instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ['compute_correlation_matrix']
