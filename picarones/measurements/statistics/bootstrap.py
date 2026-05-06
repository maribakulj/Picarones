"""``picarones.measurements.statistics.bootstrap`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.evaluation.statistics.bootstrap`.  Migration ::

    from picarones.evaluation.statistics import ...
"""

from __future__ import annotations

import warnings

from picarones.evaluation.statistics.bootstrap import (
    bootstrap_ci,
)

warnings.warn(
    "picarones.measurements.statistics.bootstrap is deprecated and will be "
    "removed in 2.0.  Import from picarones.evaluation.statistics instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ['bootstrap_ci']
