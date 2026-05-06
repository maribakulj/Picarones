"""``picarones.measurements.statistics.pareto`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.evaluation.statistics.pareto`.  Migration ::

    from picarones.evaluation.statistics import ...
"""

from __future__ import annotations

import warnings

from picarones.evaluation.statistics.pareto import (
    compute_pareto_front,
)

warnings.warn(
    "picarones.measurements.statistics.pareto is deprecated and will be "
    "removed in 2.0.  Import from picarones.evaluation.statistics instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ['compute_pareto_front']
