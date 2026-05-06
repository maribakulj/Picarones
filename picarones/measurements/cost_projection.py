"""``picarones.measurements.cost_projection`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.evaluation.metrics.cost_projection`.
"""

from __future__ import annotations

import warnings

from picarones.evaluation.metrics.cost_projection import (  # noqa: F401
    ProjectedCost,
    project_cost_total,
    project_co2_total,
    project_engine,
    project_all_engines,
    cost_gap_table,
)

warnings.warn(
    "picarones.measurements.cost_projection is deprecated and will be removed in 2.0.  "
    "Import from picarones.evaluation.metrics.cost_projection instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ['ProjectedCost', 'project_cost_total', 'project_co2_total', 'project_engine', 'project_all_engines', 'cost_gap_table']
