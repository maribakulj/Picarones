"""``picarones.measurements.narrative.detectors.pareto`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.reports_v2.narrative.detectors.pareto`.
"""

from __future__ import annotations

import warnings

from picarones.reports_v2.narrative.detectors.pareto import (  # noqa: F401
    detect_pareto_alternative,
    detect_cost_outlier,
    detect_pricing_staleness,
)

warnings.warn(
    "picarones.measurements.narrative.detectors.pareto is deprecated and will be removed in 2.0.  "
    "Import from picarones.reports_v2.narrative.detectors.pareto instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ['detect_pareto_alternative', 'detect_cost_outlier', 'detect_pricing_staleness']
