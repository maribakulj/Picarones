"""``picarones.measurements.narrative.detectors.ensemble`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.reports_v2.narrative.detectors.ensemble`.
"""

from __future__ import annotations

import warnings

from picarones.reports_v2.narrative.detectors.ensemble import (  # noqa: F401
    detect_ensemble_opportunity,
)

warnings.warn(
    "picarones.measurements.narrative.detectors.ensemble is deprecated and will be removed in 2.0.  "
    "Import from picarones.reports_v2.narrative.detectors.ensemble instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ['detect_ensemble_opportunity']
