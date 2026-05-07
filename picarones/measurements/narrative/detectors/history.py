"""``picarones.measurements.narrative.detectors.history`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.reports_v2.narrative.detectors.history`.
"""

from __future__ import annotations

import warnings

from picarones.reports_v2.narrative.detectors.history import (  # noqa: F401
    detect_engine_off_baseline,
    detect_engine_unstable,
    detect_regression_in_history,
    detect_importer_fallback,
)

warnings.warn(
    "picarones.measurements.narrative.detectors.history is deprecated and will be removed in 2.0.  "
    "Import from picarones.reports_v2.narrative.detectors.history instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ['detect_engine_off_baseline', 'detect_engine_unstable', 'detect_regression_in_history', 'detect_importer_fallback']
