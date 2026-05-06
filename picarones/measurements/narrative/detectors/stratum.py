"""``picarones.measurements.narrative.detectors.stratum`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.reports_v2.narrative.detectors.stratum`.
"""

from __future__ import annotations

import warnings

from picarones.reports_v2.narrative.detectors.stratum import (  # noqa: F401
    detect_stratum_winner,
    detect_stratum_collapse,
    detect_stratification_recommended,
    _stratum_cer_by_engine,
)

warnings.warn(
    "picarones.measurements.narrative.detectors.stratum is deprecated and will be removed in 2.0.  "
    "Import from picarones.reports_v2.narrative.detectors.stratum instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ['detect_stratum_winner', 'detect_stratum_collapse', 'detect_stratification_recommended']
