"""``picarones.measurements.narrative.detectors.ranking`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.reports_v2.narrative.detectors.ranking`.
"""

from __future__ import annotations

import warnings

from picarones.reports_v2.narrative.detectors.ranking import (  # noqa: F401
    detect_global_leader_cer,
    detect_statistical_tie,
    detect_significant_gap,
    detect_speed_winner,
    detect_median_mean_gap_warning,
)

warnings.warn(
    "picarones.measurements.narrative.detectors.ranking is deprecated and will be removed in 2.0.  "
    "Import from picarones.reports_v2.narrative.detectors.ranking instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ['detect_global_leader_cer', 'detect_statistical_tie', 'detect_significant_gap', 'detect_speed_winner', 'detect_median_mean_gap_warning']
