"""``picarones.measurements.narrative.detectors.quality`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.reports_v2.narrative.detectors.quality`.
"""

from __future__ import annotations

import warnings

from picarones.reports_v2.narrative.detectors.quality import (  # noqa: F401
    detect_error_profile_outlier,
    detect_llm_hallucination_flag,
    detect_robustness_fragile,
    detect_confidence_warning,
)

warnings.warn(
    "picarones.measurements.narrative.detectors.quality is deprecated and will be removed in 2.0.  "
    "Import from picarones.reports_v2.narrative.detectors.quality instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ['detect_error_profile_outlier', 'detect_llm_hallucination_flag', 'detect_robustness_fragile', 'detect_confidence_warning']
