"""``picarones.measurements.narrative.detectors._helpers`` — shim
re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.reports_v2.narrative.detectors._helpers`.
"""

from __future__ import annotations

import warnings

from picarones.reports_v2.narrative.detectors._helpers import (  # noqa: F401
    _engine_by_name,
    _engines_summary,
    _mean_duration_per_engine,
    _n_docs,
)

warnings.warn(
    "picarones.measurements.narrative.detectors._helpers is deprecated "
    "and will be removed in 2.0.  Import from "
    "picarones.reports_v2.narrative.detectors._helpers instead.",
    DeprecationWarning,
    stacklevel=2,
)
