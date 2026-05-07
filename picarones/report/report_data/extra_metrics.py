"""``picarones.report.report_data.extra_metrics`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.reports_v2.html.data.extra_metrics`.  Phase 5.E
du retrait du legacy.
"""

from __future__ import annotations

import warnings

from picarones.reports_v2.html.data.extra_metrics import *  # noqa: F401, F403

warnings.warn(
    "picarones.report.report_data.extra_metrics is deprecated and will be removed in 2.0.  "
    "Import from picarones.reports_v2.html.data.extra_metrics instead.",
    DeprecationWarning,
    stacklevel=2,
)
