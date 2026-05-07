"""``picarones.report.report_data`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.reports_v2.html.data`.  Phase 5.E du
retrait du legacy.
"""

from __future__ import annotations

import warnings

from picarones.reports_v2.html.data import *  # noqa: F401, F403
from picarones.reports_v2.html.data import (  # noqa: F401
    build_report_data,
)

warnings.warn(
    "picarones.report.report_data is deprecated and will be removed in 2.0.  "
    "Import from picarones.reports_v2.html.data instead.",
    DeprecationWarning,
    stacklevel=2,
)
