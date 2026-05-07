"""``picarones.report.report_data.engines`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.reports_v2.html.data.engines`.  Phase 5.E
du retrait du legacy.
"""

from __future__ import annotations

import warnings

from picarones.reports_v2.html.data.engines import *  # noqa: F401, F403

warnings.warn(
    "picarones.report.report_data.engines is deprecated and will be removed in 2.0.  "
    "Import from picarones.reports_v2.html.data.engines instead.",
    DeprecationWarning,
    stacklevel=2,
)
