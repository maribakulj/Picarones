"""``picarones.report.report_data.pareto`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.reports_v2.html.data.pareto`.  Phase 5.E
du retrait du legacy.
"""

from __future__ import annotations

import warnings

from picarones.reports_v2.html.data.pareto import *  # noqa: F401, F403

warnings.warn(
    "picarones.report.report_data.pareto is deprecated and will be removed in 2.0.  "
    "Import from picarones.reports_v2.html.data.pareto instead.",
    DeprecationWarning,
    stacklevel=2,
)
