"""``picarones.report.views.robustness`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.reports_v2.html.views.robustness`.  Phase 5.D
du retrait du legacy.
"""

from __future__ import annotations

import warnings

from picarones.reports_v2.html.views.robustness import *  # noqa: F401, F403

warnings.warn(
    "picarones.report.views.robustness is deprecated and will be removed in 2.0.  "
    "Import from picarones.reports_v2.html.views.robustness instead.",
    DeprecationWarning,
    stacklevel=2,
)
