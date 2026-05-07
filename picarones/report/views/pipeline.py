"""``picarones.report.views.pipeline`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.reports_v2.html.views.pipeline`.  Phase 5.D
du retrait du legacy.
"""

from __future__ import annotations

import warnings

from picarones.reports_v2.html.views.pipeline import *  # noqa: F401, F403

warnings.warn(
    "picarones.report.views.pipeline is deprecated and will be removed in 2.0.  "
    "Import from picarones.reports_v2.html.views.pipeline instead.",
    DeprecationWarning,
    stacklevel=2,
)
