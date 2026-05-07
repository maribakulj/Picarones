"""``picarones.report.render_helpers`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.reports_v2._helpers.render_helpers`.
Phase 5 du retrait du legacy.
"""

from __future__ import annotations

import warnings

from picarones.reports_v2._helpers.render_helpers import *  # noqa: F401, F403

warnings.warn(
    "picarones.report.render_helpers is deprecated and will be removed in 2.0.  "
    "Import from picarones.reports_v2._helpers.render_helpers instead.",
    DeprecationWarning,
    stacklevel=2,
)
