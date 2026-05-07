"""``picarones.report.generator`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.reports_v2.html.generator`.  Phase 5.E
du retrait du legacy.
"""

from __future__ import annotations

import warnings

from picarones.reports_v2.html.generator import *  # noqa: F401, F403

warnings.warn(
    "picarones.report.generator is deprecated and will be removed in 2.0.  "
    "Import from picarones.reports_v2.html.generator instead.",
    DeprecationWarning,
    stacklevel=2,
)
