"""``picarones.report.throughput_render`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.reports_v2.html.renderers.throughput`.
Phase 5.C du retrait du legacy.
"""

from __future__ import annotations

import warnings

from picarones.reports_v2.html.renderers.throughput import *  # noqa: F401, F403

warnings.warn(
    "picarones.report.throughput_render is deprecated and will be removed in 2.0.  "
    "Import from picarones.reports_v2.html.renderers.throughput instead.",
    DeprecationWarning,
    stacklevel=2,
)
