"""``picarones.report.numerical_sequences_render`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.reports_v2.html.renderers.numerical_sequences`.
Phase 5.C.batch7 du retrait du legacy.
"""

from __future__ import annotations

import warnings

from picarones.reports_v2.html.renderers.numerical_sequences import *  # noqa: F401, F403

warnings.warn(
    "picarones.report.numerical_sequences_render is deprecated and will be removed in 2.0.  "
    "Import from picarones.reports_v2.html.renderers.numerical_sequences instead.",
    DeprecationWarning,
    stacklevel=2,
)
