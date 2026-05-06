"""``picarones.core.metrics`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.evaluation.metric_result`.  Phase 4-ter
du retrait du legacy.
"""

from __future__ import annotations

import warnings

from picarones.evaluation.metric_result import *  # noqa: F401, F403

warnings.warn(
    "picarones.core.metrics is deprecated and will be removed in 2.0.  "
    "Import from picarones.evaluation.metric_result instead.",
    DeprecationWarning,
    stacklevel=2,
)
