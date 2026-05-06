"""``picarones.core.metric_registry`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.evaluation.metric_registry`.  Phase 4-ter
du retrait du legacy.
"""

from __future__ import annotations

import warnings

from picarones.evaluation.metric_registry import *  # noqa: F401, F403
from picarones.evaluation.metric_registry import (  # noqa: F401
    _METRIC_REGISTRY,
    _reset_registry_for_tests,
)

warnings.warn(
    "picarones.core.metric_registry is deprecated and will be removed in 2.0.  "
    "Import from picarones.evaluation.metric_registry instead.",
    DeprecationWarning,
    stacklevel=2,
)
