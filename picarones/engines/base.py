"""``picarones.engines.base`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.evaluation.engines.base`.  Phase 7.A
du retrait du legacy.  Cohabite avec
``picarones.adapters.ocr.base.BaseOCRAdapter`` (design canonique,
``StepExecutor``).
"""

from __future__ import annotations

import warnings

from picarones.evaluation.engines.base import *  # noqa: F401, F403
from picarones.evaluation.engines.base import (  # noqa: F401
    BaseOCREngine,
    EngineResult,
)

warnings.warn(
    "picarones.engines.base is deprecated and will be removed in 2.0.  "
    "Import from picarones.evaluation.engines.base instead.",
    DeprecationWarning,
    stacklevel=2,
)
