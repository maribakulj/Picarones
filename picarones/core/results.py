"""``picarones.core.results`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.evaluation.benchmark_result`.  Phase 4-ter
du retrait du legacy.
"""

from __future__ import annotations

import warnings

from picarones.evaluation.benchmark_result import *  # noqa: F401, F403
from picarones.evaluation.benchmark_result import (  # noqa: F401
    BenchmarkResult,
    DocumentResult,
    EngineReport,
)

warnings.warn(
    "picarones.core.results is deprecated and will be removed in 2.0.  "
    "Import from picarones.evaluation.benchmark_result instead.",
    DeprecationWarning,
    stacklevel=2,
)
