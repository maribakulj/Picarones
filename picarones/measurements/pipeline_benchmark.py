"""``picarones.measurements.pipeline_benchmark`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.evaluation.pipeline_benchmark`.
Phase 5.C.batch7 du retrait du legacy.
"""

from __future__ import annotations

import warnings

from picarones.pipeline.legacy_pipeline_benchmark import *  # noqa: F401, F403

warnings.warn(
    "picarones.measurements.pipeline_benchmark is deprecated and will be removed in 2.0.  "
    "Import from picarones.pipeline.legacy_pipeline_benchmark instead.",
    DeprecationWarning,
    stacklevel=2,
)
