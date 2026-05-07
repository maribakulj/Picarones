"""``picarones.measurements.pipeline_comparison`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.evaluation.pipeline_comparison`.
Phase 5.C.batch7 du retrait du legacy.
"""

from __future__ import annotations

import warnings

from picarones.pipeline.legacy_pipeline_comparison import *  # noqa: F401, F403

warnings.warn(
    "picarones.measurements.pipeline_comparison is deprecated and will be removed in 2.0.  "
    "Import from picarones.pipeline.legacy_pipeline_comparison instead.",
    DeprecationWarning,
    stacklevel=2,
)
