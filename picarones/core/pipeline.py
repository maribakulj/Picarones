"""``picarones.core.pipeline`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.evaluation.pipeline`.  Phase 5.C.batch7
du retrait du legacy.
"""

from __future__ import annotations

import warnings

from picarones.evaluation.pipeline import *  # noqa: F401, F403

warnings.warn(
    "picarones.core.pipeline is deprecated and will be removed in 2.0.  "
    "Import from picarones.evaluation.pipeline instead.",
    DeprecationWarning,
    stacklevel=2,
)
