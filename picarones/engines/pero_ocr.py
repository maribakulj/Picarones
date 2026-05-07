"""``picarones.engines.pero_ocr`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.adapters.legacy_engines.pero_ocr`.  Phase 7.A
du retrait du legacy.
"""

from __future__ import annotations

import warnings

from picarones.adapters.legacy_engines.pero_ocr import *  # noqa: F401, F403

warnings.warn(
    "picarones.engines.pero_ocr is deprecated and will be removed in 2.0.  "
    "Import from picarones.adapters.legacy_engines.pero_ocr instead.",
    DeprecationWarning,
    stacklevel=2,
)
