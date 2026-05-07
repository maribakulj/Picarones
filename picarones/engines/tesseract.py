"""``picarones.engines.tesseract`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.adapters.legacy_engines.tesseract`.  Phase 7.A
du retrait du legacy.
"""

from __future__ import annotations

import warnings

from picarones.adapters.legacy_engines.tesseract import *  # noqa: F401, F403

warnings.warn(
    "picarones.engines.tesseract is deprecated and will be removed in 2.0.  "
    "Import from picarones.adapters.legacy_engines.tesseract instead.",
    DeprecationWarning,
    stacklevel=2,
)
