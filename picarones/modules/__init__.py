"""``picarones.modules`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.adapters.legacy_modules`.  Phase 7.A.4
du retrait du legacy.
"""

from __future__ import annotations

import warnings

from picarones.adapters.legacy_modules.alto_text_to_mono_region import (
    TextToAltoMonoRegion,
)

warnings.warn(
    "picarones.modules is deprecated and will be removed in 2.0.  "
    "Import from picarones.adapters.legacy_modules instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["TextToAltoMonoRegion"]
