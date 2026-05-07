"""``picarones.modules.alto_text_to_mono_region`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.adapters.legacy_modules.alto_text_to_mono_region`.
Phase 7.A.4 du retrait du legacy.
"""

from __future__ import annotations

import warnings

from picarones.adapters.legacy_modules.alto_text_to_mono_region import *  # noqa: F401, F403

warnings.warn(
    "picarones.modules.alto_text_to_mono_region is deprecated and will be removed in 2.0.  "
    "Import from picarones.adapters.legacy_modules.alto_text_to_mono_region instead.",
    DeprecationWarning,
    stacklevel=2,
)
