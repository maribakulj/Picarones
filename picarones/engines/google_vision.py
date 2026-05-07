"""``picarones.engines.google_vision`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.adapters.legacy_engines.google_vision`.  Phase 7.A
du retrait du legacy.
"""

from __future__ import annotations

import warnings

from picarones.adapters.legacy_engines.google_vision import *  # noqa: F401, F403

warnings.warn(
    "picarones.engines.google_vision is deprecated and will be removed in 2.0.  "
    "Import from picarones.adapters.legacy_engines.google_vision instead.",
    DeprecationWarning,
    stacklevel=2,
)
