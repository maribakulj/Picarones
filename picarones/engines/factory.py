"""``picarones.engines.factory`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.adapters.legacy_engines.factory`.  Phase 7.A
du retrait du legacy.
"""

from __future__ import annotations

import warnings

from picarones.adapters.legacy_engines.factory import *  # noqa: F401, F403

warnings.warn(
    "picarones.engines.factory is deprecated and will be removed in 2.0.  "
    "Import from picarones.adapters.legacy_engines.factory instead.",
    DeprecationWarning,
    stacklevel=2,
)
