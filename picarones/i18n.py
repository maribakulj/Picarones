"""``picarones.i18n`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.reports.i18n`.  Phase 5.E du retrait
du legacy.
"""

from __future__ import annotations

import warnings

from picarones.reports.i18n import *  # noqa: F401, F403
from picarones.reports.i18n import (  # noqa: F401
    TRANSLATIONS,
    SUPPORTED_LANGS,
    get_labels,
    reload_translations,
)

warnings.warn(
    "picarones.i18n is deprecated and will be removed in 2.0.  "
    "Import from picarones.reports.i18n instead.",
    DeprecationWarning,
    stacklevel=2,
)
