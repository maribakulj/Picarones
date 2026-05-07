"""``picarones.report.glossary`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.reports_v2.glossary`.  Phase 5 du retrait
du legacy.
"""

from __future__ import annotations

import warnings

from picarones.reports_v2.glossary import (  # noqa: F401
    SUPPORTED_LANGS,
    load_glossary,
)

warnings.warn(
    "picarones.report.glossary is deprecated and will be removed in 2.0.  "
    "Import from picarones.reports_v2.glossary instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["load_glossary", "SUPPORTED_LANGS"]
