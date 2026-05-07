"""``picarones.engines.azure_doc_intel`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.adapters.legacy_engines.azure_doc_intel`.  Phase 7.A
du retrait du legacy.
"""

from __future__ import annotations

import warnings

from picarones.adapters.legacy_engines.azure_doc_intel import *  # noqa: F401, F403

warnings.warn(
    "picarones.engines.azure_doc_intel is deprecated and will be removed in 2.0.  "
    "Import from picarones.adapters.legacy_engines.azure_doc_intel instead.",
    DeprecationWarning,
    stacklevel=2,
)
