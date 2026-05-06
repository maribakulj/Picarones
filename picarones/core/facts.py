"""``picarones.core.facts`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.domain.facts`.  Migration ::

    from picarones.domain import Fact, FactType, FactImportance, DetectorRegistry
"""

from __future__ import annotations

import warnings

from picarones.domain.facts import (
    _DEFAULT_REGISTRY,  # noqa: F401  (lecture par tests S16 legacy)
    DetectorFn,
    DetectorRegistry,
    Fact,
    FactImportance,
    FactType,
    detect_all,
)

warnings.warn(
    "picarones.core.facts is deprecated and will be removed in 2.0.  "
    "Import from picarones.domain instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "DetectorFn", "DetectorRegistry", "Fact", "FactImportance",
    "FactType", "detect_all",
]
