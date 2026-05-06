"""``picarones.measurements.narrative.registry`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.reports_v2.narrative.registry`.
"""

from __future__ import annotations

import warnings

from picarones.reports_v2.narrative.registry import (  # noqa: F401
    DetectorEntry,
    register_detector,
    unregister,
    iter_detectors,
    detector_for,
    clear_registry,
    default_type_order,
    populate_legacy_registry,
    _REGISTRY,
    _REGISTRY_LOCK,
    _verify_unique_priorities,
)

warnings.warn(
    "picarones.measurements.narrative.registry is deprecated and will be removed in 2.0.  "
    "Import from picarones.reports_v2.narrative.registry instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ['DetectorEntry', 'register_detector', 'unregister', 'iter_detectors', 'detector_for', 'clear_registry', 'default_type_order', 'populate_legacy_registry']
