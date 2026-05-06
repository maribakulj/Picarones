"""``picarones.measurements.narrative`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.reports_v2.narrative`.
"""

from __future__ import annotations

import warnings

from picarones.reports_v2.narrative import (  # noqa: F401
    Fact,
    FactType,
    FactImportance,
    DetectorRegistry,
    detect_all,
    select_facts,
    render_fact,
    render_synthesis,
    extract_numbers,
    build_synthesis,
    register_default_detectors,
    DETECTORS_BY_TYPE,
)

# Privé ré-exporté pour rétrocompat des tests S19 qui le lisent.
from picarones.domain.facts import _DEFAULT_REGISTRY  # noqa: F401

warnings.warn(
    "picarones.measurements.narrative is deprecated and will be removed in 2.0.  "
    "Import from picarones.reports_v2.narrative instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ['Fact', 'FactType', 'FactImportance', 'DetectorRegistry', 'detect_all', 'select_facts', 'render_fact', 'render_synthesis', 'extract_numbers', 'build_synthesis', 'register_default_detectors', 'DETECTORS_BY_TYPE']
