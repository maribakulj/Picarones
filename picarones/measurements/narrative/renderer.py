"""``picarones.measurements.narrative.renderer`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.reports_v2.narrative.renderer`.
"""

from __future__ import annotations

import warnings

from picarones.reports_v2.narrative.renderer import (  # noqa: F401
    logger,
    render_fact,
    render_synthesis,
    extract_numbers,
    _TEMPLATES_DIR,
    _TEMPLATES_CACHE,
    _load_templates,
    _SafeFormatMap,
)

warnings.warn(
    "picarones.measurements.narrative.renderer is deprecated and will be removed in 2.0.  "
    "Import from picarones.reports_v2.narrative.renderer instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ['logger', 'render_fact', 'render_synthesis', 'extract_numbers']
