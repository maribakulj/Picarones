"""``picarones.report.views`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.reports_v2.html.views`.  Phase 5.D du
retrait du legacy.
"""

from __future__ import annotations

import warnings

from picarones.reports_v2.html.views import (  # noqa: F401
    build_advanced_taxonomy_view_html,
    build_diagnostics_view_html,
    build_economics_view_html,
    build_pipeline_view_html,
    build_robustness_view_html,
)

warnings.warn(
    "picarones.report.views is deprecated and will be removed in 2.0.  "
    "Import from picarones.reports_v2.html.views instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "build_advanced_taxonomy_view_html",
    "build_diagnostics_view_html",
    "build_economics_view_html",
    "build_pipeline_view_html",
    "build_robustness_view_html",
]
