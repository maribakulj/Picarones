"""``picarones.report.pipeline_dag_render`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.reports_v2.html.renderers.pipeline_dag`.
Phase 5.C du retrait du legacy.
"""

from __future__ import annotations

import warnings

from picarones.reports_v2.html.renderers.pipeline_dag import *  # noqa: F401, F403

warnings.warn(
    "picarones.report.pipeline_dag_render is deprecated and will be removed in 2.0.  "
    "Import from picarones.reports_v2.html.renderers.pipeline_dag instead.",
    DeprecationWarning,
    stacklevel=2,
)
