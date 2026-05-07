"""``picarones.report`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.reports_v2.html`.  Phase 5.E du retrait
du legacy.
"""

from __future__ import annotations

import warnings

from picarones.reports_v2.html import ReportGenerator  # noqa: F401

warnings.warn(
    "picarones.report is deprecated and will be removed in 2.0.  "
    "Import ReportGenerator from picarones.reports_v2.html instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ReportGenerator"]
