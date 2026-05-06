"""``picarones.measurements.statistics.cdd_render`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.evaluation.statistics.cdd_render`.  Migration ::

    from picarones.evaluation.statistics import ...
"""

from __future__ import annotations

import warnings

from picarones.evaluation.statistics.cdd_render import (
    build_critical_difference_svg,
)

warnings.warn(
    "picarones.measurements.statistics.cdd_render is deprecated and will be "
    "removed in 2.0.  Import from picarones.evaluation.statistics instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ['build_critical_difference_svg']
