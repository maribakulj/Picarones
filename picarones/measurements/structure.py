"""``picarones.measurements.structure`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.evaluation.metrics.structure`.
"""

from __future__ import annotations

import warnings

from picarones.evaluation.metrics.structure import (  # noqa: F401
    StructureResult,
    analyze_structure,
    aggregate_structure,
    _count_line_changes,
    _reading_order_score,
    _paragraph_conservation_score,
)

warnings.warn(
    "picarones.measurements.structure is deprecated and will be removed in 2.0.  "
    "Import from picarones.evaluation.metrics.structure instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ['StructureResult', 'analyze_structure', 'aggregate_structure']
