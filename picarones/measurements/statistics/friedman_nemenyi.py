"""``picarones.measurements.statistics.friedman_nemenyi`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.evaluation.statistics.friedman_nemenyi`.  Migration ::

    from picarones.evaluation.statistics import ...
"""

from __future__ import annotations

import warnings

from picarones.evaluation.statistics.friedman_nemenyi import (
    friedman_test,
    nemenyi_posthoc,
    _chi_square_sf,
    _nemenyi_critical_value,
    _rank_row,
)

warnings.warn(
    "picarones.measurements.statistics.friedman_nemenyi is deprecated and will be "
    "removed in 2.0.  Import from picarones.evaluation.statistics instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ['friedman_test', 'nemenyi_posthoc', '_chi_square_sf', '_nemenyi_critical_value', '_rank_row']
