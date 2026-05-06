"""``picarones.measurements.statistics.clustering`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.evaluation.statistics.clustering`.  Migration ::

    from picarones.evaluation.statistics import ...
"""

from __future__ import annotations

import warnings

from picarones.evaluation.statistics.clustering import (
    ErrorCluster,
    cluster_errors,
)

warnings.warn(
    "picarones.measurements.statistics.clustering is deprecated and will be "
    "removed in 2.0.  Import from picarones.evaluation.statistics instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ['ErrorCluster', 'cluster_errors']
