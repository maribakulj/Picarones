"""``picarones.measurements.statistics`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.evaluation.statistics`.  Migration ::

    from picarones.evaluation.statistics import (
        bootstrap_ci, wilcoxon_test, friedman_test, ...
    )

Tous les symboles publics de l'API legacy (incluant les privés
``_SCIPY_AVAILABLE``, ``_chi_square_sf``, ``_nemenyi_critical_value``,
``_rank_row`` consommés par certains tests) restent accessibles
identiquement.
"""

from __future__ import annotations

import warnings

from picarones.evaluation.statistics import (
    _SCIPY_AVAILABLE,
    _chi_square_sf,
    _nemenyi_critical_value,
    _rank_row,
    ErrorCluster,
    bootstrap_ci,
    build_critical_difference_svg,
    cluster_errors,
    compute_correlation_matrix,
    compute_pairwise_stats,
    compute_pareto_front,
    compute_reliability_curve,
    compute_venn_data,
    friedman_test,
    nemenyi_posthoc,
    wilcoxon_test,
)

warnings.warn(
    "picarones.measurements.statistics is deprecated and will be "
    "removed in 2.0.  Import from picarones.evaluation.statistics instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "bootstrap_ci",
    "wilcoxon_test", "compute_pairwise_stats",
    "friedman_test", "nemenyi_posthoc", "build_critical_difference_svg",
    "compute_pareto_front",
    "ErrorCluster", "cluster_errors",
    "compute_correlation_matrix",
    "compute_reliability_curve", "compute_venn_data",
    "_SCIPY_AVAILABLE", "_chi_square_sf",
    "_nemenyi_critical_value", "_rank_row",
]
