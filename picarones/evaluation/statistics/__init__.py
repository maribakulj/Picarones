"""Tests statistiques pour Picarones — sous-package canonique.

Code mathématique pur (stdlib + scipy + numpy) consommé par les
détecteurs narratifs, le rendu HTML et la couche d'agrégation
inter-pipelines.

Familles
--------

- :mod:`bootstrap` — IC bootstrap par rééchantillonnage.
- :mod:`wilcoxon` — Test signé-rangé + matrice pairwise.
- :mod:`friedman_nemenyi` — Friedman multi-pipelines + post-hoc
  Nemenyi (calcul uniquement, pas de rendu).
- :mod:`cdd_render` — Rendu SVG du Critical Difference Diagram.
- :mod:`pareto` — Frontière de Pareto multi-objectifs.
- :mod:`clustering` — Regroupement des patterns d'erreur OCR/HTR.
- :mod:`correlation` — Matrice de corrélation entre métriques.
- :mod:`distributions` — Reliability curve et données Venn 2/3.

Migration Phase 2
-----------------

Migré depuis :mod:`picarones.measurements.statistics` qui devient
un shim re-export avec ``DeprecationWarning``.  Comportement
identique bit-for-bit (même seed pour le bootstrap, mêmes
algorithmes scipy, même rendu SVG).  Suppression du shim legacy
en version 2.0.

Les symboles privés ``_SCIPY_AVAILABLE``, ``_chi_square_sf``,
``_nemenyi_critical_value``, ``_rank_row`` sont également
ré-exportés car certains tests les consomment directement.
"""

from __future__ import annotations

from picarones.evaluation.statistics.bootstrap import bootstrap_ci
from picarones.evaluation.statistics.cdd_render import (
    build_critical_difference_svg,
)
from picarones.evaluation.statistics.clustering import (
    ErrorCluster,
    cluster_errors,
)
from picarones.evaluation.statistics.correlation import (
    compute_correlation_matrix,
)
from picarones.evaluation.statistics.distributions import (
    compute_reliability_curve,
    compute_venn_data,
)
from picarones.evaluation.statistics.friedman_nemenyi import (
    _chi_square_sf,
    _nemenyi_critical_value,
    _rank_row,
    friedman_test,
    nemenyi_posthoc,
)
from picarones.evaluation.statistics.pareto import compute_pareto_front
from picarones.evaluation.statistics.wilcoxon import (
    _SCIPY_AVAILABLE,
    _normal_sf,
    compute_pairwise_stats,
    wilcoxon_test,
)

__all__ = [
    # Bootstrap
    "bootstrap_ci",
    # Wilcoxon
    "wilcoxon_test",
    "compute_pairwise_stats",
    # Friedman / Nemenyi
    "friedman_test",
    "nemenyi_posthoc",
    "build_critical_difference_svg",
    # Pareto
    "compute_pareto_front",
    # Clustering
    "ErrorCluster",
    "cluster_errors",
    # Correlation
    "compute_correlation_matrix",
    # Distributions
    "compute_reliability_curve",
    "compute_venn_data",
    # Privés ré-exportés (consommés par certains tests)
    "_SCIPY_AVAILABLE",
    "_chi_square_sf",
    "_nemenyi_critical_value",
    "_normal_sf",
    "_rank_row",
]
