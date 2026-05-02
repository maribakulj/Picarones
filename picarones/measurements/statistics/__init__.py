"""Tests statistiques et clustering d'erreurs pour Picarones.

Avant le sprint « découpage de statistics.py » (2026-05-02) ce module
était un fichier unique de 1128 lignes mélangeant Wilcoxon, Friedman,
Nemenyi, bootstrap, Pareto, clustering, corrélation, courbes de
distribution et rendu SVG du Critical Difference Diagram.

Le sous-package éclate la responsabilité par famille statistique :

- :mod:`bootstrap` — IC bootstrap par rééchantillonnage.
- :mod:`wilcoxon` — Test signé-rangé + matrice pairwise.
- :mod:`friedman_nemenyi` — Friedman multi-moteurs + post-hoc Nemenyi
  (calcul uniquement, pas de rendu).
- :mod:`cdd_render` — Rendu SVG du Critical Difference Diagram.
- :mod:`pareto` — Frontière de Pareto multi-objectifs.
- :mod:`clustering` — Regroupement des patterns d'erreur OCR/HTR.
- :mod:`correlation` — Matrice de corrélation entre métriques.
- :mod:`distributions` — Reliability curve et données Venn 2/3.

Ce ``__init__.py`` ré-exporte toute l'API publique historique pour
que les ~30 fichiers qui importent depuis
``picarones.measurements.statistics`` continuent à fonctionner sans
modification. Les symboles privés ``_SCIPY_AVAILABLE``,
``_chi_square_sf``, ``_nemenyi_critical_value``, ``_rank_row`` sont
également ré-exportés car certains tests les consomment directement.
"""

from picarones.measurements.statistics.bootstrap import bootstrap_ci
from picarones.measurements.statistics.cdd_render import (
    build_critical_difference_svg,
)
from picarones.measurements.statistics.clustering import (
    ErrorCluster,
    cluster_errors,
)
from picarones.measurements.statistics.correlation import (
    compute_correlation_matrix,
)
from picarones.measurements.statistics.distributions import (
    compute_reliability_curve,
    compute_venn_data,
)
from picarones.measurements.statistics.friedman_nemenyi import (
    _chi_square_sf,
    _nemenyi_critical_value,
    _rank_row,
    friedman_test,
    nemenyi_posthoc,
)
from picarones.measurements.statistics.pareto import compute_pareto_front
from picarones.measurements.statistics.wilcoxon import (
    _SCIPY_AVAILABLE,
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
    "_rank_row",
]
