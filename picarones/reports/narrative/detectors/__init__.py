"""Détecteurs narratifs — package thématique (chantier 5 post-Sprint 97).

Avant le chantier 5, ce module était un fichier monolithe de 1229 lignes
(``narrative/detectors.py``) contenant 18 détecteurs. Pour aligner la
structure de code avec celle du registre déclaratif (Sprint 29), les
détecteurs ont été regroupés par **famille thématique** :

- :mod:`ranking`   — global leader, statistical tie, significant gap,
  speed winner, median/mean gap warning   (5 détecteurs)
- :mod:`pareto`    — Pareto alternative, cost outlier   (2 détecteurs)
- :mod:`stratum`   — stratum winner / collapse, stratification
  recommended   (3 détecteurs)
- :mod:`quality`   — error profile outlier, LLM hallucination flag,
  robustness fragile, confidence warning   (4 détecteurs)
- :mod:`history`   — engine off baseline, engine unstable, regression
  in history   (3 détecteurs)
- :mod:`ensemble`  — ensemble opportunity   (1 détecteur)

Total : 18 détecteurs (≠ "12" mentionné dans CLAUDE.md historique —
le chantier 5 corrige ce comptage).

Rétrocompatibilité absolue
--------------------------
Tous les noms exportés par l'ancien fichier ``detectors.py``
(``detect_*``, ``DETECTORS_BY_TYPE``, ``register_default_detectors``)
restent accessibles via ``from picarones.reports.narrative.detectors
import ...``. Les tests Sprints 20, 23, 29, 36, 44, 46, 73 importent
directement ces noms et continuent à fonctionner sans modification.

L'enregistrement automatique des détecteurs via ``@register_detector``
se fait à l'import de ce package — chaque sous-module est importé ici
en cascade.
"""

from __future__ import annotations

# Imports en cascade des 6 sous-modules : déclenche l'enregistrement
# automatique via les décorateurs ``@register_detector`` au chargement.
from picarones.reports.narrative.detectors.ranking import (
    detect_global_leader_cer,
    detect_median_mean_gap_warning,
    detect_significant_gap,
    detect_speed_winner,
    detect_statistical_tie,
)
from picarones.reports.narrative.detectors.pareto import (
    detect_cost_outlier,
    detect_pareto_alternative,
    detect_pricing_staleness,
)
from picarones.reports.narrative.detectors.stratum import (
    detect_stratification_recommended,
    detect_stratum_collapse,
    detect_stratum_winner,
)
from picarones.reports.narrative.detectors.quality import (
    detect_confidence_warning,
    detect_error_profile_outlier,
    detect_llm_hallucination_flag,
    detect_robustness_fragile,
)
from picarones.reports.narrative.detectors.history import (
    detect_engine_off_baseline,
    detect_engine_unstable,
    detect_importer_fallback,
    detect_regression_in_history,
)
from picarones.reports.narrative.detectors.ensemble import (
    detect_ensemble_opportunity,
)

# Snapshot du registre + helper d'enregistrement legacy — déplacés
# verbatim depuis l'ancien ``detectors.py`` (lignes 1193-1229).
from picarones.domain.facts import DetectorFn, FactType
from picarones.reports.narrative.registry import (
    iter_detectors as _iter_detectors,
    populate_detector_registry as _populate_detector_registry,
)


def _build_detectors_by_type() -> dict[FactType, DetectorFn]:
    """Snapshot du registre déclaratif vers un dict ``{type: fn}``."""
    return {entry.fact_type: entry.fn for entry in _iter_detectors()}


# Vue figée à l'import — utile pour les tests qui parcourent les types
# enregistrés sans instancier un ``DetectorRegistry``.
DETECTORS_BY_TYPE = _build_detectors_by_type()


def register_default_detectors(registry) -> None:
    """Enregistre les détecteurs du registre déclaratif dans un
    ``DetectorRegistry`` historique.

    Sprint 29 : la source de vérité est maintenant le décorateur
    ``@register_detector`` ; cette fonction se contente de pousser
    le contenu du registre vers l'objet ``DetectorRegistry`` que les
    consommateurs externes (``DetectorRegistry.run``) instancient.
    """
    _populate_detector_registry(registry)


__all__ = [
    # ranking
    "detect_global_leader_cer",
    "detect_median_mean_gap_warning",
    "detect_significant_gap",
    "detect_speed_winner",
    "detect_statistical_tie",
    # pareto
    "detect_cost_outlier",
    "detect_pareto_alternative",
    "detect_pricing_staleness",
    # stratum
    "detect_stratification_recommended",
    "detect_stratum_collapse",
    "detect_stratum_winner",
    # quality
    "detect_confidence_warning",
    "detect_error_profile_outlier",
    "detect_llm_hallucination_flag",
    "detect_robustness_fragile",
    # history
    "detect_engine_off_baseline",
    "detect_engine_unstable",
    "detect_importer_fallback",
    "detect_regression_in_history",
    # ensemble
    "detect_ensemble_opportunity",
    # legacy
    "DETECTORS_BY_TYPE",
    "register_default_detectors",
]
