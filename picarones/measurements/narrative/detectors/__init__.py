"""``picarones.measurements.narrative.detectors`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.reports_v2.narrative.detectors`.
"""

from __future__ import annotations

import warnings

from picarones.reports_v2.narrative.detectors import (  # noqa: F401
    detect_global_leader_cer,
    detect_median_mean_gap_warning,
    detect_significant_gap,
    detect_speed_winner,
    detect_statistical_tie,
    detect_cost_outlier,
    detect_pareto_alternative,
    detect_pricing_staleness,
    detect_stratification_recommended,
    detect_stratum_collapse,
    detect_stratum_winner,
    detect_confidence_warning,
    detect_error_profile_outlier,
    detect_llm_hallucination_flag,
    detect_robustness_fragile,
    detect_engine_off_baseline,
    detect_engine_unstable,
    detect_importer_fallback,
    detect_regression_in_history,
    detect_ensemble_opportunity,
    DETECTORS_BY_TYPE,
    register_default_detectors,
    _build_detectors_by_type,
)

warnings.warn(
    "picarones.measurements.narrative.detectors is deprecated and will be removed in 2.0.  "
    "Import from picarones.reports_v2.narrative.detectors instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ['detect_global_leader_cer', 'detect_median_mean_gap_warning', 'detect_significant_gap', 'detect_speed_winner', 'detect_statistical_tie', 'detect_cost_outlier', 'detect_pareto_alternative', 'detect_pricing_staleness', 'detect_stratification_recommended', 'detect_stratum_collapse', 'detect_stratum_winner', 'detect_confidence_warning', 'detect_error_profile_outlier', 'detect_llm_hallucination_flag', 'detect_robustness_fragile', 'detect_engine_off_baseline', 'detect_engine_unstable', 'detect_importer_fallback', 'detect_regression_in_history', 'detect_ensemble_opportunity', 'DETECTORS_BY_TYPE', 'register_default_detectors']
