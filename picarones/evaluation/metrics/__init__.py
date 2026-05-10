"""Métriques — calculs purs sur des paires (référence, hypothèse).

~37 modules de calcul autonomes :

Calculs de qualité textuelle pure :
  ``rare_tokens``, ``lexical_modernization``, ``calibration``,
  ``confusion``, ``line_metrics``, ``text_metrics``.

Calculs structurels et géométriques :
  ``layout``, ``image_quality``, ``image_predictive``,
  ``alto_metrics``, ``alto_structural``.

Calculs économiques :
  ``pricing``, ``marginal_cost``, ``throughput``,
  ``incremental_comparison``, ``cost_projection``.

Calculs analytiques (post-traitement) :
  ``error_absorption``, ``hallucination``, ``robustness``,
  ``robustness_projection``, ``longitudinal``,
  ``baseline_comparison``, ``levers``, ``worst_lines``,
  ``module_policy``, ``history``, ``modern_archives``.

Calculs inter-moteurs :
  ``inter_engine``, ``specialization``, ``taxonomy``,
  ``taxonomy_intra_doc``, ``taxonomy_cooccurrence``,
  ``taxonomy_comparison``.

Calculs philologiques :
  ``mufi``, ``abbreviations``, ``unicode_blocks``,
  ``roman_numerals``, ``numerical_sequences``,
  ``early_modern_typography``, ``reading_order``.

Calculs sémantiques :
  ``ner``, ``readability``, ``searchability``,
  ``equivalence_profile``, ``over_normalization``.
"""

from __future__ import annotations

# Re-exports des 23 fichiers déplacés au S10.  Volontairement
# explicite (pas de wildcard import) pour qu'un caller du nouveau
# code ait une vue claire de ce qui est exposé.
from picarones.evaluation.metrics import (  # noqa: F401
    baseline_comparison,
    calibration,
    confusion,
    error_absorption,
    hallucination,
    image_predictive,
    image_quality,
    incremental_comparison,
    inter_engine,
    layout,
    levers,
    lexical_modernization,
    line_metrics,
    longitudinal,
    marginal_cost,
    module_policy,
    pricing,
    rare_tokens,
    robustness_projection,
    taxonomy_comparison,
    taxonomy_cooccurrence,
    throughput,
    worst_lines,
)

__all__ = [
    "baseline_comparison",
    "calibration",
    "confusion",
    "error_absorption",
    "hallucination",
    "image_predictive",
    "image_quality",
    "incremental_comparison",
    "inter_engine",
    "layout",
    "levers",
    "lexical_modernization",
    "line_metrics",
    "longitudinal",
    "marginal_cost",
    "module_policy",
    "pricing",
    "rare_tokens",
    "robustness_projection",
    "taxonomy_comparison",
    "taxonomy_cooccurrence",
    "throughput",
    "worst_lines",
]
