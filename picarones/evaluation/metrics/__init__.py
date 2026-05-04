"""Métriques — calculs purs sur des paires (référence, hypothèse).

Sprint A14-S10 : déplacement de **23 fichiers de calcul autonomes**
depuis ``picarones.measurements``.

Calculs de qualité textuelle pure :
  ``rare_tokens``, ``lexical_modernization``, ``calibration``,
  ``confusion``, ``line_metrics``.

Calculs structurels et géométriques :
  ``layout``, ``image_quality``, ``image_predictive``.

Calculs économiques :
  ``pricing``, ``marginal_cost``, ``throughput``,
  ``incremental_comparison``.

Calculs analytiques (post-traitement) :
  ``error_absorption``, ``hallucination``, ``robustness_projection``,
  ``longitudinal``, ``baseline_comparison``, ``levers``,
  ``worst_lines``, ``module_policy``.

Calculs inter-moteurs :
  ``inter_engine``, ``taxonomy_cooccurrence``,
  ``taxonomy_comparison``.

Reste à migrer (différé)
------------------------

Catégorie B — utilisent ``@register_metric`` du registre global
``core.metric_registry`` (singleton avec side-effect d'import) :
  ``mufi``, ``abbreviations``, ``unicode_blocks``, ``roman_numerals``,
  ``early_modern_typography``, ``modern_archives``, ``reading_order``,
  ``ner``, ``readability``, ``searchability``, ``numerical_sequences``.

Migrés au S20 quand le ``MetricRegistry`` instancié explicitement
(S5) deviendra le seul registre, via le ``registry_service``
applicatif.

Catégorie C — dépendances vers anciens packages :
  ``robustness`` (importe ``picarones.core.corpus`` +
  ``picarones.engines.base`` + ``picarones.measurements.metrics``).
  Ne peut être migré qu'après les Sprints S11 (déplacement des
  adapters) et S12 (équivalence numérique).

Catégorie D — dépendances inter-fichiers à orchestrer :
  ``cost_projection`` (→ pricing), ``equivalence_profile``
  (→ formats.text.normalization), ``specialization``
  (→ inter_engine), ``taxonomy_intra_doc`` (→ taxonomy),
  ``taxonomy`` (→ char_scores).

Règle de migration (S10) : un fichier déplacé = un commit avec
uniquement le déplacement et un re-export à l'ancien emplacement.
La logique reste identique.  Aucun test modifié.
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
