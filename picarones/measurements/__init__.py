"""Métriques officielles Picarones — paquet legacy en cours de retrait.

Ce paquet, historiquement nommé « Cercle 2 — logique métier », est
progressivement vidé au profit du paquet canonique
:mod:`picarones.evaluation.metrics`.  Les modules qui restent ici ne
sont pas encore migrés (Catégorie B/C/D du plan de migration) :

Coeur (toujours legacy) :

- :mod:`metrics`              compute_metrics (CER/WER/MER/WIL via jiwer)
- :mod:`statistics`           Wilcoxon, Friedman, Nemenyi, Pareto, CDD
- :mod:`runner`               run_benchmark — orchestration parallèle
- :mod:`builtin_hooks`        12 hooks doc + 12 agrégateurs natifs
- :mod:`builtin_metrics`      enregistrement métriques dans le registry
- :mod:`alto_metrics`         métriques jonction TEXT/ALTO

Métriques philologiques (Catégorie B — register_metric singleton) :

- :mod:`mufi`, :mod:`abbreviations`, :mod:`unicode_blocks`,
  :mod:`early_modern_typography`, :mod:`modern_archives`,
  :mod:`reading_order`, :mod:`ner`, :mod:`readability`,
  :mod:`searchability`.

Câblages adaptifs (suffixe ``_hooks``) :

- :mod:`readability_hooks`, :mod:`searchability_hooks`,
  :mod:`numerical_sequences_hooks`, :mod:`philological_hooks`.

Auxiliaires :

- :mod:`equivalence_profile`, :mod:`reliability`, :mod:`history`,
  :mod:`robustness`.

Modules retirés (Lot D, mai 2026)
---------------------------------
Tous les shims qui ne faisaient que ré-exporter
``picarones.evaluation.metrics.X`` ont été supprimés en bloc :
``baseline_comparison``, ``calibration``, ``char_scores``,
``confusion``, ``cost_projection``, ``difficulty``,
``error_absorption``, ``hallucination``, ``image_predictive``,
``image_quality``, ``incremental_comparison``, ``inter_engine``,
``layout``, ``levers``, ``lexical_modernization``,
``line_metrics``, ``longitudinal``, ``marginal_cost``,
``module_policy``, ``ner_backends``, ``normalization``,
``numerical_sequences``, ``pricing``, ``rare_tokens``,
``robustness_projection``, ``roman_numerals``, ``specialization``,
``structure``, ``taxonomy``, ``taxonomy_comparison``,
``taxonomy_cooccurrence``, ``taxonomy_intra_doc``, ``throughput``,
``worst_lines``.  Importer désormais depuis
:mod:`picarones.evaluation.metrics`.

Moteur narratif :

- :mod:`narrative` (sous-package) : arbiter, registry, renderer,
  18 détecteurs en 6 familles. Le modèle de données (``Fact``,
  ``FactType``, ``DetectorRegistry``) vit en couche 1 dans
  :mod:`picarones.domain.facts`.

Voir :doc:`docs/explanation/architecture.md` pour la cartographie complète.
"""

# ──────────────────────────────────────────────────────────────────────────
# Cérémonie d'enregistrement des métriques typées dans le registre
# Sprint 34.  Tout consommateur qui veut utiliser ``compute_at_junction``
# (``picarones.evaluation.metric_registry``) doit avoir importé soit
# ``picarones.measurements`` soit ``picarones.evaluation.metrics`` au
# moins une fois pour que les décorateurs ``@register_metric`` aient
# été exécutés.
#
# Sans ces imports, ``compute_at_junction`` trouverait un registre vide
# et ne calculerait rien aux jonctions.
# ──────────────────────────────────────────────────────────────────────────

# Sprint 34 : cer / wer / mer / wil + stub TEXT→ALTO
from picarones.measurements import builtin_metrics  # noqa: F401
# Sprints 55-60 : métriques philologiques (Catégorie B — restent ici).
from picarones.measurements import abbreviations  # noqa: F401
from picarones.measurements import early_modern_typography  # noqa: F401
from picarones.measurements import modern_archives  # noqa: F401
from picarones.measurements import mufi  # noqa: F401
from picarones.measurements import unicode_blocks  # noqa: F401
# Sprint 53 : reading order F1.  Sprints 38, 52 : NER, readability.
from picarones.measurements import ner  # noqa: F401
from picarones.measurements import readability  # noqa: F401
from picarones.measurements import reading_order  # noqa: F401
# Chantier 1 (post-Sprint 97) : métriques (ALTO, ALTO) pour évaluer
# les reconstructeurs ALTO contre une GT ALTO du document.
from picarones.measurements import alto_metrics  # noqa: F401

# Lot D — les décorateurs ``@register_metric`` du paquet canonique
# ``picarones.evaluation.metrics`` sont exécutés dès cet import,
# garantissant que le registre Sprint 34 contient toutes les métriques
# canoniques sans avoir besoin des shims supprimés.
import picarones.evaluation.metrics  # noqa: F401

# Modules conservés en couche measurements (pas de shim canonique
# correspondant ; restent ici jusqu'à leur propre relocalisation).
from picarones.measurements import equivalence_profile  # noqa: F401  # curseur HTML
from picarones.measurements import reliability  # noqa: F401  # multi-runs

# Modules canoniques re-exposés pour rétrocompat de
# ``from picarones.measurements import roman_numerals`` (utilisé par
# d'anciens callers internes ; au prochain Lot, ils migreront vers
# ``picarones.evaluation.metrics.roman_numerals``).
from picarones.evaluation.metrics import roman_numerals  # noqa: F401
