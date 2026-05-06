"""Picarones — Cercle 1 : abstractions de domaine.

Ce package contient **uniquement des contrats** : dataclasses du
domaine, types d'artefacts, interfaces abstraites, registres, modèle
narratif. Pas de logique métier, pas de calcul, pas d'I/O.

Règle de dépendance : un module du cercle 1 peut importer un autre
module du cercle 1. Il ne peut **rien** importer des cercles 2 ou 3
(``measurements``, ``engines``, ``llm``, ``pipelines``, ``modules``,
``extras``, ``report``, ``cli``, ``web``).

Modules
-------
- :mod:`corpus`           Document, Corpus, GTLevel + payloads typés
- :mod:`modules`          BaseModule, ArtifactType
- :mod:`results`          DocumentResult, EngineReport, BenchmarkResult
- :mod:`metrics`          MetricsResult (dataclass), aggregate_metrics
- :mod:`metric_registry`  MetricSpec, register_metric, compute_at_junction
- :mod:`metric_hooks`     register_document_metric, register_corpus_aggregator
- :mod:`pipeline`         PipelineRunner, PipelineSpec, PipelineStep
- :mod:`facts`            Fact, FactType, FactImportance, DetectorRegistry

Voir :doc:`docs/explanation/architecture.md` pour le manifeste complet et
:doc:`docs/reference/api-stable.md` pour le contrat de stabilité de chaque
nom exporté.
"""
