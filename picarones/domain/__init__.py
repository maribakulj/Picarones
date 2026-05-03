"""Cercle 1 — Domain.

Types purs et abstractions du modèle métier de Picarones.

Ce cercle n'importe **que** la stdlib, ``pydantic`` et
``typing_extensions``.  Il ne dépend d'aucun moteur OCR, d'aucune
métrique calculée, d'aucun rendu, d'aucune couche réseau.

Objets centraux (à venir aux Sprints S4-S6) :

- ``Artifact`` / ``ArtifactType`` — toute sortie d'une étape de
  pipeline est un artefact traçable (id, type, hash, provenance).
- ``DocumentRef`` — référence à un document du corpus + ses GT
  multi-niveaux (TEXT, ALTO, PAGE, ENTITIES, READING_ORDER).
- ``CorpusSpec`` — description immuable d'un corpus.
- ``PipelineSpec`` / ``PipelineStep`` — DAG déclaratif d'une chaîne
  de transformation documentaire.
- ``EvaluationSpec`` / ``EvaluationView`` / ``ProjectionSpec`` /
  ``MetricSpec`` — contrats des vues d'évaluation.
- ``ProvenanceRecord`` — empreinte (timestamp, code_version,
  parameters_hash) attachée à chaque artefact.

Règle d'or : si tu hésites à mettre quelque chose ici, c'est qu'il
ne devrait pas y être.  Le domain ne fait presque rien.  Il décrit.

Voir ``docs/roadmap/rewrite-2026.md`` pour le contexte du Sprint S3.
"""

from __future__ import annotations

__all__: list[str] = []
