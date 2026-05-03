"""Cercle 1 — Domain.

Types purs et abstractions du modèle métier de Picarones.

Ce cercle n'importe **que** la stdlib, ``pydantic`` et
``typing_extensions``.  Il ne dépend d'aucun moteur OCR, d'aucune
métrique calculée, d'aucun rendu, d'aucune couche réseau.

API publique (S4)
-----------------

- ``Artifact`` / ``ArtifactType`` / ``compute_content_hash`` —
  toute sortie d'une étape de pipeline est un artefact traçable
  (id, type, hash, provenance).
- ``DocumentRef`` / ``GroundTruthRef`` — référence à un document
  du corpus + ses GT multi-niveaux.
- ``CorpusSpec`` — description immuable d'un corpus.
- ``ProvenanceRecord`` — empreinte (timestamp, code_version,
  parameters_hash) attachée à chaque artefact.
- ``PicaronesError`` (et sous-classes) — racine de la hiérarchie
  d'erreurs métier.

À venir aux Sprints S5-S6 :

- ``EvaluationSpec`` / ``EvaluationView`` / ``ProjectionSpec`` /
  ``MetricSpec`` — contrats des vues d'évaluation (S5).
- ``PipelineSpec`` / ``PipelineStep`` — DAG déclaratif d'une chaîne
  de transformation documentaire (S6).

Règle d'or : si tu hésites à mettre quelque chose ici, c'est qu'il
ne devrait pas y être.  Le domain ne fait presque rien.  Il décrit.

Voir ``docs/roadmap/rewrite-2026.md`` pour le plan complet.
"""

from __future__ import annotations

from picarones.domain.artifacts import Artifact, ArtifactType, compute_content_hash
from picarones.domain.corpus import CorpusSpec
from picarones.domain.documents import DocumentRef, GroundTruthRef
from picarones.domain.errors import (
    ArtifactValidationError,
    CorpusSpecError,
    PicaronesError,
    ProjectionError,
)
from picarones.domain.provenance import ProvenanceRecord

__all__ = [
    # Artifacts
    "Artifact",
    "ArtifactType",
    "compute_content_hash",
    # Corpus + documents
    "CorpusSpec",
    "DocumentRef",
    "GroundTruthRef",
    # Provenance
    "ProvenanceRecord",
    # Errors
    "PicaronesError",
    "ArtifactValidationError",
    "CorpusSpecError",
    "ProjectionError",
]
