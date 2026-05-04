"""Cercle 1 — Domain.

Types purs et abstractions du modèle métier de Picarones.

Ce cercle n'importe **que** la stdlib, ``pydantic`` et
``typing_extensions``.  Il ne dépend d'aucun moteur OCR, d'aucune
métrique calculée, d'aucun rendu, d'aucune couche réseau.

API publique (S4 + S5)
----------------------

S4 — modèle de base :

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

S5 — contrats des vues d'évaluation :

- ``MetricSpec`` — déclaration d'une métrique (signature de types).
- ``EvaluationView`` — déclaration d'une vue (sélecteur + projection
  + métriques + dimensions ignorées).
- ``EvaluationSpec`` — container de N vues qu'un benchmark applique.
- ``ProjectionSpec`` — déclaration d'une projection entre types.

À venir au Sprint S6 :

- ``PipelineSpec`` / ``PipelineStep`` — DAG déclaratif d'une chaîne
  de transformation documentaire.

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
from picarones.domain.evaluation_spec import (
    EvaluationSpec,
    EvaluationView,
    MetricSpec,
)
from picarones.domain.projection_spec import ProjectionSpec
from picarones.domain.provenance import ProvenanceRecord
from picarones.domain.run_manifest import RunManifest, utcnow

# Note S26 — ``RunResult`` / ``RunDocumentResult`` ont été déplacés
# vers ``picarones.app.results`` car ils agrègent des objets de
# ``evaluation/`` et ``pipeline/`` (couches plus externes que
# ``domain``).  Le domain reste pur — il ne décrit que des contrats.

__all__ = [
    # S4 — Artifacts
    "Artifact",
    "ArtifactType",
    "compute_content_hash",
    # S4 — Corpus + documents
    "CorpusSpec",
    "DocumentRef",
    "GroundTruthRef",
    # S4 — Provenance
    "ProvenanceRecord",
    # S4 — Errors
    "PicaronesError",
    "ArtifactValidationError",
    "CorpusSpecError",
    "ProjectionError",
    # S5 — Evaluation contracts
    "MetricSpec",
    "EvaluationView",
    "EvaluationSpec",
    "ProjectionSpec",
    # S17 — Run manifest (pure domain ; RunResult vit dans app/)
    "RunManifest",
    "utcnow",
]
