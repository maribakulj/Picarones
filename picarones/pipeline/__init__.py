"""Cercle 2 — Pipeline execution.

Exécution séquentielle ou DAG-branchante d'une chaîne de modules
tiers (``StepExecutor``).  Picarones ne fournit **aucun module
métier** — l'utilisateur amène ses propres adapters OCR/LLM/VLM/
correcteur/reconstructeur ALTO ; le pipeline executor les compose,
valide les types aux jonctions et évalue automatiquement chaque
artefact produit contre la GT correspondante.

Modules livrés au S6
--------------------
- ``spec.py`` — ``PipelineStep``, ``PipelineSpec``, ``INITIAL_STEP_ID``.
  Spec déclarative sérialisable en YAML (cf. ``yaml_io.py``).
- ``types.py`` — ``RunContext``, ``StepResult``, ``PipelineResult``.
  Types runtime de l'executor.
- ``protocols.py`` — ``StepExecutor`` (Protocol), ``ExecutionMode``.
  Contrat d'un adapter exécutable.
- ``validation.py`` — ``validate_spec(spec, available_adapters)``,
  ``ValidationError``.  Validation statique sans instancier de module.
- ``yaml_io.py`` — ``dump_spec_to_yaml`` / ``load_spec_from_yaml``.

À venir aux Sprints S7-S8
-------------------------
- ``executor.py`` — ``PipelineExecutor.run(spec, document, inputs,
  context)`` exécute mono-document avec capture gracieuse des erreurs.
- ``runner.py`` — ``CorpusRunner`` orchestre l'executor sur un corpus
  complet avec **backpressure**, **timeout depuis le début
  d'exécution réelle**, **annulation propre**.
- ``cache.py`` — ``ArtifactCache`` indexé par
  ``hash(content + spec + code_version)``.

Cible du Sprint S12 : équivalence numérique CER/WER avec l'ancien
``measurements.runner`` à 1e-9 près sur les fixtures.
"""

from __future__ import annotations

from picarones.pipeline.protocols import ExecutionMode, StepExecutor
from picarones.pipeline.spec import INITIAL_STEP_ID, PipelineSpec, PipelineStep
from picarones.pipeline.types import PipelineResult, RunContext, StepResult
from picarones.pipeline.validation import ValidationError, validate_spec
from picarones.pipeline.yaml_io import dump_spec_to_yaml, load_spec_from_yaml

__all__ = [
    # Spec déclarative
    "PipelineSpec",
    "PipelineStep",
    "INITIAL_STEP_ID",
    # Runtime types
    "RunContext",
    "StepResult",
    "PipelineResult",
    # Protocol
    "StepExecutor",
    "ExecutionMode",
    # Validation
    "validate_spec",
    "ValidationError",
    # YAML IO
    "dump_spec_to_yaml",
    "load_spec_from_yaml",
]
