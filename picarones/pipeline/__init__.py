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

Modules livrés au S7
--------------------
- ``executor.py`` — ``PipelineExecutor.run(spec, document,
  initial_inputs, context)`` exécute mono-document avec capture
  gracieuse des erreurs et bag d'artefacts versionné.
  ``AdapterResolver`` type alias.
- ``cache.py`` — ``ArtifactCache`` minimal in-memory indexé par
  ``hash(content + spec + code_version)``.

Modules livrés au S8
--------------------
- ``runner.py`` — ``CorpusRunner`` orchestre ``PipelineExecutor``
  sur un corpus complet avec :

  * **backpressure** (``max_in_flight``, jamais plus de N futures
    en vol),
  * **timeout depuis le début d'exécution réelle** (pas depuis la
    submission au pool),
  * **annulation propre** via ``threading.Event``.

  ``CorpusRunResult`` agrège ``DocumentOutcome``, qui distingue
  ``succeeded`` / ``failed`` / ``timed_out`` / ``cancelled``.

Cible du Sprint S12
-------------------
Équivalence numérique CER/WER avec l'ancien
``measurements.runner`` à 1e-9 près sur les fixtures.
"""

from __future__ import annotations

from picarones.pipeline.cache import ArtifactCache
from picarones.pipeline.executor import (
    AdapterResolver,
    PipelineExecutor,
    PipelineSpecInvalid,
)
from picarones.pipeline.llm_pipeline_builder import (
    OCRLLMPipelineMode,
    make_ocr_llm_pipeline_spec,
)
from picarones.pipeline.planner import (
    ExecutionPlan,
    MetricJunction,
    PipelinePlanner,
    PlanningError,
    ResolvedStep,
    StepInputBinding,
)
from picarones.pipeline.protocols import ExecutionMode, StepExecutor
from picarones.pipeline.runner import (
    ContextFactory,
    CorpusRunResult,
    CorpusRunner,
    DocumentOutcome,
    InitialInputsFactory,
)
from picarones.domain.pipeline_spec import INITIAL_STEP_ID, PipelineSpec, PipelineStep
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
    # Executor (S7)
    "PipelineExecutor",
    "PipelineSpecInvalid",
    "AdapterResolver",
    # Builder OCR+LLM (Phase 6 volet 2)
    "make_ocr_llm_pipeline_spec",
    "OCRLLMPipelineMode",
    # Planner (S28)
    "PipelinePlanner",
    "PlanningError",
    "ExecutionPlan",
    "ResolvedStep",
    "StepInputBinding",
    "MetricJunction",
    # Cache (S7)
    "ArtifactCache",
    # CorpusRunner (S8)
    "CorpusRunner",
    "CorpusRunResult",
    "DocumentOutcome",
    "InitialInputsFactory",
    "ContextFactory",
]
