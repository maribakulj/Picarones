"""Pont legacy ↔ canonique — Phase 7.B.3.

Helpers partagés entre :mod:`legacy_runner` (mono-document) et
:mod:`legacy_pipeline_benchmark` (corpus-wide) pour exécuter une
``PipelineSpec`` legacy via le ``PipelineExecutor`` canonique
:mod:`picarones.pipeline.executor` et reconstruire les types de
retour legacy (``PipelineResult``, ``StepResult``, dataclasses du
Sprint 63) attendus par les ~440 tests existants.

Pourquoi ce module
------------------
La sub-phase 7.B.2 avait introduit ces helpers en privé dans
:mod:`legacy_runner`.  La 7.B.3 doit faire que
:mod:`legacy_pipeline_benchmark` exécute lui-même les pipelines via
``PipelineExecutor.run_plan`` (au lieu de transiter par
``PipelineRunner.run`` du legacy_runner) — pour ça, les helpers
de traduction doivent être partageables.

L'API publique de ce module est strictement interne au package
``picarones.pipeline`` et sera supprimée en sub-phase 7.D, en même
temps que le runner legacy lui-même.

Anti-sur-ingénierie
-------------------
- Pas de cache de plan (le ``PipelinePlanner`` est instanciable et
  léger — chaque appel re-plan).
- Pas d'instance partagée d'``_PayloadRegistry`` entre documents :
  un registre par exécution de pipeline mono-doc, conforme au
  contrat de :class:`_BaseModuleAdapter`.
- Pas de provenance détaillée (``Artifact.provenance=None``) — le
  legacy ne portait pas cette info.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

from picarones.domain.artifacts import ArtifactType
from picarones.domain.documents import DocumentRef
from picarones.domain.pipeline_spec import (
    PipelineSpec as _DomainPipelineSpec,
    PipelineStep as _DomainPipelineStep,
)
from picarones.evaluation.corpus import Document, GTLevel
from picarones.evaluation.metric_registry import compute_at_junction
from picarones.pipeline._legacy_module_adapter import (
    _BaseModuleAdapter,
    _PayloadRegistry,
    wrap_initial_inputs,
)
from picarones.pipeline.executor import PipelineExecutor
from picarones.pipeline.types import (
    PipelineResult as _CanonicalPipelineResult,
    RunContext,
    StepResult as _CanonicalStepResult,
)

if TYPE_CHECKING:
    # Import paresseux pour éviter la dépendance cyclique
    # (legacy_runner importe ce module via les helpers,
    # ce module connaît ``PipelineSpec``/``PipelineStep`` legacy).
    from picarones.pipeline.legacy_runner import (
        PipelineResult as _LegacyPipelineResult,
        PipelineSpec as _LegacyPipelineSpec,
        PipelineStep as _LegacyPipelineStep,
        StepResult as _LegacyStepResult,
    )

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Conversion ArtifactType <-> GTLevel
# ──────────────────────────────────────────────────────────────────────────


_ARTIFACT_TO_GT_LEVEL: dict[ArtifactType, GTLevel] = {
    ArtifactType.RAW_TEXT: GTLevel.TEXT,
    ArtifactType.CORRECTED_TEXT: GTLevel.TEXT,
    ArtifactType.ALTO_XML: GTLevel.ALTO,
    ArtifactType.PAGE_XML: GTLevel.PAGE,
    ArtifactType.ENTITIES: GTLevel.ENTITIES,
    ArtifactType.READING_ORDER: GTLevel.READING_ORDER,
}


def artifact_type_to_gt_level(at: ArtifactType) -> Optional[GTLevel]:
    """Retourne le ``GTLevel`` correspondant à un ``ArtifactType``.

    ``IMAGE`` et les types pré-pipeline (``CONFIDENCES``, ``ALIGNMENT``,
    ``CANONICAL_DOCUMENT``) n'ont pas de niveau de GT direct.
    """
    return _ARTIFACT_TO_GT_LEVEL.get(at)


def gt_payload_to_value(payload: Any) -> Any:
    """Extrait la valeur exploitable d'un ``GTPayload`` typé.

    Pour ``TextGT`` on veut juste la chaîne ; pour les autres
    payloads on retourne le payload entier (la métrique sait quoi
    en faire selon sa signature de types).
    """
    from picarones.evaluation.corpus import (
        AltoGT, EntitiesGT, PageGT, ReadingOrderGT, TextGT,
    )
    if isinstance(payload, TextGT):
        return payload.text
    if isinstance(payload, EntitiesGT):
        return payload.entities
    if isinstance(payload, ReadingOrderGT):
        return payload.region_order
    if isinstance(payload, (AltoGT, PageGT)):
        return payload
    return payload


# ──────────────────────────────────────────────────────────────────────────
# Conversion spec legacy → spec canonique
# ──────────────────────────────────────────────────────────────────────────


def legacy_spec_to_canonical_spec(
    legacy_spec: "_LegacyPipelineSpec",
    initial_input_types: tuple[ArtifactType, ...],
) -> tuple[_DomainPipelineSpec, dict[str, _BaseModuleAdapter]]:
    """Convertit une ``PipelineSpec`` legacy en ``domain.PipelineSpec``.

    Retourne aussi un dict ``{step.name: _BaseModuleAdapter sans
    registry}`` — l'appelant doit injecter un ``_PayloadRegistry``
    par exécution mono-document avant d'utiliser les adapters.
    """
    canonical_steps: list[_DomainPipelineStep] = []
    adapter_factories: dict[str, _BaseModuleAdapter] = {}
    for step in legacy_spec.steps:
        canonical_steps.append(
            _DomainPipelineStep(
                id=step.name,
                kind="legacy_module",
                adapter_name=step.name,
                input_types=tuple(step.input_types),
                output_types=tuple(step.output_types),
                inputs_from=dict(step.inputs_from),
            ),
        )
        # Note : on construit l'adapter **sans** registry — l'appelant
        # devra créer le registry et le passer au moment de l'usage.
        # On stocke l'instance pour le mapping ; le registry lié à
        # cette instance reste à fournir.
        adapter_factories[step.name] = step.module  # type: ignore[assignment]
    canonical_spec = _DomainPipelineSpec(
        name=legacy_spec.name,
        initial_inputs=initial_input_types,
        steps=tuple(canonical_steps),
    )
    return canonical_spec, adapter_factories


def build_adapter_resolver(
    legacy_spec: "_LegacyPipelineSpec",
    registry: _PayloadRegistry,
):
    """Construit un ``adapter_resolver`` pour ``PipelineExecutor``.

    Pour chaque step legacy, fabrique un ``_BaseModuleAdapter``
    lié au registre fourni.  Le résolveur retourne l'adapter via
    ``__getitem__`` (lève ``KeyError`` si nom inconnu — ce qui est
    le comportement attendu par ``PipelineExecutor``).
    """
    adapter_map: dict[str, _BaseModuleAdapter] = {
        step.name: _BaseModuleAdapter(step.module, registry)
        for step in legacy_spec.steps
    }
    return adapter_map.__getitem__


# ──────────────────────────────────────────────────────────────────────────
# Exécution mono-document via le canonique
# ──────────────────────────────────────────────────────────────────────────


def execute_legacy_spec_via_canonical(
    legacy_spec: "_LegacyPipelineSpec",
    document: Document,
    initial_inputs: dict[ArtifactType, Any],
) -> tuple[_CanonicalPipelineResult, _PayloadRegistry]:
    """Exécute ``legacy_spec`` via :class:`PipelineExecutor`.

    Construit la ``domain.PipelineSpec`` canonique équivalente, un
    ``adapter_resolver`` ad-hoc qui mappe ``step.name →
    _BaseModuleAdapter``, et délègue à l'executor.  Retourne le
    ``PipelineResult`` canonique + le registre de payloads (dont le
    caller a besoin pour reconstruire les ``junction_metrics`` du
    contrat legacy).

    Mono-document.  Le caller corpus-wide
    (``legacy_pipeline_benchmark.run_pipeline_benchmark``) n'utilise
    PAS cette fonction : il a son propre flow qui plan une fois pour
    tout le corpus.
    """
    registry = _PayloadRegistry()
    canonical_inputs = wrap_initial_inputs(
        initial_inputs, registry, document.doc_id,
    )

    canonical_spec, _ = legacy_spec_to_canonical_spec(
        legacy_spec, tuple(initial_inputs.keys()),
    )
    resolver = build_adapter_resolver(legacy_spec, registry)

    document_ref = DocumentRef(id=document.doc_id)
    context = RunContext(
        document_id=document.doc_id,
        code_version="legacy_runner",
        pipeline_name=legacy_spec.name,
    )
    executor = PipelineExecutor(adapter_resolver=resolver)
    canonical_result = executor.run(
        canonical_spec, document_ref, canonical_inputs, context,
    )
    return canonical_result, registry


# ──────────────────────────────────────────────────────────────────────────
# Reconstruction des types legacy depuis le canonique
# ──────────────────────────────────────────────────────────────────────────


def translate_canonical_error(canonical_error: str | None) -> Optional[str]:
    """Traduit un message d'erreur canonique vers le format legacy.

    Le ``PipelineExecutor`` produit des messages structurés avec un
    préfixe (``adapter_raised:``, ``missing_input:``, ``missing_output:``,
    ``adapter_not_found:``).  Les tests legacy s'attendent à des
    messages français du Sprint 63 — on convertit pour préserver
    rétrocompat strict tant que la sub-phase 7.C n'a pas migré les
    tests.
    """
    if canonical_error is None:
        return None
    if canonical_error.startswith("adapter_raised: "):
        return canonical_error[len("adapter_raised: "):]
    if canonical_error.startswith("missing_input: "):
        miss = canonical_error[len("missing_input: "):]
        return f"entrée manquante : {miss}"
    if canonical_error.startswith("missing_output: "):
        miss_repr = canonical_error[len("missing_output: "):]
        miss = miss_repr.strip("[]").replace("'", "").replace(" ", "")
        return f"sortie manquante : {miss}"
    if canonical_error.startswith("adapter_not_found: "):
        adapter = canonical_error[len("adapter_not_found: "):]
        return f"adapter introuvable : {adapter}"
    if canonical_error.startswith("adapter_resolver_failed: "):
        msg = canonical_error[len("adapter_resolver_failed: "):]
        return f"résolution adapter échouée : {msg}"
    return canonical_error


def compute_junction_metrics_for_step(
    produced_at: list[ArtifactType],
    canonical_sr: _CanonicalStepResult,
    registry: _PayloadRegistry,
    document: Document,
) -> dict[str, dict[str, Any]]:
    """Calcule ``junction_metrics`` en post-traitant les outputs.

    Pour chaque ``ArtifactType`` produit, retrouve le payload via
    ``registry`` puis appelle
    ``compute_at_junction(gt, payload, (T, T))`` exactement comme le
    Sprint 63.  Les exceptions par jonction sont logguées et la
    jonction est silencieusement ignorée — comportement historique.
    """
    junction_metrics: dict[str, dict[str, Any]] = {}
    for at in produced_at:
        gt_level = artifact_type_to_gt_level(at)
        if gt_level is None:
            continue
        gt_payload = document.get_gt(gt_level)
        if gt_payload is None:
            continue
        artifact_id = canonical_sr.produced_artifacts.get(at.value)
        if artifact_id is None or artifact_id not in registry:
            continue
        payload = registry.get(artifact_id)
        try:
            metrics = compute_at_junction(
                gt_payload_to_value(gt_payload),
                payload,
                (at, at),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[legacy_translator] évaluation à la jonction %s "
                "a levé : %s",
                at.value, exc,
            )
            continue
        if metrics:
            junction_metrics[at.value] = metrics

    # Phase 4-bis : double-clé pour rétrocompat.
    from picarones.domain.artifacts import expand_legacy_keys
    expand_legacy_keys(junction_metrics)
    return junction_metrics


def build_legacy_step_result(
    legacy_step: "_LegacyPipelineStep",
    canonical_sr: _CanonicalStepResult,
    registry: _PayloadRegistry,
    document: Document,
) -> "_LegacyStepResult":
    """Reconstruit un ``StepResult`` legacy depuis le canonique."""
    from picarones.pipeline.legacy_runner import StepResult as _LegacyStepResult

    error = translate_canonical_error(canonical_sr.error)

    produced_at: list[ArtifactType] = []
    for type_value in canonical_sr.produced_artifacts:
        try:
            produced_at.append(ArtifactType(type_value))
        except ValueError:
            continue

    junction_metrics = compute_junction_metrics_for_step(
        produced_at, canonical_sr, registry, document,
    )

    return _LegacyStepResult(
        step_name=legacy_step.name,
        duration_seconds=canonical_sr.duration_seconds,
        output_types=tuple(produced_at),
        junction_metrics=junction_metrics,
        error=error,
    )


def build_legacy_pipeline_result(
    legacy_spec: "_LegacyPipelineSpec",
    document: Document,
    canonical_result: _CanonicalPipelineResult,
    registry: _PayloadRegistry,
) -> "_LegacyPipelineResult":
    """Reconstruit un ``PipelineResult`` legacy complet depuis le canonique.

    Itère sur les paires (step legacy, step result canonique) et
    délègue à :func:`build_legacy_step_result` pour chaque.
    """
    from picarones.pipeline.legacy_runner import PipelineResult as _LegacyPipelineResult

    result = _LegacyPipelineResult(
        pipeline_name=legacy_spec.name,
        doc_id=document.doc_id,
        total_duration_seconds=canonical_result.duration_seconds,
    )
    for legacy_step, canonical_sr in zip(
        legacy_spec.steps, canonical_result.step_results,
    ):
        result.steps.append(
            build_legacy_step_result(
                legacy_step, canonical_sr, registry, document,
            ),
        )
    return result


__all__ = [
    "artifact_type_to_gt_level",
    "build_adapter_resolver",
    "build_legacy_pipeline_result",
    "build_legacy_step_result",
    "compute_junction_metrics_for_step",
    "execute_legacy_spec_via_canonical",
    "gt_payload_to_value",
    "legacy_spec_to_canonical_spec",
    "translate_canonical_error",
]
