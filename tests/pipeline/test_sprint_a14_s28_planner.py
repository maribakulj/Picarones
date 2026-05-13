"""Sprint A14-S28 — ``PipelinePlanner`` + ``ExecutionPlan``.

Tests du planner introduit par S28 pour transformer une
``PipelineSpec`` en plan d'exécution immuable consommé par
le ``PipelineExecutor.run_plan``.

Couvre :

1. ``PipelinePlanner.plan`` :
   - spec valide → ExecutionPlan avec resolved_steps + bindings ;
   - spec invalide → PlanningError avec liste d'erreurs ;
   - DAG branchant (inputs_from explicite) → bindings non implicites ;
   - validation d'adapters (set fourni) ;
   - validation d'adapters (None → skip).

2. Détection des jonctions de métriques :
   - sans MetricRegistry → metric_junctions = () ;
   - avec MetricRegistry → 1 junction par sortie de step ;
   - sortie sans métrique applicable → candidate_metrics = () ;
   - tri alphabétique déterministe des noms.

3. ``ExecutionPlan`` API :
   - frozen dataclass ;
   - step_by_id() ;
   - junctions_for_step().

4. Intégration avec ``PipelineExecutor`` :
   - run_plan(plan) consume un plan pré-calculé ;
   - run(spec) plan internement et exécute ;
   - executor.plan(spec) sucre.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.domain.documents import DocumentRef
from picarones.domain.evaluation_spec import MetricSpec
from picarones.evaluation.registry import MetricRegistry
from picarones.pipeline.executor import PipelineExecutor, PipelineSpecInvalid
from picarones.pipeline.planner import (
    ExecutionPlan,
    MetricJunction,
    PipelinePlanner,
    PlanningError,
    StepInputBinding,
)
from picarones.domain.pipeline_spec import (
    INITIAL_STEP_ID,
    PipelineSpec,
    PipelineStep,
)
from picarones.pipeline.types import RunContext


# ──────────────────────────────────────────────────────────────────────
# Stub adapter
# ──────────────────────────────────────────────────────────────────────


class _IdentityAdapter:
    """Adapter qui retourne directement ses inputs comme outputs."""

    name = "identity"
    input_types = frozenset()  # ne sert pas — l'executor lit step.input_types
    output_types = frozenset()
    execution_mode = "io"

    def execute(self, inputs, params, context):
        return {
            t: Artifact(
                id=f"{context.document_id}:{t.value}",
                document_id=context.document_id,
                type=t,
            )
            for t in inputs
        }


class _OCRStub:
    name = "ocr_stub"
    input_types = frozenset({ArtifactType.IMAGE})
    output_types = frozenset({ArtifactType.RAW_TEXT})
    execution_mode = "io"

    def execute(self, inputs, params, context):
        return {
            ArtifactType.RAW_TEXT: Artifact(
                id=f"{context.document_id}:raw",
                document_id=context.document_id,
                type=ArtifactType.RAW_TEXT,
            ),
        }


# ──────────────────────────────────────────────────────────────────────
# PipelinePlanner — validation
# ──────────────────────────────────────────────────────────────────────


class TestPipelinePlannerConstructor:
    def test_no_args(self) -> None:
        planner = PipelinePlanner()
        assert planner is not None

    def test_with_metric_registry(self) -> None:
        planner = PipelinePlanner(metric_registry=MetricRegistry())
        assert planner is not None

    def test_rejects_non_metric_registry(self) -> None:
        with pytest.raises(TypeError, match="metric_registry"):
            PipelinePlanner(metric_registry="nope")  # type: ignore[arg-type]

    def test_with_available_adapters(self) -> None:
        planner = PipelinePlanner(available_adapters={"adapter_a", "adapter_b"})
        assert planner is not None


class TestPipelinePlannerErrors:
    def test_empty_spec_raises_planning_error(self) -> None:
        spec = PipelineSpec(name="empty", steps=())
        planner = PipelinePlanner()
        with pytest.raises(PlanningError) as exc_info:
            planner.plan(spec)
        assert exc_info.value.errors
        assert exc_info.value.errors[0].code == "empty_pipeline"

    def test_unknown_adapter_raises_when_set_provided(self) -> None:
        spec = PipelineSpec(
            name="unknown_adapter",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(PipelineStep(
                id="s1",
                kind="ocr",
                adapter_name="not_in_registry",
                input_types=(ArtifactType.IMAGE,),
                output_types=(ArtifactType.RAW_TEXT,),
            ),),
        )
        planner = PipelinePlanner(available_adapters={"foo", "bar"})
        with pytest.raises(PlanningError) as exc_info:
            planner.plan(spec)
        assert any(
            e.code == "unknown_adapter" for e in exc_info.value.errors
        )

    def test_unknown_adapter_skipped_when_set_none(self) -> None:
        """Sans set d'adapters fourni, la validation est sautée."""
        spec = PipelineSpec(
            name="unknown_adapter",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(PipelineStep(
                id="s1",
                kind="ocr",
                adapter_name="any_name",
                input_types=(ArtifactType.IMAGE,),
                output_types=(ArtifactType.RAW_TEXT,),
            ),),
        )
        planner = PipelinePlanner()
        plan = planner.plan(spec)  # ne lève pas
        assert isinstance(plan, ExecutionPlan)

    def test_planning_error_carries_all_errors(self) -> None:
        """Le planner ne short-circuit pas — il récolte toutes les erreurs."""
        spec = PipelineSpec(
            name="multi_err",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(
                PipelineStep(
                    id="s1",
                    kind="ocr",
                    adapter_name="bad_a",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(ArtifactType.RAW_TEXT,),
                ),
                PipelineStep(
                    id="s1",  # duplicated id !
                    kind="other",
                    adapter_name="bad_b",
                    input_types=(ArtifactType.RAW_TEXT,),
                    output_types=(ArtifactType.CORRECTED_TEXT,),
                ),
            ),
        )
        planner = PipelinePlanner(available_adapters={"only_one"})
        with pytest.raises(PlanningError) as exc_info:
            planner.plan(spec)
        codes = {e.code for e in exc_info.value.errors}
        assert "duplicate_id" in codes
        assert "unknown_adapter" in codes


# ──────────────────────────────────────────────────────────────────────
# PipelinePlanner — résolution des bindings
# ──────────────────────────────────────────────────────────────────────


class TestPipelinePlannerBindings:
    def test_simple_chain_resolves_to_initial(self) -> None:
        spec = PipelineSpec(
            name="simple",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(PipelineStep(
                id="ocr",
                kind="ocr",
                adapter_name="ocr_stub",
                input_types=(ArtifactType.IMAGE,),
                output_types=(ArtifactType.RAW_TEXT,),
            ),),
        )
        plan = PipelinePlanner().plan(spec)
        assert len(plan.resolved_steps) == 1
        rs = plan.resolved_steps[0]
        assert rs.id == "ocr"
        assert len(rs.input_bindings) == 1
        binding = rs.input_bindings[0]
        assert binding.input_type == ArtifactType.IMAGE
        assert binding.source_step_id == INITIAL_STEP_ID

    def test_two_step_chain_resolves_to_previous(self) -> None:
        spec = PipelineSpec(
            name="two_step",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(
                PipelineStep(
                    id="ocr",
                    kind="ocr",
                    adapter_name="ocr_stub",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(ArtifactType.RAW_TEXT,),
                ),
                PipelineStep(
                    id="post",
                    kind="post_correction",
                    adapter_name="llm_corrector",
                    input_types=(ArtifactType.RAW_TEXT,),
                    output_types=(ArtifactType.CORRECTED_TEXT,),
                ),
            ),
        )
        plan = PipelinePlanner().plan(spec)
        assert len(plan.resolved_steps) == 2
        # 1er step : IMAGE depuis __initial__
        assert plan.resolved_steps[0].input_bindings[0].source_step_id == INITIAL_STEP_ID
        # 2e step : RAW_TEXT depuis le step "ocr"
        assert plan.resolved_steps[1].input_bindings[0].source_step_id == "ocr"

    def test_inputs_from_explicit_overrides_latest(self) -> None:
        """Si inputs_from désigne une étape antérieure non-récente,
        le binding doit pointer vers cette étape, pas vers le
        dernier producteur."""
        spec = PipelineSpec(
            name="explicit_dag",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(
                PipelineStep(
                    id="ocr_a",
                    kind="ocr",
                    adapter_name="ocr_a",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(ArtifactType.RAW_TEXT,),
                ),
                PipelineStep(
                    id="ocr_b",
                    kind="ocr",
                    adapter_name="ocr_b",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(ArtifactType.RAW_TEXT,),
                ),
                PipelineStep(
                    id="post_from_a",
                    kind="post_correction",
                    adapter_name="llm",
                    input_types=(ArtifactType.RAW_TEXT,),
                    output_types=(ArtifactType.CORRECTED_TEXT,),
                    # On veut explicitement le RAW_TEXT de ocr_a, pas ocr_b
                    # qui serait le « dernier producteur ».
                    inputs_from={ArtifactType.RAW_TEXT: "ocr_a"},
                ),
            ),
        )
        plan = PipelinePlanner().plan(spec)
        assert plan.resolved_steps[2].input_bindings[0].source_step_id == "ocr_a"

    def test_resolved_step_preserves_input_order(self) -> None:
        spec = PipelineSpec(
            name="multi_input",
            initial_inputs=(ArtifactType.IMAGE, ArtifactType.RAW_TEXT),
            steps=(PipelineStep(
                id="merge",
                kind="merge",
                adapter_name="m",
                input_types=(ArtifactType.IMAGE, ArtifactType.RAW_TEXT),
                output_types=(ArtifactType.CORRECTED_TEXT,),
            ),),
        )
        plan = PipelinePlanner().plan(spec)
        types = [b.input_type for b in plan.resolved_steps[0].input_bindings]
        assert types == [ArtifactType.IMAGE, ArtifactType.RAW_TEXT]


# ──────────────────────────────────────────────────────────────────────
# PipelinePlanner — détection des jonctions de métriques
# ──────────────────────────────────────────────────────────────────────


def _registry_with_text_metric() -> MetricRegistry:
    reg = MetricRegistry()
    reg.register(
        MetricSpec(
            name="cer",
            input_types=(ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT),
        ),
        lambda r, h: 0.0,
    )
    reg.register(
        MetricSpec(
            name="wer",
            input_types=(ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT),
        ),
        lambda r, h: 0.0,
    )
    return reg


class TestPipelinePlannerJunctions:
    def test_no_registry_means_empty_junctions(self) -> None:
        spec = PipelineSpec(
            name="x",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(PipelineStep(
                id="ocr", kind="ocr", adapter_name="ocr_stub",
                input_types=(ArtifactType.IMAGE,),
                output_types=(ArtifactType.RAW_TEXT,),
            ),),
        )
        plan = PipelinePlanner().plan(spec)
        assert plan.metric_junctions == ()

    def test_registry_yields_junctions_per_output(self) -> None:
        spec = PipelineSpec(
            name="x",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(PipelineStep(
                id="ocr", kind="ocr", adapter_name="ocr_stub",
                input_types=(ArtifactType.IMAGE,),
                output_types=(ArtifactType.RAW_TEXT,),
            ),),
        )
        plan = PipelinePlanner(
            metric_registry=_registry_with_text_metric(),
        ).plan(spec)
        assert len(plan.metric_junctions) == 1
        j = plan.metric_junctions[0]
        assert j.step_id == "ocr"
        assert j.artifact_type == ArtifactType.RAW_TEXT
        # Tri alphabétique déterministe
        assert j.candidate_metrics == ("cer", "wer")

    def test_output_without_metric_yields_empty_candidates(self) -> None:
        """Un type d'output sans métrique enregistrée donne tout de
        même une jonction (utile pour le diagnostic) avec
        candidate_metrics vide."""
        spec = PipelineSpec(
            name="x",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(PipelineStep(
                id="alto",
                kind="alto",
                adapter_name="alto_stub",
                input_types=(ArtifactType.IMAGE,),
                output_types=(ArtifactType.ALTO_XML,),
            ),),
        )
        plan = PipelinePlanner(
            metric_registry=_registry_with_text_metric(),
        ).plan(spec)
        assert len(plan.metric_junctions) == 1
        j = plan.metric_junctions[0]
        assert j.step_id == "alto"
        assert j.artifact_type == ArtifactType.ALTO_XML
        assert j.candidate_metrics == ()


# ──────────────────────────────────────────────────────────────────────
# ExecutionPlan API
# ──────────────────────────────────────────────────────────────────────


class TestExecutionPlanAPI:
    def test_step_by_id(self) -> None:
        spec = PipelineSpec(
            name="x",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(
                PipelineStep(
                    id="a", kind="ocr", adapter_name="x",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(ArtifactType.RAW_TEXT,),
                ),
                PipelineStep(
                    id="b", kind="post", adapter_name="y",
                    input_types=(ArtifactType.RAW_TEXT,),
                    output_types=(ArtifactType.CORRECTED_TEXT,),
                ),
            ),
        )
        plan = PipelinePlanner().plan(spec)
        a = plan.step_by_id("a")
        assert a is not None
        assert a.id == "a"
        assert plan.step_by_id("missing") is None

    def test_junctions_for_step(self) -> None:
        spec = PipelineSpec(
            name="x",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(
                PipelineStep(
                    id="ocr", kind="ocr", adapter_name="o",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(ArtifactType.RAW_TEXT,),
                ),
                PipelineStep(
                    id="post", kind="post", adapter_name="p",
                    input_types=(ArtifactType.RAW_TEXT,),
                    output_types=(ArtifactType.CORRECTED_TEXT,),
                ),
            ),
        )
        plan = PipelinePlanner(
            metric_registry=_registry_with_text_metric(),
        ).plan(spec)
        ocr_junctions = plan.junctions_for_step("ocr")
        assert len(ocr_junctions) == 1
        assert ocr_junctions[0].artifact_type == ArtifactType.RAW_TEXT
        assert plan.junctions_for_step("missing") == ()

    def test_dataclass_frozen(self) -> None:
        spec = PipelineSpec(
            name="x",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(PipelineStep(
                id="ocr", kind="ocr", adapter_name="o",
                input_types=(ArtifactType.IMAGE,),
                output_types=(ArtifactType.RAW_TEXT,),
            ),),
        )
        plan = PipelinePlanner().plan(spec)
        with pytest.raises(FrozenInstanceError):
            plan.spec = None  # type: ignore[misc]

    def test_step_input_binding_frozen(self) -> None:
        b = StepInputBinding(
            input_type=ArtifactType.IMAGE,
            source_step_id="x",
        )
        with pytest.raises(FrozenInstanceError):
            b.source_step_id = "y"  # type: ignore[misc]

    def test_resolved_step_frozen(self) -> None:
        spec = PipelineSpec(
            name="x",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(PipelineStep(
                id="s", kind="k", adapter_name="a",
                input_types=(ArtifactType.IMAGE,),
                output_types=(ArtifactType.RAW_TEXT,),
            ),),
        )
        plan = PipelinePlanner().plan(spec)
        rs = plan.resolved_steps[0]
        with pytest.raises(FrozenInstanceError):
            rs.step = None  # type: ignore[misc]

    def test_metric_junction_frozen(self) -> None:
        j = MetricJunction(
            step_id="x",
            artifact_type=ArtifactType.RAW_TEXT,
            candidate_metrics=("cer",),
        )
        with pytest.raises(FrozenInstanceError):
            j.candidate_metrics = ()  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────────────
# Intégration Planner + Executor
# ──────────────────────────────────────────────────────────────────────


class TestPipelineExecutorWithPlanner:
    def test_executor_plan_returns_execution_plan(self) -> None:
        spec = PipelineSpec(
            name="x",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(PipelineStep(
                id="ocr", kind="ocr", adapter_name="ocr_stub",
                input_types=(ArtifactType.IMAGE,),
                output_types=(ArtifactType.RAW_TEXT,),
            ),),
        )
        executor = PipelineExecutor(
            adapter_resolver=lambda n: _OCRStub(),
        )
        plan = executor.plan(spec)
        assert isinstance(plan, ExecutionPlan)
        assert len(plan.resolved_steps) == 1

    def test_executor_plan_raises_pipeline_spec_invalid_on_bad_spec(self) -> None:
        spec = PipelineSpec(name="bad", steps=())
        executor = PipelineExecutor(
            adapter_resolver=lambda n: _OCRStub(),
        )
        with pytest.raises(PipelineSpecInvalid, match="invalide"):
            executor.plan(spec)

    def test_run_plan_executes_pre_planned(self) -> None:
        spec = PipelineSpec(
            name="x",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(PipelineStep(
                id="ocr", kind="ocr", adapter_name="ocr_stub",
                input_types=(ArtifactType.IMAGE,),
                output_types=(ArtifactType.RAW_TEXT,),
            ),),
        )
        executor = PipelineExecutor(
            adapter_resolver=lambda n: _OCRStub(),
        )
        plan = executor.plan(spec)

        doc = DocumentRef(id="d1", image_uri="/tmp/img.png")
        ctx = RunContext(
            document_id="d1",
            code_version="1.0.0",
            pipeline_name="x",
        )
        result = executor.run_plan(
            plan=plan,
            document=doc,
            initial_inputs={
                ArtifactType.IMAGE: Artifact(
                    id="d1:img", document_id="d1", type=ArtifactType.IMAGE,
                ),
            },
            context=ctx,
        )
        assert result.succeeded
        assert len(result.step_results) == 1
        assert result.step_results[0].step_id == "ocr"

    def test_run_plan_rejects_non_plan(self) -> None:
        executor = PipelineExecutor(
            adapter_resolver=lambda n: _OCRStub(),
        )
        with pytest.raises(Exception, match="ExecutionPlan"):
            executor.run_plan(
                plan="not a plan",  # type: ignore[arg-type]
                document=DocumentRef(id="d1"),
                initial_inputs={},
                context=RunContext(
                    document_id="d1", code_version="1.0",
                    pipeline_name="x",
                ),
            )

    def test_run_spec_still_works_via_planning(self) -> None:
        """Sucre run(spec) — plan internement et exécute."""
        spec = PipelineSpec(
            name="x",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(PipelineStep(
                id="ocr", kind="ocr", adapter_name="ocr_stub",
                input_types=(ArtifactType.IMAGE,),
                output_types=(ArtifactType.RAW_TEXT,),
            ),),
        )
        executor = PipelineExecutor(
            adapter_resolver=lambda n: _OCRStub(),
        )
        doc = DocumentRef(id="d1", image_uri="/tmp/img.png")
        ctx = RunContext(
            document_id="d1",
            code_version="1.0.0",
            pipeline_name="x",
        )
        result = executor.run(
            spec=spec,
            document=doc,
            initial_inputs={
                ArtifactType.IMAGE: Artifact(
                    id="d1:img", document_id="d1", type=ArtifactType.IMAGE,
                ),
            },
            context=ctx,
        )
        assert result.succeeded

    def test_planner_injection(self) -> None:
        """Le caller peut injecter son propre planner (ex: avec
        MetricRegistry pour avoir les jonctions)."""
        custom_planner = PipelinePlanner(
            metric_registry=_registry_with_text_metric(),
        )
        executor = PipelineExecutor(
            adapter_resolver=lambda n: _OCRStub(),
            planner=custom_planner,
        )
        spec = PipelineSpec(
            name="x",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(PipelineStep(
                id="ocr", kind="ocr", adapter_name="ocr_stub",
                input_types=(ArtifactType.IMAGE,),
                output_types=(ArtifactType.RAW_TEXT,),
            ),),
        )
        plan = executor.plan(spec)
        assert plan.metric_junctions  # non vide grâce au registry injecté

    def test_planner_must_be_pipeline_planner(self) -> None:
        with pytest.raises(Exception, match="PipelinePlanner"):
            PipelineExecutor(
                adapter_resolver=lambda n: _OCRStub(),
                planner="not a planner",  # type: ignore[arg-type]
            )
