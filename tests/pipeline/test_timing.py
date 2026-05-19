"""Sprint A14-S7 — mesure de temps par étape.

Vérifie que ``StepResult.duration_seconds`` reflète le temps réel
d'exécution de l'adapter (pas zéro, pas négatif), et que la durée
totale est cohérente avec la somme des étapes.

Définition de done : pipeline mock en moins de 100 ms.
"""

from __future__ import annotations

import time

import pytest

from picarones.domain import Artifact, ArtifactType, DocumentRef
from picarones.pipeline import (
    PipelineExecutor,
    PipelineSpec,
    PipelineStep,
    RunContext,
)


class _SlowStub:
    """Adapter qui dort un certain temps avant de retourner."""

    def __init__(self, sleep_seconds: float) -> None:
        self._sleep = sleep_seconds

    name = "slow"
    input_types = frozenset({ArtifactType.IMAGE})
    output_types = frozenset({ArtifactType.RAW_TEXT})
    execution_mode = "cpu"

    def execute(self, inputs, params, context):
        time.sleep(self._sleep)
        return {
            ArtifactType.RAW_TEXT: Artifact(
                id=f"{context.document_id}:slow:raw_text",
                document_id=context.document_id,
                type=ArtifactType.RAW_TEXT,
                produced_by_step="slow",
            ),
        }


class _InstantStub:
    name = "instant"
    input_types = frozenset({ArtifactType.RAW_TEXT})
    output_types = frozenset({ArtifactType.CORRECTED_TEXT})
    execution_mode = "io"

    def execute(self, inputs, params, context):
        return {
            ArtifactType.CORRECTED_TEXT: Artifact(
                id=f"{context.document_id}:instant:corrected",
                document_id=context.document_id,
                type=ArtifactType.CORRECTED_TEXT,
                produced_by_step="instant",
            ),
        }


@pytest.fixture
def doc() -> DocumentRef:
    return DocumentRef(id="d1", image_uri="/tmp/x.png")


@pytest.fixture
def ctx() -> RunContext:
    return RunContext(
        document_id="d1", code_version="1.0.0", pipeline_name="timing",
    )


@pytest.fixture
def image_artifact() -> Artifact:
    return Artifact(
        id="d1:image", document_id="d1", type=ArtifactType.IMAGE,
        uri="/tmp/x.png",
    )


def _spec_two_steps() -> PipelineSpec:
    return PipelineSpec(
        name="timing",
        initial_inputs=(ArtifactType.IMAGE,),
        steps=(
            PipelineStep(
                id="slow", kind="ocr", adapter_name="slow",
                input_types=(ArtifactType.IMAGE,),
                output_types=(ArtifactType.RAW_TEXT,),
            ),
            PipelineStep(
                id="instant", kind="post_correction",
                adapter_name="instant",
                input_types=(ArtifactType.RAW_TEXT,),
                output_types=(ArtifactType.CORRECTED_TEXT,),
                inputs_from={ArtifactType.RAW_TEXT: "slow"},
            ),
        ),
    )


class TestExecutorTiming:
    def test_step_duration_reflects_sleep(
        self, doc, ctx, image_artifact,
    ) -> None:
        registry = {"slow": _SlowStub(0.05), "instant": _InstantStub()}
        executor = PipelineExecutor(adapter_resolver=lambda n: registry[n])

        result = executor.run(
            _spec_two_steps(), doc,
            {ArtifactType.IMAGE: image_artifact}, ctx,
        )
        assert result.succeeded
        slow_dur = result.step_result_by_id("slow").duration_seconds  # type: ignore[union-attr]
        # Marges larges pour absorber le bruit OS.
        assert 0.04 < slow_dur < 0.5

    def test_total_duration_at_least_sum_of_steps(
        self, doc, ctx, image_artifact,
    ) -> None:
        registry = {"slow": _SlowStub(0.02), "instant": _InstantStub()}
        executor = PipelineExecutor(adapter_resolver=lambda n: registry[n])

        result = executor.run(
            _spec_two_steps(), doc,
            {ArtifactType.IMAGE: image_artifact}, ctx,
        )
        sum_steps = sum(r.duration_seconds for r in result.step_results)
        # Le total inclut l'overhead orchestration → légèrement >.
        assert result.duration_seconds >= sum_steps - 0.01
        # Marge raisonnable pour ne pas exploser à cause du timing.
        assert result.duration_seconds < sum_steps + 0.5

    def test_duration_is_non_negative_even_on_failure(
        self, doc, ctx, image_artifact,
    ) -> None:
        class _Crasher:
            name = "crash"
            input_types = frozenset({ArtifactType.IMAGE})
            output_types = frozenset({ArtifactType.RAW_TEXT})
            execution_mode = "cpu"

            def execute(self, *a, **kw):
                raise RuntimeError("boom")

        registry = {"crash": _Crasher()}
        executor = PipelineExecutor(adapter_resolver=lambda n: registry[n])
        spec = PipelineSpec(
            name="crashing",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(
                PipelineStep(
                    id="bad", kind="ocr", adapter_name="crash",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(ArtifactType.RAW_TEXT,),
                ),
            ),
        )
        result = executor.run(
            spec, doc, {ArtifactType.IMAGE: image_artifact}, ctx,
        )
        assert not result.succeeded
        assert result.step_results[0].duration_seconds >= 0.0

    def test_def_of_done_under_100ms(
        self, doc, ctx, image_artifact,
    ) -> None:
        """Définition de done du S7 : pipeline mock en < 100ms."""
        registry = {
            "slow": _SlowStub(0.0),  # pas de sleep
            "instant": _InstantStub(),
        }
        executor = PipelineExecutor(adapter_resolver=lambda n: registry[n])

        t0 = time.perf_counter()
        result = executor.run(
            _spec_two_steps(), doc,
            {ArtifactType.IMAGE: image_artifact}, ctx,
        )
        elapsed = time.perf_counter() - t0

        assert result.succeeded
        # Marge généreuse pour la CI : 100ms est largement atteignable.
        assert elapsed < 0.1, f"trop lent : {elapsed * 1000:.2f}ms"
