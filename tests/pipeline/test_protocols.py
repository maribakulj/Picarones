"""Sprint A14-S6 — protocoles ``StepExecutor`` + types runtime.

Vérifie que :

- une classe minimale satisfait ``StepExecutor`` ;
- ``RunContext``, ``StepResult``, ``PipelineResult`` se construisent
  et sérialisent ;
- ``isinstance(x, StepExecutor)`` rejette les classes non-conformes.
"""

from __future__ import annotations

import pytest

from picarones.domain import Artifact, ArtifactType
from picarones.pipeline import (
    PipelineResult,
    RunContext,
    StepExecutor,
    StepResult,
)


# ──────────────────────────────────────────────────────────────────────
# RunContext
# ──────────────────────────────────────────────────────────────────────


class TestRunContext:
    def test_minimal_context(self) -> None:
        ctx = RunContext(
            document_id="d1",
            code_version="1.0.0",
            pipeline_name="ocr_only",
        )
        assert ctx.workspace_uri is None

    def test_with_workspace(self) -> None:
        ctx = RunContext(
            document_id="d1",
            code_version="1.0.0",
            pipeline_name="ocr_only",
            workspace_uri="/tmp/picarones/runs/abc",
        )
        assert ctx.workspace_uri == "/tmp/picarones/runs/abc"

    def test_frozen(self) -> None:
        from pydantic import ValidationError

        ctx = RunContext(document_id="d", code_version="v", pipeline_name="p")
        with pytest.raises(ValidationError):
            ctx.document_id = "x"  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────────────
# StepResult & PipelineResult
# ──────────────────────────────────────────────────────────────────────


class TestStepResult:
    def test_success(self) -> None:
        r = StepResult(
            step_id="ocr",
            succeeded=True,
            duration_seconds=2.5,
            produced_artifacts={"raw_text": "d1:ocr:raw_text"},
        )
        assert r.succeeded
        assert r.error is None

    def test_failure(self) -> None:
        r = StepResult(
            step_id="ocr",
            succeeded=False,
            duration_seconds=0.1,
            error="Tesseract introuvable",
        )
        assert not r.succeeded
        assert r.produced_artifacts == {}
        assert r.error == "Tesseract introuvable"

    def test_negative_duration_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            StepResult(step_id="x", succeeded=True, duration_seconds=-1.0)


class TestPipelineResult:
    def test_with_artifacts(self) -> None:
        a = Artifact(id="d1:ocr:raw_text", document_id="d1",
                     type=ArtifactType.RAW_TEXT)
        b = Artifact(id="d1:ocr:alto_xml", document_id="d1",
                     type=ArtifactType.ALTO_XML)
        result = PipelineResult(
            pipeline_name="ocr_only",
            document_id="d1",
            step_results=(
                StepResult(step_id="ocr", succeeded=True, duration_seconds=1.0,
                           produced_artifacts={
                               "raw_text": a.id, "alto_xml": b.id,
                           }),
            ),
            succeeded=True,
            duration_seconds=1.05,
            artifacts=(a, b),
        )
        assert result.step_result_by_id("ocr") is not None
        assert result.step_result_by_id("missing") is None
        text_arts = result.artifacts_of_type(ArtifactType.RAW_TEXT)
        assert len(text_arts) == 1
        assert text_arts[0].id == a.id


# ──────────────────────────────────────────────────────────────────────
# StepExecutor protocol
# ──────────────────────────────────────────────────────────────────────


class _StubExecutor:
    """Minimum pour satisfaire ``StepExecutor``."""

    name = "tesseract"
    input_types = frozenset({ArtifactType.IMAGE})
    output_types = frozenset({ArtifactType.RAW_TEXT})
    execution_mode = "cpu"

    def execute(
        self,
        inputs: dict[ArtifactType, Artifact],
        params: dict[str, str | int | float | bool],
        context: RunContext,
    ) -> dict[ArtifactType, Artifact]:
        # Vérifie la présence sans utiliser la valeur — l'appel a un
        # effet de bord en termes de validation des inputs.
        _ = inputs[ArtifactType.IMAGE]
        return {
            ArtifactType.RAW_TEXT: Artifact(
                id=f"{context.document_id}:tesseract:raw_text",
                document_id=context.document_id,
                type=ArtifactType.RAW_TEXT,
                produced_by_step="ocr",
            ),
        }


class TestStepExecutorProtocol:
    def test_stub_satisfies_protocol(self) -> None:
        ex = _StubExecutor()
        assert isinstance(ex, StepExecutor)

    def test_non_conforming_does_not_satisfy(self) -> None:
        class _NotAnExecutor:
            pass
        assert not isinstance(_NotAnExecutor(), StepExecutor)

    def test_stub_can_execute(self) -> None:
        ex = _StubExecutor()
        ctx = RunContext(document_id="d1", code_version="v", pipeline_name="p")
        img = Artifact(id="d1:img", document_id="d1", type=ArtifactType.IMAGE)
        out = ex.execute({ArtifactType.IMAGE: img}, {}, ctx)
        assert ArtifactType.RAW_TEXT in out
        assert out[ArtifactType.RAW_TEXT].document_id == "d1"
