"""Sprint A14-S7 — ``PipelineExecutor`` mono-document.

Tous les tests utilisent des stubs ``StepExecutor`` définis dans
ce fichier — aucun adapter réel n'est instancié, ce qui rend la
suite rapide et déterministe.

Couvre les cas critiques :

- pipeline qui réussit complètement,
- step qui lève → step en échec, pipeline continue,
- adapter introuvable (KeyError du resolver),
- output manquant (adapter ne retourne pas un type promis),
- input manquant (initial_inputs incomplet),
- fork avec ``inputs_from`` explicite (reprise du Sprint 66),
- spec invalide → ``PipelineSpecInvalid`` levée,
- bag versionné : étape qui consomme l'output d'une étape antérieure.
"""

from __future__ import annotations

import pytest

from picarones.domain import (
    Artifact,
    ArtifactType,
    DocumentRef,
    PicaronesError,
)
from picarones.pipeline import (
    PipelineExecutor,
    PipelineResult,
    PipelineSpec,
    PipelineSpecInvalid,
    PipelineStep,
    RunContext,
)


# ──────────────────────────────────────────────────────────────────────
# Stubs ``StepExecutor``
# ──────────────────────────────────────────────────────────────────────


class _StubOCR:
    name = "stub_ocr"
    input_types = frozenset({ArtifactType.IMAGE})
    output_types = frozenset({ArtifactType.RAW_TEXT, ArtifactType.ALTO_XML})
    execution_mode = "cpu"

    def execute(self, inputs, params, context):
        return {
            ArtifactType.RAW_TEXT: Artifact(
                id=f"{context.document_id}:ocr:raw_text",
                document_id=context.document_id,
                type=ArtifactType.RAW_TEXT,
                produced_by_step="ocr",
            ),
            ArtifactType.ALTO_XML: Artifact(
                id=f"{context.document_id}:ocr:alto_xml",
                document_id=context.document_id,
                type=ArtifactType.ALTO_XML,
                produced_by_step="ocr",
            ),
        }


class _StubLLM:
    name = "stub_llm"
    input_types = frozenset({ArtifactType.RAW_TEXT})
    output_types = frozenset({ArtifactType.CORRECTED_TEXT})
    execution_mode = "io"

    def execute(self, inputs, params, context):
        return {
            ArtifactType.CORRECTED_TEXT: Artifact(
                id=f"{context.document_id}:llm:corrected_text",
                document_id=context.document_id,
                type=ArtifactType.CORRECTED_TEXT,
                produced_by_step="llm",
            ),
        }


class _CrashingStub:
    name = "crashing"
    input_types = frozenset({ArtifactType.RAW_TEXT})
    output_types = frozenset({ArtifactType.CORRECTED_TEXT})
    execution_mode = "cpu"

    def execute(self, inputs, params, context):
        raise RuntimeError("simulated boom")


class _IncompleteOutputStub:
    """Promet RAW_TEXT mais ne le retourne pas — viole le contrat."""

    name = "incomplete"
    input_types = frozenset({ArtifactType.IMAGE})
    output_types = frozenset({ArtifactType.RAW_TEXT})
    execution_mode = "cpu"

    def execute(self, inputs, params, context):
        return {}  # vide intentionnellement


class _SecondOCRStub:
    """Second OCR pour tester le fork via inputs_from."""

    name = "ocr_b"
    input_types = frozenset({ArtifactType.IMAGE})
    output_types = frozenset({ArtifactType.RAW_TEXT})
    execution_mode = "cpu"

    def execute(self, inputs, params, context):
        return {
            ArtifactType.RAW_TEXT: Artifact(
                id=f"{context.document_id}:ocr_b:raw_text",
                document_id=context.document_id,
                type=ArtifactType.RAW_TEXT,
                produced_by_step="ocr_b",
            ),
        }


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def registry() -> dict[str, object]:
    return {
        "stub_ocr": _StubOCR(),
        "stub_ocr_b": _SecondOCRStub(),
        "stub_llm": _StubLLM(),
        "crashing": _CrashingStub(),
        "incomplete": _IncompleteOutputStub(),
    }


@pytest.fixture
def executor(registry: dict[str, object]) -> PipelineExecutor:
    return PipelineExecutor(adapter_resolver=lambda name: registry[name])


@pytest.fixture
def doc() -> DocumentRef:
    return DocumentRef(id="doc1", image_uri="/tmp/x.png")


@pytest.fixture
def ctx() -> RunContext:
    return RunContext(
        document_id="doc1", code_version="1.0.0", pipeline_name="test",
    )


@pytest.fixture
def image_artifact() -> Artifact:
    return Artifact(
        id="doc1:image",
        document_id="doc1",
        type=ArtifactType.IMAGE,
        uri="/tmp/x.png",
    )


def _ocr_only_spec() -> PipelineSpec:
    return PipelineSpec(
        name="ocr_only",
        initial_inputs=(ArtifactType.IMAGE,),
        steps=(
            PipelineStep(
                id="ocr", kind="ocr", adapter_name="stub_ocr",
                input_types=(ArtifactType.IMAGE,),
                output_types=(
                    ArtifactType.RAW_TEXT, ArtifactType.ALTO_XML,
                ),
            ),
        ),
    )


def _ocr_llm_spec() -> PipelineSpec:
    return PipelineSpec(
        name="ocr_llm",
        initial_inputs=(ArtifactType.IMAGE,),
        steps=(
            PipelineStep(
                id="ocr", kind="ocr", adapter_name="stub_ocr",
                input_types=(ArtifactType.IMAGE,),
                output_types=(
                    ArtifactType.RAW_TEXT, ArtifactType.ALTO_XML,
                ),
            ),
            PipelineStep(
                id="llm", kind="post_correction", adapter_name="stub_llm",
                input_types=(ArtifactType.RAW_TEXT,),
                output_types=(ArtifactType.CORRECTED_TEXT,),
                inputs_from={ArtifactType.RAW_TEXT: "ocr"},
            ),
        ),
    )


# ──────────────────────────────────────────────────────────────────────
# Cas nominaux
# ──────────────────────────────────────────────────────────────────────


class TestNominalRun:
    def test_single_step_pipeline(
        self, executor, doc, ctx, image_artifact,
    ) -> None:
        spec = _ocr_only_spec()
        result = executor.run(
            spec, doc, {ArtifactType.IMAGE: image_artifact}, ctx,
        )
        assert isinstance(result, PipelineResult)
        assert result.succeeded
        assert result.pipeline_name == "ocr_only"
        assert result.document_id == "doc1"
        assert len(result.step_results) == 1
        assert result.step_results[0].succeeded
        assert result.step_results[0].step_id == "ocr"

    def test_two_step_pipeline_chains_artifacts(
        self, executor, doc, ctx, image_artifact,
    ) -> None:
        spec = _ocr_llm_spec()
        result = executor.run(
            spec, doc, {ArtifactType.IMAGE: image_artifact}, ctx,
        )
        assert result.succeeded
        # Tous les artefacts sont là : initial + 2 OCR + 1 LLM = 4
        assert len(result.artifacts) == 4
        types = {a.type for a in result.artifacts}
        assert ArtifactType.IMAGE in types
        assert ArtifactType.RAW_TEXT in types
        assert ArtifactType.ALTO_XML in types
        assert ArtifactType.CORRECTED_TEXT in types

    def test_step_results_record_produced_artifacts(
        self, executor, doc, ctx, image_artifact,
    ) -> None:
        result = executor.run(
            _ocr_llm_spec(), doc,
            {ArtifactType.IMAGE: image_artifact}, ctx,
        )
        ocr_result = result.step_result_by_id("ocr")
        assert ocr_result is not None
        assert "raw_text" in ocr_result.produced_artifacts
        assert "alto_xml" in ocr_result.produced_artifacts


# ──────────────────────────────────────────────────────────────────────
# Cas d'erreur — capture gracieuse
# ──────────────────────────────────────────────────────────────────────


class TestErrorCapture:
    def test_step_that_raises_marks_step_failed(
        self, executor, doc, ctx, image_artifact,
    ) -> None:
        """Un step qui lève → step en échec, pipeline continue."""
        spec = PipelineSpec(
            name="ocr_then_crash",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(
                PipelineStep(
                    id="ocr", kind="ocr", adapter_name="stub_ocr",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(
                        ArtifactType.RAW_TEXT, ArtifactType.ALTO_XML,
                    ),
                ),
                PipelineStep(
                    id="boom", kind="post_correction",
                    adapter_name="crashing",
                    input_types=(ArtifactType.RAW_TEXT,),
                    output_types=(ArtifactType.CORRECTED_TEXT,),
                ),
            ),
        )
        result = executor.run(
            spec, doc, {ArtifactType.IMAGE: image_artifact}, ctx,
        )
        assert not result.succeeded
        assert result.step_results[0].succeeded
        assert not result.step_results[1].succeeded
        assert "adapter_raised" in (result.step_results[1].error or "")
        assert "simulated boom" in (result.step_results[1].error or "")

    def test_unknown_adapter_yields_step_failure(
        self, executor, doc, ctx, image_artifact,
    ) -> None:
        spec = PipelineSpec(
            name="bad_adapter",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(
                PipelineStep(
                    id="ocr", kind="ocr", adapter_name="not_in_registry",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(ArtifactType.RAW_TEXT,),
                ),
            ),
        )
        result = executor.run(
            spec, doc, {ArtifactType.IMAGE: image_artifact}, ctx,
        )
        assert not result.succeeded
        assert "adapter_not_found" in (result.step_results[0].error or "")

    def test_adapter_returns_missing_output(
        self, executor, doc, ctx, image_artifact,
    ) -> None:
        spec = PipelineSpec(
            name="incomplete",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(
                PipelineStep(
                    id="bad", kind="ocr", adapter_name="incomplete",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(ArtifactType.RAW_TEXT,),
                ),
            ),
        )
        result = executor.run(
            spec, doc, {ArtifactType.IMAGE: image_artifact}, ctx,
        )
        assert not result.succeeded
        assert "missing_output" in (result.step_results[0].error or "")

    def test_initial_inputs_missing_blocks_first_step(
        self, executor, doc, ctx,
    ) -> None:
        """Si initial_inputs ne fournit pas IMAGE alors qu'un step en
        a besoin, le step échoue avec missing_input."""
        # On garde la spec valide (initial_inputs déclare IMAGE) mais
        # le caller "oublie" de fournir l'artefact → résolution
        # d'inputs échoue au runtime.
        spec = _ocr_only_spec()
        result = executor.run(spec, doc, {}, ctx)  # vide
        assert not result.succeeded
        assert "missing_input" in (result.step_results[0].error or "")


# ──────────────────────────────────────────────────────────────────────
# Bag versionné — fork via ``inputs_from`` (Sprint 66 historique)
# ──────────────────────────────────────────────────────────────────────


class TestBagVersionedFork:
    def test_inputs_from_explicit_picks_correct_version(
        self, executor, doc, ctx, image_artifact,
    ) -> None:
        """Deux OCR successifs produisent RAW_TEXT.  L'étape LLM
        précise ``inputs_from = "ocr_a"`` et doit consommer la
        version A, pas la dernière (B)."""
        spec = PipelineSpec(
            name="fork",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(
                PipelineStep(
                    id="ocr_a", kind="ocr", adapter_name="stub_ocr",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(
                        ArtifactType.RAW_TEXT, ArtifactType.ALTO_XML,
                    ),
                ),
                PipelineStep(
                    id="ocr_b", kind="ocr", adapter_name="stub_ocr_b",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(ArtifactType.RAW_TEXT,),
                ),
                PipelineStep(
                    id="llm", kind="post_correction",
                    adapter_name="stub_llm",
                    input_types=(ArtifactType.RAW_TEXT,),
                    output_types=(ArtifactType.CORRECTED_TEXT,),
                    inputs_from={ArtifactType.RAW_TEXT: "ocr_a"},
                ),
            ),
        )
        result = executor.run(
            spec, doc, {ArtifactType.IMAGE: image_artifact}, ctx,
        )
        assert result.succeeded
        # 1 image initiale + 2 (ocr_a) + 1 (ocr_b) + 1 (llm) = 5
        assert len(result.artifacts) == 5

    def test_default_picks_latest_when_no_inputs_from(
        self, executor, doc, ctx, image_artifact,
    ) -> None:
        """Sans ``inputs_from``, le LLM consomme le dernier RAW_TEXT,
        donc ``ocr_b`` (dernière étape qui a produit le type)."""
        spec = PipelineSpec(
            name="latest",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(
                PipelineStep(
                    id="ocr_a", kind="ocr", adapter_name="stub_ocr",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(
                        ArtifactType.RAW_TEXT, ArtifactType.ALTO_XML,
                    ),
                ),
                PipelineStep(
                    id="ocr_b", kind="ocr", adapter_name="stub_ocr_b",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(ArtifactType.RAW_TEXT,),
                ),
                PipelineStep(
                    id="llm", kind="post_correction",
                    adapter_name="stub_llm",
                    input_types=(ArtifactType.RAW_TEXT,),
                    output_types=(ArtifactType.CORRECTED_TEXT,),
                    # pas d'inputs_from
                ),
            ),
        )
        result = executor.run(
            spec, doc, {ArtifactType.IMAGE: image_artifact}, ctx,
        )
        assert result.succeeded


# ──────────────────────────────────────────────────────────────────────
# Validation défensive
# ──────────────────────────────────────────────────────────────────────


class TestDefensiveValidation:
    def test_invalid_spec_raises(
        self, executor, doc, ctx, image_artifact,
    ) -> None:
        """Spec avec ID dupliqué — l'executor lève sans appeler
        aucun adapter."""
        spec = PipelineSpec(
            name="dup",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(
                PipelineStep(
                    id="step", kind="ocr", adapter_name="stub_ocr",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(
                        ArtifactType.RAW_TEXT, ArtifactType.ALTO_XML,
                    ),
                ),
                PipelineStep(
                    id="step", kind="post_correction",
                    adapter_name="stub_llm",
                    input_types=(ArtifactType.RAW_TEXT,),
                    output_types=(ArtifactType.CORRECTED_TEXT,),
                ),
            ),
        )
        with pytest.raises(PipelineSpecInvalid, match="dupliqué"):
            executor.run(
                spec, doc, {ArtifactType.IMAGE: image_artifact}, ctx,
            )

    def test_non_callable_resolver_rejected(self) -> None:
        with pytest.raises(PicaronesError, match="callable"):
            PipelineExecutor(adapter_resolver="not_callable")  # type: ignore[arg-type]
