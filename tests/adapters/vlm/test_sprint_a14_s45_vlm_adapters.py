"""Sprint A14-S45 — VLM adapters (4 fournisseurs).

Tests des 4 adapters VLM qui héritent de ``BaseVLMAdapter`` +
leur LLM sibling (composition par MRO multiple).
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from picarones.adapters.vlm.base import VLMAdapterError
from picarones.adapters.vlm import (
    AnthropicVLMAdapter,
    BaseVLMAdapter,
    MistralVLMAdapter,
    OllamaVLMAdapter,
    OpenAIVLMAdapter,
)
from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.pipeline.types import RunContext


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


class _StubVLMAdapter(BaseVLMAdapter):
    """VLM stub pour tests : retourne un texte fixe."""

    def __init__(
        self,
        response_text="texte transcrit",
        raise_on_call=False,
        config=None,
    ):
        super().__init__(config=config or {"max_retries": 0})
        self._response = response_text
        self._raise = raise_on_call
        self.last_image_b64 = None

    @property
    def name(self) -> str:
        return "stub_vlm"

    @property
    def default_model(self) -> str:
        return "stub-vlm-1.0"

    def _call(self, prompt, image_b64=None):
        self.last_image_b64 = image_b64
        if self._raise:
            raise RuntimeError("VLM crashed")
        return self._response


def _make_image_artifact(uri: str) -> Artifact:
    return Artifact(
        id="doc01:image",
        document_id="doc01",
        type=ArtifactType.IMAGE,
        uri=uri,
    )


def _make_context() -> RunContext:
    return RunContext(
        document_id="doc01",
        code_version="1.0.0",
        pipeline_name="test",
    )


# ──────────────────────────────────────────────────────────────────────
# Contrat StepExecutor (BaseVLMAdapter)
# ──────────────────────────────────────────────────────────────────────


class TestBaseVLMAdapterContract:
    def test_input_types_is_image(self) -> None:
        adapter = _StubVLMAdapter()
        assert adapter.input_types == frozenset({ArtifactType.IMAGE})

    def test_output_types_is_raw_text(self) -> None:
        adapter = _StubVLMAdapter()
        assert adapter.output_types == frozenset({ArtifactType.RAW_TEXT})

    def test_execution_mode_is_io(self) -> None:
        # Hérité de BaseLLMAdapter.
        assert _StubVLMAdapter.execution_mode == "io"


class TestVLMExecuteNominal:
    def test_basic_transcription(self, tmp_path: Path) -> None:
        image_path = tmp_path / "doc01.png"
        image_path.write_bytes(b"PNGBYTES")
        adapter = _StubVLMAdapter(response_text="ceci est le texte")

        result = adapter.execute(
            inputs={ArtifactType.IMAGE: _make_image_artifact(str(image_path))},
            params={},
            context=_make_context(),
        )
        assert ArtifactType.RAW_TEXT in result
        produced = result[ArtifactType.RAW_TEXT]
        assert produced.type == ArtifactType.RAW_TEXT
        assert produced.document_id == "doc01"
        out_path = Path(produced.uri)
        assert out_path.exists()
        assert out_path.read_text(encoding="utf-8") == "ceci est le texte"
        assert out_path.name == "doc01.stub_vlm.txt"

    def test_image_passed_to_llm_as_base64(self, tmp_path: Path) -> None:
        image_path = tmp_path / "doc01.png"
        image_path.write_bytes(b"VLM_TEST_BYTES")
        adapter = _StubVLMAdapter()
        adapter.execute(
            inputs={ArtifactType.IMAGE: _make_image_artifact(str(image_path))},
            params={},
            context=_make_context(),
        )
        decoded = base64.b64decode(adapter.last_image_b64)
        assert decoded == b"VLM_TEST_BYTES"

    def test_artifact_id_uses_adapter_name(self, tmp_path: Path) -> None:
        image_path = tmp_path / "doc01.png"
        image_path.write_bytes(b"x")
        adapter = _StubVLMAdapter()
        result = adapter.execute(
            inputs={ArtifactType.IMAGE: _make_image_artifact(str(image_path))},
            params={},
            context=_make_context(),
        )
        produced = result[ArtifactType.RAW_TEXT]
        assert produced.id == "doc01:stub_vlm:raw_text"
        assert produced.produced_by_step == "vlm_transcription"

    def test_custom_transcription_prompt(self, tmp_path: Path) -> None:
        image_path = tmp_path / "doc01.png"
        image_path.write_bytes(b"x")
        adapter = _StubVLMAdapter(config={
            "max_retries": 0,
            "transcription_prompt": "Custom VLM prompt",
        })
        # On capture le prompt en surchargeant _call.
        captured = {}

        def _capture_call(prompt, image_b64=None):
            captured["prompt"] = prompt
            return "x"

        adapter._call = _capture_call  # type: ignore[method-assign]
        adapter.execute(
            inputs={ArtifactType.IMAGE: _make_image_artifact(str(image_path))},
            params={},
            context=_make_context(),
        )
        assert captured["prompt"] == "Custom VLM prompt"


# ──────────────────────────────────────────────────────────────────────
# Erreurs
# ──────────────────────────────────────────────────────────────────────


class TestVLMExecuteErrors:
    def test_missing_image_raises(self) -> None:
        adapter = _StubVLMAdapter()
        with pytest.raises(VLMAdapterError, match="IMAGE manquant"):
            adapter.execute(inputs={}, params={}, context=_make_context())

    def test_image_without_uri_raises(self) -> None:
        adapter = _StubVLMAdapter()
        artifact = Artifact(
            id="x",
            document_id="doc01",
            type=ArtifactType.IMAGE,
            uri=None,
        )
        with pytest.raises(VLMAdapterError, match="sans URI"):
            adapter.execute(
                inputs={ArtifactType.IMAGE: artifact},
                params={},
                context=_make_context(),
            )

    def test_image_path_not_existing_raises(self) -> None:
        adapter = _StubVLMAdapter()
        with pytest.raises(VLMAdapterError, match="introuvable"):
            adapter.execute(
                inputs={ArtifactType.IMAGE: _make_image_artifact(
                    "/nonexistent/img.png",
                )},
                params={},
                context=_make_context(),
            )

    def test_vlm_call_failing_raises(self, tmp_path: Path) -> None:
        image_path = tmp_path / "doc.png"
        image_path.write_bytes(b"x")
        adapter = _StubVLMAdapter(raise_on_call=True)
        with pytest.raises(VLMAdapterError, match="VLM a échoué"):
            adapter.execute(
                inputs={ArtifactType.IMAGE: _make_image_artifact(str(image_path))},
                params={},
                context=_make_context(),
            )


# ──────────────────────────────────────────────────────────────────────
# Adapters concrets — héritage MRO
# ──────────────────────────────────────────────────────────────────────


class TestConcreteVLMAdapters:
    @pytest.mark.parametrize("adapter_cls,expected_name", [
        (AnthropicVLMAdapter, "anthropic_vlm"),
        (OpenAIVLMAdapter, "openai_vlm"),
        (MistralVLMAdapter, "mistral_vlm"),
        (OllamaVLMAdapter, "ollama_vlm"),
    ])
    def test_adapter_name(self, adapter_cls, expected_name) -> None:
        adapter = adapter_cls()
        assert adapter.name == expected_name

    @pytest.mark.parametrize("adapter_cls", [
        AnthropicVLMAdapter,
        OpenAIVLMAdapter,
        MistralVLMAdapter,
        OllamaVLMAdapter,
    ])
    def test_adapter_input_types(self, adapter_cls) -> None:
        # input_types vient de BaseVLMAdapter par MRO.
        adapter = adapter_cls()
        assert adapter.input_types == frozenset({ArtifactType.IMAGE})

    @pytest.mark.parametrize("adapter_cls", [
        AnthropicVLMAdapter,
        OpenAIVLMAdapter,
        MistralVLMAdapter,
        OllamaVLMAdapter,
    ])
    def test_adapter_output_types(self, adapter_cls) -> None:
        adapter = adapter_cls()
        assert adapter.output_types == frozenset({ArtifactType.RAW_TEXT})

    @pytest.mark.parametrize("adapter_cls", [
        AnthropicVLMAdapter,
        OpenAIVLMAdapter,
        MistralVLMAdapter,
        OllamaVLMAdapter,
    ])
    def test_adapter_has_execute(self, adapter_cls) -> None:
        # execute() vient de BaseVLMAdapter par MRO.
        assert hasattr(adapter_cls, "execute")

    def test_mistral_default_model_is_pixtral(self) -> None:
        adapter = MistralVLMAdapter()
        assert "pixtral" in adapter.default_model.lower()

    def test_ollama_default_model_is_vision_capable(self) -> None:
        adapter = OllamaVLMAdapter()
        # Modèle par défaut doit être un modèle vision (llava family).
        assert "llava" in adapter.default_model.lower()


# ──────────────────────────────────────────────────────────────────────
# Intégration pipeline (utilisation comme StepExecutor)
# ──────────────────────────────────────────────────────────────────────


class TestVLMPipelineIntegration:
    def test_used_as_pipeline_step(self, tmp_path: Path) -> None:
        from picarones.pipeline.executor import PipelineExecutor
        from picarones.domain.pipeline_spec import PipelineSpec, PipelineStep
        from picarones.domain.documents import DocumentRef

        image_path = tmp_path / "doc01.png"
        image_path.write_bytes(b"PNG_BYTES")

        adapter = _StubVLMAdapter(response_text="VLM transcription")
        executor = PipelineExecutor(adapter_resolver=lambda name: adapter)
        spec = PipelineSpec(
            name="vlm_pipeline",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(
                PipelineStep(
                    id="vlm",
                    kind="vlm_transcription",
                    adapter_name="stub_vlm",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(ArtifactType.RAW_TEXT,),
                ),
            ),
        )
        result = executor.run(
            spec=spec,
            document=DocumentRef(id="doc01"),
            initial_inputs={
                ArtifactType.IMAGE: _make_image_artifact(str(image_path)),
            },
            context=_make_context(),
        )
        assert result.succeeded
        raw_text_artifacts = [
            a for a in result.artifacts
            if a.type == ArtifactType.RAW_TEXT
        ]
        assert len(raw_text_artifacts) == 1
        out_path = Path(raw_text_artifacts[0].uri)
        assert out_path.read_text(encoding="utf-8") == "VLM transcription"
