"""Sprint A14-S44 — ``BaseLLMAdapter`` implémente le contrat StepExecutor.

Tests de l'intégration native des 4 LLM adapters dans le pipeline :
``execute(inputs, params, context) -> dict[ArtifactType, Artifact]``
ajouté à ``BaseLLMAdapter`` (sans wrapper / sans shim).

Couvre :
1. ``BaseLLMAdapter.input_types`` / ``output_types`` / ``execution_mode``
2. ``execute`` lit RAW_TEXT, appelle ``complete``, écrit
   ``<stem>.<name>.corrected.txt``, retourne CORRECTED_TEXT.
3. Erreurs : RAW_TEXT manquant, sans URI, fichier inexistant,
   complete() en échec.
4. Image optionnelle : ``inputs[IMAGE]`` est encodée en base64 et
   passée au ``complete``.
5. Les 4 adapters concrets (Anthropic, Mistral, OpenAI, Ollama)
   héritent bien du contrat.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from picarones.adapters.llm.base import BaseLLMAdapter
from picarones.adapters.llm.base import LLMAdapterError
from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.pipeline.types import RunContext


# ──────────────────────────────────────────────────────────────────────
# Adapter de test concret
# ──────────────────────────────────────────────────────────────────────


class _StubLLMAdapter(BaseLLMAdapter):
    """LLM stub pour tester ``execute`` sans appeler une vraie API."""

    @property
    def name(self) -> str:
        return "stub_llm"

    @property
    def default_model(self) -> str:
        return "stub-model-1.0"

    def __init__(
        self,
        response_text: str = "TEXTE CORRIGÉ",
        raise_on_call: bool = False,
        model=None,
        config=None,
    ) -> None:
        super().__init__(model=model, config=config)
        self._response = response_text
        self._raise = raise_on_call
        self.last_prompt = None
        self.last_image_b64 = None

    def _call(self, prompt, image_b64=None):
        self.last_prompt = prompt
        self.last_image_b64 = image_b64
        if self._raise:
            raise RuntimeError("LLM crashed")
        return self._response


def _make_context() -> RunContext:
    return RunContext(
        document_id="doc01",
        code_version="1.0.0",
        pipeline_name="test",
    )


def _make_text_artifact(uri: str) -> Artifact:
    return Artifact(
        id="doc01:ocr:raw_text",
        document_id="doc01",
        type=ArtifactType.RAW_TEXT,
        uri=uri,
    )


def _make_image_artifact(uri: str) -> Artifact:
    return Artifact(
        id="doc01:image",
        document_id="doc01",
        type=ArtifactType.IMAGE,
        uri=uri,
    )


# ──────────────────────────────────────────────────────────────────────
# Contract StepExecutor
# ──────────────────────────────────────────────────────────────────────


class TestBaseLLMAdapterContract:
    def test_input_types_default_raw_text(self) -> None:
        adapter = _StubLLMAdapter()
        assert ArtifactType.RAW_TEXT in adapter.input_types

    def test_output_types_default_corrected_text(self) -> None:
        adapter = _StubLLMAdapter()
        assert ArtifactType.CORRECTED_TEXT in adapter.output_types

    def test_execution_mode_default_io(self) -> None:
        # Class attribute, pas instance.
        assert BaseLLMAdapter.execution_mode == "io"


# ──────────────────────────────────────────────────────────────────────
# execute() — chemin nominal
# ──────────────────────────────────────────────────────────────────────


class TestLLMExecuteNominal:
    def test_basic_correction(self, tmp_path: Path) -> None:
        text_path = tmp_path / "doc01.txt"
        text_path.write_text("texte avec erreurs", encoding="utf-8")

        adapter = _StubLLMAdapter(response_text="texte sans erreurs")
        result = adapter.execute(
            inputs={ArtifactType.RAW_TEXT: _make_text_artifact(str(text_path))},
            params={},
            context=_make_context(),
        )
        assert ArtifactType.CORRECTED_TEXT in result
        produced = result[ArtifactType.CORRECTED_TEXT]
        assert produced.type == ArtifactType.CORRECTED_TEXT
        assert produced.document_id == "doc01"

        out_path = Path(produced.uri)
        assert out_path.exists()
        assert out_path.read_text(encoding="utf-8") == "texte sans erreurs"
        assert out_path.name == "doc01.stub_llm.corrected.txt"

    def test_artifact_id_uses_adapter_name(self, tmp_path: Path) -> None:
        text_path = tmp_path / "doc01.txt"
        text_path.write_text("x", encoding="utf-8")
        adapter = _StubLLMAdapter()
        result = adapter.execute(
            inputs={ArtifactType.RAW_TEXT: _make_text_artifact(str(text_path))},
            params={},
            context=_make_context(),
        )
        produced = result[ArtifactType.CORRECTED_TEXT]
        assert produced.id == "doc01:stub_llm:corrected_text"
        assert produced.produced_by_step == "post_correction"

    def test_prompt_template_formatted_with_text(self, tmp_path: Path) -> None:
        text_path = tmp_path / "doc01.txt"
        text_path.write_text("input text", encoding="utf-8")
        adapter = _StubLLMAdapter()
        adapter.execute(
            inputs={ArtifactType.RAW_TEXT: _make_text_artifact(str(text_path))},
            params={},
            context=_make_context(),
        )
        # Le prompt doit contenir le texte d'entrée.
        assert "input text" in adapter.last_prompt

    def test_custom_prompt_via_config(self, tmp_path: Path) -> None:
        text_path = tmp_path / "doc01.txt"
        text_path.write_text("input", encoding="utf-8")
        adapter = _StubLLMAdapter(config={
            "correction_prompt": "Custom: {text}",
        })
        adapter.execute(
            inputs={ArtifactType.RAW_TEXT: _make_text_artifact(str(text_path))},
            params={},
            context=_make_context(),
        )
        assert adapter.last_prompt == "Custom: input"


# ──────────────────────────────────────────────────────────────────────
# Erreurs
# ──────────────────────────────────────────────────────────────────────


class TestLLMExecuteErrors:
    def test_missing_raw_text_raises(self) -> None:
        adapter = _StubLLMAdapter()
        with pytest.raises(LLMAdapterError, match="RAW_TEXT manquant"):
            adapter.execute(
                inputs={},
                params={},
                context=_make_context(),
            )

    def test_text_artifact_without_uri_raises(self) -> None:
        adapter = _StubLLMAdapter()
        artifact = Artifact(
            id="x",
            document_id="doc01",
            type=ArtifactType.RAW_TEXT,
            uri=None,
        )
        with pytest.raises(LLMAdapterError, match="sans URI"):
            adapter.execute(
                inputs={ArtifactType.RAW_TEXT: artifact},
                params={},
                context=_make_context(),
            )

    def test_text_path_not_existing_raises(self) -> None:
        adapter = _StubLLMAdapter()
        with pytest.raises(LLMAdapterError, match="introuvable"):
            adapter.execute(
                inputs={ArtifactType.RAW_TEXT: _make_text_artifact(
                    "/nonexistent/x.txt",
                )},
                params={},
                context=_make_context(),
            )

    def test_llm_call_failing_raises(self, tmp_path: Path) -> None:
        text_path = tmp_path / "x.txt"
        text_path.write_text("x", encoding="utf-8")
        adapter = _StubLLMAdapter(raise_on_call=True, config={
            "max_retries": 0,  # pas de retry pour accélérer le test
        })
        with pytest.raises(LLMAdapterError, match="LLM a échoué"):
            adapter.execute(
                inputs={ArtifactType.RAW_TEXT: _make_text_artifact(str(text_path))},
                params={},
                context=_make_context(),
            )


# ──────────────────────────────────────────────────────────────────────
# Image optionnelle (mode VLM)
# ──────────────────────────────────────────────────────────────────────


class TestLLMExecuteWithImage:
    def test_image_passed_to_llm_as_base64(self, tmp_path: Path) -> None:
        text_path = tmp_path / "doc.txt"
        text_path.write_text("x", encoding="utf-8")
        image_path = tmp_path / "doc.png"
        image_path.write_bytes(b"PNGBYTES")

        adapter = _StubLLMAdapter()
        adapter.execute(
            inputs={
                ArtifactType.RAW_TEXT: _make_text_artifact(str(text_path)),
                ArtifactType.IMAGE: _make_image_artifact(str(image_path)),
            },
            params={},
            context=_make_context(),
        )
        # L'image doit être encodée en base64.
        assert adapter.last_image_b64 is not None
        decoded = base64.b64decode(adapter.last_image_b64)
        assert decoded == b"PNGBYTES"

    def test_image_omitted_when_not_provided(self, tmp_path: Path) -> None:
        text_path = tmp_path / "doc.txt"
        text_path.write_text("x", encoding="utf-8")
        adapter = _StubLLMAdapter()
        adapter.execute(
            inputs={ArtifactType.RAW_TEXT: _make_text_artifact(str(text_path))},
            params={},
            context=_make_context(),
        )
        assert adapter.last_image_b64 is None


# ──────────────────────────────────────────────────────────────────────
# Adapters concrets héritent du contrat
# ──────────────────────────────────────────────────────────────────────


class TestConcreteAdaptersInheritContract:
    def test_openai_has_execute(self) -> None:
        from picarones.adapters.llm.openai_adapter import OpenAIAdapter
        # Vérifie que la méthode execute est héritée.
        assert hasattr(OpenAIAdapter, "execute")
        assert hasattr(OpenAIAdapter, "input_types")
        assert hasattr(OpenAIAdapter, "output_types")

    def test_anthropic_has_execute(self) -> None:
        from picarones.adapters.llm.anthropic_adapter import AnthropicAdapter
        assert hasattr(AnthropicAdapter, "execute")

    def test_mistral_has_execute(self) -> None:
        from picarones.adapters.llm.mistral_adapter import MistralAdapter
        assert hasattr(MistralAdapter, "execute")

    def test_ollama_has_execute(self) -> None:
        from picarones.adapters.llm.ollama_adapter import OllamaAdapter
        assert hasattr(OllamaAdapter, "execute")


# ──────────────────────────────────────────────────────────────────────
# Intégration pipeline (utilisation comme StepExecutor)
# ──────────────────────────────────────────────────────────────────────


class TestPipelineIntegration:
    def test_used_as_pipeline_step(self, tmp_path: Path) -> None:
        """Un adapter LLM se branche directement comme step de pipeline."""
        from picarones.pipeline.executor import PipelineExecutor
        from picarones.pipeline.spec import PipelineSpec, PipelineStep
        from picarones.domain.documents import DocumentRef

        text_path = tmp_path / "doc01.txt"
        text_path.write_text("input ocr", encoding="utf-8")

        adapter = _StubLLMAdapter(response_text="cleaned text")
        executor = PipelineExecutor(
            adapter_resolver=lambda name: adapter,
        )
        spec = PipelineSpec(
            name="post_correction",
            initial_inputs=(ArtifactType.RAW_TEXT,),
            steps=(
                PipelineStep(
                    id="llm",
                    kind="post_correction",
                    adapter_name="stub_llm",
                    input_types=(ArtifactType.RAW_TEXT,),
                    output_types=(ArtifactType.CORRECTED_TEXT,),
                ),
            ),
        )
        result = executor.run(
            spec=spec,
            document=DocumentRef(id="doc01"),
            initial_inputs={
                ArtifactType.RAW_TEXT: _make_text_artifact(str(text_path)),
            },
            context=_make_context(),
        )
        assert result.succeeded
        # Trouve le CORRECTED_TEXT artefact.
        corrected = [
            a for a in result.artifacts
            if a.type == ArtifactType.CORRECTED_TEXT
        ]
        assert len(corrected) == 1
