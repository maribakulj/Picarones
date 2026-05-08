"""Sprint A.3 (plan v2.0) — intégration OCR legacy + LLM rewrite.

Vérifie que :

1. ``LegacyOCREngineExecutor`` (Sprint A.1) wrap correctement un
   ``BaseOCREngine`` legacy en ``StepExecutor`` rewrite.
2. ``BaseLLMAdapter.execute()`` (Sprint A.2) accepte un
   ``params["prompt_template"]`` avec convention legacy
   (``{ocr_output}`` / ``{image_b64}``) en plus de la convention
   rewrite (``{text}``).
3. Les deux briques s'enchaînent dans ``PipelineExecutor`` via la
   spec produite par ``make_ocr_llm_pipeline_spec`` (Phase 6 volet 2,
   commit f894bf0) et produisent le texte attendu.

Ce test prouve que la délégation prévue au Sprint B (refactor de
``OCRLLMPipeline.run()``) est techniquement réalisable — le pont
entre l'API legacy et le rewrite est fonctionnel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pytest

from picarones.adapters.legacy_engines._step_executor import (
    LegacyOCREngineExecutor,
)
from picarones.adapters.legacy_engines.base import BaseOCREngine
from picarones.adapters.llm.base import (
    BaseLLMAdapter,
    _substitute_prompt_variables,
)
from picarones.adapters.ocr.base import OCRAdapterError
from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.domain.documents import DocumentRef
from picarones.pipeline import (
    PipelineExecutor,
    RunContext,
    make_ocr_llm_pipeline_spec,
)


# ──────────────────────────────────────────────────────────────────────
# Mocks — OCR engine legacy + LLM adapter rewrite
# ──────────────────────────────────────────────────────────────────────


class _MockOCREngine(BaseOCREngine):
    """OCR engine déterministe (texte fixe quel que soit l'image)."""

    def __init__(self, fixed_text: str = "ocr output text") -> None:
        super().__init__(config={})
        self._fixed_text = fixed_text

    @property
    def name(self) -> str:
        return "mock_ocr"

    def version(self) -> str:
        return "1.0.0"

    def _run_ocr(self, image_path: Path) -> str:
        return self._fixed_text


class _MockLLMAdapter(BaseLLMAdapter):
    """LLM adapter qui renvoie le prompt reçu en upper-case.

    Utile pour vérifier ce que l'adapter a effectivement reçu après
    substitution des variables — le test peut grep le ``LLMResult.text``.
    """

    def __init__(self) -> None:
        super().__init__(model="mock-1", config={})
        self.last_prompt: Optional[str] = None
        self.last_image_b64: Optional[str] = None

    @property
    def name(self) -> str:
        return "mock_llm"

    @property
    def default_model(self) -> str:
        return "mock-1"

    def _call(self, prompt: str, image_b64: Optional[str] = None) -> str:
        self.last_prompt = prompt
        self.last_image_b64 = image_b64
        # Renvoie le prompt entier en upper-case pour qu'on puisse le
        # vérifier côté test.
        return prompt.upper()


# ──────────────────────────────────────────────────────────────────────
# A.1 — LegacyOCREngineExecutor seul
# ──────────────────────────────────────────────────────────────────────


class TestLegacyOCREngineExecutor:
    def test_static_contract(self) -> None:
        """Les attributs StepExecutor sont déclarés correctement."""
        assert LegacyOCREngineExecutor.input_types == frozenset(
            {ArtifactType.IMAGE},
        )
        assert LegacyOCREngineExecutor.output_types == frozenset(
            {ArtifactType.RAW_TEXT},
        )

    def test_rejects_non_engine(self) -> None:
        # Duck-typing : un str n'a ni .run() ni .name → rejeté.
        with pytest.raises(OCRAdapterError):
            LegacyOCREngineExecutor("not an engine")  # type: ignore[arg-type]

    def test_inherits_execution_mode_from_engine(self) -> None:
        engine = _MockOCREngine()
        engine.execution_mode = "cpu"
        step = LegacyOCREngineExecutor(engine)
        assert step.execution_mode == "cpu"

    def test_name_delegates_to_engine(self) -> None:
        step = LegacyOCREngineExecutor(_MockOCREngine())
        assert step.name == "mock_ocr"

    def test_execute_writes_text_artifact(self, tmp_path: Path) -> None:
        """Le wrapper écrit le texte OCR dans le workspace et retourne
        un Artifact RAW_TEXT pointant sur ce fichier."""
        engine = _MockOCREngine(fixed_text="bonjour le monde")
        step = LegacyOCREngineExecutor(engine)

        # Préparer un fichier image factice (le mock n'utilise pas son
        # contenu, mais le wrapper vérifie son existence).
        image_path = tmp_path / "input.png"
        image_path.write_bytes(b"\x89PNG fake")
        image_artifact = Artifact(
            id="doc1:initial:image",
            document_id="doc1",
            type=ArtifactType.IMAGE,
            uri=str(image_path),
        )
        context = RunContext(
            document_id="doc1",
            code_version="test",
            pipeline_name="test_pipe",
            workspace_uri=str(tmp_path),
        )

        outputs = step.execute(
            inputs={ArtifactType.IMAGE: image_artifact},
            params={},
            context=context,
        )

        assert ArtifactType.RAW_TEXT in outputs
        text_artifact = outputs[ArtifactType.RAW_TEXT]
        assert text_artifact.document_id == "doc1"
        assert text_artifact.produced_by_step == "ocr"
        text_path = Path(text_artifact.uri)
        assert text_path.exists()
        assert text_path.read_text(encoding="utf-8") == "bonjour le monde"

    def test_execute_raises_on_missing_image(self, tmp_path: Path) -> None:
        step = LegacyOCREngineExecutor(_MockOCREngine())
        context = RunContext(
            document_id="doc1",
            code_version="test",
            pipeline_name="test_pipe",
            workspace_uri=str(tmp_path),
        )
        with pytest.raises(OCRAdapterError, match="IMAGE manquant"):
            step.execute(inputs={}, params={}, context=context)


# ──────────────────────────────────────────────────────────────────────
# A.2 — _substitute_prompt_variables et BaseLLMAdapter avec params
# ──────────────────────────────────────────────────────────────────────


class TestPromptSubstitution:
    def test_rewrite_format_text(self) -> None:
        out = _substitute_prompt_variables("Corrige : {text}", "ocr", None)
        assert out == "Corrige : ocr"

    def test_legacy_format_ocr_output(self) -> None:
        out = _substitute_prompt_variables(
            "Corrige : {ocr_output}", "ocr", None,
        )
        assert out == "Corrige : ocr"

    def test_legacy_format_with_image_b64(self) -> None:
        out = _substitute_prompt_variables(
            "Img: {image_b64} OCR: {ocr_output}", "ocr", "b64data",
        )
        assert out == "Img: b64data OCR: ocr"

    def test_legacy_format_image_none_becomes_empty(self) -> None:
        out = _substitute_prompt_variables(
            "Img: {image_b64}, OCR: {ocr_output}", "ocr", None,
        )
        assert out == "Img: , OCR: ocr"

    def test_only_image_b64_no_ocr_output(self) -> None:
        """Un template legacy peut n'avoir que ``{image_b64}`` (mode
        zero-shot avec convention legacy)."""
        out = _substitute_prompt_variables(
            "Transcris l'image : {image_b64}", "", "b64data",
        )
        assert out == "Transcris l'image : b64data"


class TestBaseLLMAdapterAcceptsParamsPromptTemplate:
    def test_params_prompt_template_overrides_config(
        self, tmp_path: Path,
    ) -> None:
        """Sprint A.2 — un caller qui construit une PipelineSpec peut
        injecter un prompt_template via ``params``, qui prime sur
        ``self.config["correction_prompt"]``."""
        adapter = _MockLLMAdapter()
        adapter.config["correction_prompt"] = "OLD CONFIG: {text}"

        # Préparer un text artifact (le LLM lit depuis disque).
        text_path = tmp_path / "ocr.txt"
        text_path.write_text("ocr text here", encoding="utf-8")
        text_artifact = Artifact(
            id="doc1:ocr:raw_text",
            document_id="doc1",
            type=ArtifactType.RAW_TEXT,
            uri=str(text_path),
        )
        context = RunContext(
            document_id="doc1",
            code_version="test",
            pipeline_name="test_pipe",
            workspace_uri=str(tmp_path),
        )

        adapter.execute(
            inputs={ArtifactType.RAW_TEXT: text_artifact},
            params={"prompt_template": "NEW PARAM: {ocr_output}"},
            context=context,
        )

        # Le prompt utilisé doit venir de params, pas de config.
        assert adapter.last_prompt == "NEW PARAM: ocr text here"

    def test_params_legacy_template_with_image(
        self, tmp_path: Path,
    ) -> None:
        """Le template legacy ``{ocr_output}`` + ``{image_b64}`` est
        substitué correctement quand l'image est dans les inputs."""
        adapter = _MockLLMAdapter()
        text_path = tmp_path / "ocr.txt"
        text_path.write_text("hello", encoding="utf-8")
        image_path = tmp_path / "img.png"
        image_path.write_bytes(b"\x89PNG fake")
        text_artifact = Artifact(
            id="doc1:ocr:raw_text",
            document_id="doc1",
            type=ArtifactType.RAW_TEXT,
            uri=str(text_path),
        )
        image_artifact = Artifact(
            id="doc1:initial:image",
            document_id="doc1",
            type=ArtifactType.IMAGE,
            uri=str(image_path),
        )
        context = RunContext(
            document_id="doc1",
            code_version="test",
            pipeline_name="test_pipe",
            workspace_uri=str(tmp_path),
        )

        adapter.execute(
            inputs={
                ArtifactType.RAW_TEXT: text_artifact,
                ArtifactType.IMAGE: image_artifact,
            },
            params={
                "prompt_template": (
                    "T:{ocr_output}|I:{image_b64}"
                ),
            },
            context=context,
        )

        assert adapter.last_prompt is not None
        assert adapter.last_prompt.startswith("T:hello|I:")
        # L'image a été passée au LLM (mode multimodal).
        assert adapter.last_image_b64 is not None


# ──────────────────────────────────────────────────────────────────────
# A.3 — Intégration OCR legacy + LLM rewrite via PipelineExecutor
# ──────────────────────────────────────────────────────────────────────


class TestEndToEndOCRPlusLLM:
    """Le scénario clé : un caller qui aujourd'hui construit un
    ``OCRLLMPipeline(...)`` peut, dès Sprint A, le remplacer par
    une ``PipelineSpec`` exécutée via ``PipelineExecutor`` avec un
    OCR engine legacy wrappé."""

    def _build_executor(
        self,
        ocr_engine: BaseOCREngine,
        llm_adapter: BaseLLMAdapter,
    ) -> PipelineExecutor:
        ocr_step = LegacyOCREngineExecutor(ocr_engine)

        def resolver(name: str) -> Any:
            if name == ocr_engine.name:
                return ocr_step
            if name == "mock_llm:mock-1":
                return llm_adapter
            raise KeyError(f"adapter inconnu : {name}")

        return PipelineExecutor(adapter_resolver=resolver)

    def test_text_only_pipeline_runs_end_to_end(
        self, tmp_path: Path,
    ) -> None:
        """Mode TEXT_ONLY — OCR legacy → LLM rewrite produit
        ``CORRECTED_TEXT``."""
        ocr = _MockOCREngine(fixed_text="texte ocr brut")
        llm = _MockLLMAdapter()

        spec = make_ocr_llm_pipeline_spec(
            mode="text_only",
            ocr_adapter_name=ocr.name,
            llm_adapter_name="mock_llm:mock-1",
        )

        # Image factice
        image_path = tmp_path / "scan.png"
        image_path.write_bytes(b"\x89PNG fake")
        document = DocumentRef(id="doc_e2e", image_uri=str(image_path))
        context = RunContext(
            document_id="doc_e2e",
            code_version="test",
            pipeline_name=spec.name,
            workspace_uri=str(tmp_path),
        )
        initial_inputs = {
            ArtifactType.IMAGE: Artifact(
                id="doc_e2e:initial:image",
                document_id="doc_e2e",
                type=ArtifactType.IMAGE,
                uri=str(image_path),
            ),
        }

        executor = self._build_executor(ocr, llm)
        result = executor.run(spec, document, initial_inputs, context)

        assert result.succeeded, f"pipeline failed: {result}"
        # Le résultat porte une liste plate d'artifacts ; on filtre par
        # type pour récupérer le CORRECTED_TEXT produit en bout de chaîne.
        corrected_artifacts = [
            a for a in result.artifacts if a.type == ArtifactType.CORRECTED_TEXT
        ]
        assert len(corrected_artifacts) == 1
        corrected = corrected_artifacts[0]
        text = Path(corrected.uri).read_text(encoding="utf-8")
        # Le mock LLM met le prompt en upper-case ; le texte OCR est
        # quelque part dans cette upper-case version.
        assert "TEXTE OCR BRUT" in text
        # Le LLM a bien reçu le texte OCR (pas l'image en text-only).
        assert "texte ocr brut" in (llm.last_prompt or "")
        assert llm.last_image_b64 is None

    def test_text_and_image_pipeline_passes_image_to_llm(
        self, tmp_path: Path,
    ) -> None:
        """Mode TEXT_AND_IMAGE — le LLM reçoit l'image en plus du
        RAW_TEXT issu de l'OCR."""
        ocr = _MockOCREngine(fixed_text="ocr txt")
        llm = _MockLLMAdapter()

        spec = make_ocr_llm_pipeline_spec(
            mode="text_and_image",
            ocr_adapter_name=ocr.name,
            llm_adapter_name="mock_llm:mock-1",
        )

        image_path = tmp_path / "scan.png"
        image_path.write_bytes(b"\x89PNG fake bytes")
        document = DocumentRef(id="doc_e2e", image_uri=str(image_path))
        context = RunContext(
            document_id="doc_e2e",
            code_version="test",
            pipeline_name=spec.name,
            workspace_uri=str(tmp_path),
        )
        initial_inputs = {
            ArtifactType.IMAGE: Artifact(
                id="doc_e2e:initial:image",
                document_id="doc_e2e",
                type=ArtifactType.IMAGE,
                uri=str(image_path),
            ),
        }

        executor = self._build_executor(ocr, llm)
        result = executor.run(spec, document, initial_inputs, context)

        assert result.succeeded
        # En mode multimodal, le LLM a reçu l'image (encodée base64).
        assert llm.last_image_b64 is not None
        assert len(llm.last_image_b64) > 0
