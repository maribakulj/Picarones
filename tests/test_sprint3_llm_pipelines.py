"""Tests Sprint 3 — Pipelines OCR+LLM, adaptateurs LLM, bibliothèque de prompts, sur-normalisation.

Ces tests couvrent :
- La détection de sur-normalisation LLM (classe 10)
- L'OCRLLMPipeline : modes, chargement de prompts, métadonnées
- Les adaptateurs LLM (instanciation, structure)
- L'intégration dans les fixtures (tesseract → gpt-4o)
- La présence des données pipeline dans le rapport HTML
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Détection de sur-normalisation (classe 10)
# ---------------------------------------------------------------------------

class TestOverNormalization:

    def test_no_over_normalization(self):
        from picarones.pipelines.over_normalization import detect_over_normalization
        gt  = "nostre seigneur le roy"
        ocr = "noltre seigneur le roy"   # erreur OCR sur 'nostre'
        llm = "nostre seigneur le roy"   # LLM corrige → correct
        result = detect_over_normalization(gt, ocr, llm)
        assert result.score == 0.0
        assert result.over_normalized_count == 0

    def test_perfect_llm_no_over_norm(self):
        from picarones.pipelines.over_normalization import detect_over_normalization
        gt  = "nostre seigneur le roy"
        ocr = "nostre seigneur le roy"   # OCR correct
        llm = "nostre seigneur le roy"   # LLM conserve
        result = detect_over_normalization(gt, ocr, llm)
        assert result.score == 0.0
        assert result.total_correct_ocr_words == 4

    def test_over_normalization_detected(self):
        from picarones.pipelines.over_normalization import detect_over_normalization
        gt  = "nostre seigneur le roy"
        ocr = "nostre seigneur le roy"   # OCR correct
        llm = "notre seigneur le roy"    # LLM modifie 'nostre' → 'notre' : sur-normalisation
        result = detect_over_normalization(gt, ocr, llm)
        assert result.over_normalized_count == 1
        assert result.score > 0.0
        assert len(result.over_normalized_passages) == 1
        passage = result.over_normalized_passages[0]
        assert passage["gt"] == "nostre"
        assert passage["ocr"] == "nostre"
        assert passage["llm"] == "notre"

    def test_over_normalization_score_formula(self):
        from picarones.pipelines.over_normalization import detect_over_normalization
        # 4 mots, OCR correct sur tous, LLM modifie 2 → score = 2/4 = 0.5
        gt  = "maistre jehan nostre dame"
        ocr = "maistre jehan nostre dame"
        llm = "maître jehan notre dame"
        result = detect_over_normalization(gt, ocr, llm)
        assert result.total_correct_ocr_words == 4
        assert result.over_normalized_count == 2
        assert result.score == pytest.approx(0.5)

    def test_as_dict_keys(self):
        from picarones.pipelines.over_normalization import detect_over_normalization
        result = detect_over_normalization("foo bar", "foo baz", "foo baz")
        d = result.as_dict()
        assert "score" in d
        assert "total_correct_ocr_words" in d
        assert "over_normalized_count" in d
        assert "over_normalized_passages" in d

    def test_empty_texts(self):
        from picarones.pipelines.over_normalization import detect_over_normalization
        result = detect_over_normalization("", "", "")
        assert result.score == 0.0

    def test_aggregate_over_normalization(self):
        from picarones.pipelines.over_normalization import (
            OverNormalizationResult,
            aggregate_over_normalization,
        )
        results = [
            OverNormalizationResult(total_correct_ocr_words=10, over_normalized_count=1),
            OverNormalizationResult(total_correct_ocr_words=10, over_normalized_count=2),
            None,
        ]
        agg = aggregate_over_normalization(results)
        assert agg["total_correct_ocr_words"] == 20
        assert agg["over_normalized_count"] == 3
        assert agg["score"] == pytest.approx(0.15)
        assert agg["document_count"] == 2


# ---------------------------------------------------------------------------
# Bibliothèque de prompts
# ---------------------------------------------------------------------------

class TestPromptsLibrary:

    _PROMPTS_DIR = Path(__file__).parent.parent / "picarones" / "prompts"

    def test_prompts_directory_exists(self):
        assert self._PROMPTS_DIR.is_dir()

    def test_required_prompt_files_exist(self):
        expected = [
            "correction_medieval_french.txt",
            "correction_imprime_ancien.txt",
            "correction_image_medieval_french.txt",
            "zero_shot_medieval_french.txt",
            "zero_shot_imprime_ancien.txt",
        ]
        for fname in expected:
            assert (self._PROMPTS_DIR / fname).exists(), f"Prompt manquant : {fname}"

    def test_correction_prompt_has_ocr_variable(self):
        text = (self._PROMPTS_DIR / "correction_medieval_french.txt").read_text(encoding="utf-8")
        assert "{ocr_output}" in text

    def test_image_prompt_has_both_variables(self):
        text = (self._PROMPTS_DIR / "correction_image_medieval_french.txt").read_text(encoding="utf-8")
        assert "{ocr_output}" in text

    def test_zero_shot_prompt_has_no_ocr_variable(self):
        text = (self._PROMPTS_DIR / "zero_shot_medieval_french.txt").read_text(encoding="utf-8")
        assert "{ocr_output}" not in text

    def test_prompts_not_empty(self):
        for f in self._PROMPTS_DIR.glob("*.txt"):
            assert len(f.read_text(encoding="utf-8").strip()) > 100, f"Prompt trop court : {f.name}"


# ---------------------------------------------------------------------------
# PipelineMode enum
# ---------------------------------------------------------------------------

class TestPipelineMode:

    def test_enum_values(self):
        from picarones.pipelines.base import PipelineMode
        assert PipelineMode.TEXT_ONLY.value == "text_only"
        assert PipelineMode.TEXT_AND_IMAGE.value == "text_and_image"
        assert PipelineMode.ZERO_SHOT.value == "zero_shot"

    def test_from_string(self):
        from picarones.pipelines.base import PipelineMode
        assert PipelineMode("text_only") == PipelineMode.TEXT_ONLY


# ---------------------------------------------------------------------------
# Adaptateurs LLM — structure
# ---------------------------------------------------------------------------

class TestLLMAdapters:

    def test_openai_adapter_structure(self):
        from picarones.llm.openai_adapter import OpenAIAdapter
        adapter = OpenAIAdapter(model="gpt-4o")
        assert adapter.name == "openai"
        assert adapter.model == "gpt-4o"

    def test_anthropic_adapter_structure(self):
        from picarones.llm.anthropic_adapter import AnthropicAdapter
        adapter = AnthropicAdapter()
        assert adapter.name == "anthropic"
        assert "claude" in adapter.model.lower()

    def test_mistral_adapter_structure(self):
        from picarones.llm.mistral_adapter import MistralAdapter
        adapter = MistralAdapter()
        assert adapter.name == "mistral"
        assert "mistral" in adapter.model.lower()

    def test_ollama_adapter_structure(self):
        from picarones.llm.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter(model="llama3")
        assert adapter.name == "ollama"
        assert adapter.model == "llama3"

    def test_ollama_custom_base_url(self):
        from picarones.llm.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter(config={"base_url": "http://myserver:11434"})
        assert adapter._base_url == "http://myserver:11434"

    def test_llm_result_dataclass(self):
        from picarones.llm.base import LLMResult
        r = LLMResult(model_id="gpt-4o", text="bonjour", duration_seconds=1.2)
        assert r.success is True
        r_err = LLMResult(model_id="gpt-4o", text="", duration_seconds=0.1, error="fail")
        assert r_err.success is False

    def test_missing_api_key_raises(self):
        import os
        from picarones.llm.openai_adapter import OpenAIAdapter
        adapter = OpenAIAdapter()
        adapter._api_key = None  # simuler clé manquante
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            adapter._call("test prompt")


# ---------------------------------------------------------------------------
# OCRLLMPipeline — prompt loading, name, steps
# ---------------------------------------------------------------------------

class TestOCRLLMPipeline:

    def _mock_llm(self, response: str = "texte corrigé"):
        """Crée un adaptateur LLM mock qui retourne toujours la même réponse."""
        from picarones.llm.base import BaseLLMAdapter
        class MockLLM(BaseLLMAdapter):
            @property
            def name(self): return "mock"
            @property
            def default_model(self): return "mock-v1"
            def _call(self, prompt, image_b64=None): return response
        return MockLLM()

    def test_load_builtin_prompt(self):
        from picarones.pipelines.base import OCRLLMPipeline, PipelineMode
        pipeline = OCRLLMPipeline(
            llm_adapter=self._mock_llm(),
            mode=PipelineMode.TEXT_ONLY,
            prompt="correction_medieval_french.txt",
        )
        assert "{ocr_output}" in pipeline._prompt_template

    def test_prompt_substitution_text_only(self):
        from picarones.pipelines.base import OCRLLMPipeline, PipelineMode
        pipeline = OCRLLMPipeline(
            llm_adapter=self._mock_llm(),
            mode=PipelineMode.TEXT_ONLY,
            prompt="correction_medieval_french.txt",
        )
        built = pipeline._build_prompt(ocr_text="mon texte ocr")
        assert "mon texte ocr" in built
        assert "{ocr_output}" not in built

    def test_auto_name_text_only(self):
        from picarones.pipelines.base import OCRLLMPipeline, PipelineMode
        from picarones.engines.tesseract import TesseractEngine
        pipeline = OCRLLMPipeline(
            ocr_engine=TesseractEngine(),
            llm_adapter=self._mock_llm(),
            mode=PipelineMode.TEXT_ONLY,
        )
        assert "tesseract" in pipeline.name.lower()
        assert "mock-v1" in pipeline.name

    def test_auto_name_zero_shot(self):
        from picarones.pipelines.base import OCRLLMPipeline, PipelineMode
        pipeline = OCRLLMPipeline(
            llm_adapter=self._mock_llm(),
            mode=PipelineMode.ZERO_SHOT,
        )
        assert "zero-shot" in pipeline.name

    def test_custom_name(self):
        from picarones.pipelines.base import OCRLLMPipeline, PipelineMode
        pipeline = OCRLLMPipeline(
            llm_adapter=self._mock_llm(),
            mode=PipelineMode.TEXT_ONLY,
            pipeline_name="mon_pipeline_custom",
        )
        assert pipeline.name == "mon_pipeline_custom"

    def test_pipeline_steps_without_ocr(self):
        from picarones.pipelines.base import OCRLLMPipeline, PipelineMode
        pipeline = OCRLLMPipeline(
            llm_adapter=self._mock_llm(),
            mode=PipelineMode.ZERO_SHOT,
        )
        steps = pipeline._build_steps_info()
        assert len(steps) == 1
        assert steps[0]["type"] == "llm"
        assert steps[0]["mode"] == "zero_shot"

    def test_pipeline_steps_with_ocr(self):
        from picarones.engines.tesseract import TesseractEngine
        from picarones.pipelines.base import OCRLLMPipeline, PipelineMode
        pipeline = OCRLLMPipeline(
            ocr_engine=TesseractEngine(),
            llm_adapter=self._mock_llm(),
            mode=PipelineMode.TEXT_ONLY,
        )
        steps = pipeline._build_steps_info()
        assert len(steps) == 2
        assert steps[0]["type"] == "ocr"
        assert steps[1]["type"] == "llm"

    def test_load_nonexistent_prompt_raises(self):
        from picarones.pipelines.base import OCRLLMPipeline, PipelineMode
        with pytest.raises(FileNotFoundError):
            OCRLLMPipeline(
                llm_adapter=self._mock_llm(),
                mode=PipelineMode.TEXT_ONLY,
                prompt="inexistant_prompt_xyz.txt",
            )

    def test_text_only_requires_ocr_engine(self):
        from picarones.pipelines.base import OCRLLMPipeline, PipelineMode
        pipeline = OCRLLMPipeline(
            llm_adapter=self._mock_llm(),
            mode=PipelineMode.TEXT_ONLY,
        )
        with pytest.raises(ValueError, match="ocr_engine"):
            pipeline._run_ocr(Path("/nonexistent/image.jpg"))

    def test_is_pipeline_flag(self):
        from picarones.pipelines.base import OCRLLMPipeline, PipelineMode
        from picarones.engines.base import BaseOCREngine
        pipeline = OCRLLMPipeline(
            llm_adapter=self._mock_llm(),
            mode=PipelineMode.ZERO_SHOT,
        )
        # Doit être utilisable comme BaseOCREngine
        assert isinstance(pipeline, BaseOCREngine)


# ---------------------------------------------------------------------------
# Intégration fixtures — pipeline tesseract → gpt-4o
# ---------------------------------------------------------------------------

class TestFixturesPipeline:

    @pytest.fixture(scope="class")
    def benchmark(self):
        from picarones.fixtures import generate_sample_benchmark
        return generate_sample_benchmark(n_docs=3, seed=42)

    def test_pipeline_engine_present(self, benchmark):
        names = [r.engine_name for r in benchmark.engine_reports]
        assert "tesseract → gpt-4o" in names

    def test_pipeline_report_has_pipeline_info(self, benchmark):
        report = next(r for r in benchmark.engine_reports if r.engine_name == "tesseract → gpt-4o")
        assert report.is_pipeline
        assert report.pipeline_info.get("pipeline_mode") == "text_and_image"
        assert report.pipeline_info.get("llm_model") == "gpt-4o"

    def test_pipeline_documents_have_ocr_intermediate(self, benchmark):
        report = next(r for r in benchmark.engine_reports if r.engine_name == "tesseract → gpt-4o")
        for dr in report.document_results:
            assert dr.ocr_intermediate is not None, f"ocr_intermediate manquant sur {dr.doc_id}"
            assert len(dr.ocr_intermediate) > 0

    def test_pipeline_documents_have_over_normalization(self, benchmark):
        report = next(r for r in benchmark.engine_reports if r.engine_name == "tesseract → gpt-4o")
        for dr in report.document_results:
            on = dr.pipeline_metadata.get("over_normalization")
            assert on is not None, f"over_normalization manquant sur {dr.doc_id}"
            assert "score" in on
            assert "total_correct_ocr_words" in on

    def test_pipeline_report_has_aggregated_over_normalization(self, benchmark):
        report = next(r for r in benchmark.engine_reports if r.engine_name == "tesseract → gpt-4o")
        on = report.pipeline_info.get("over_normalization")
        assert on is not None
        assert "score" in on
        assert on["document_count"] == 3

    def test_pipeline_pipeline_steps_in_info(self, benchmark):
        report = next(r for r in benchmark.engine_reports if r.engine_name == "tesseract → gpt-4o")
        steps = report.pipeline_info.get("pipeline_steps", [])
        assert len(steps) == 2
        assert steps[0]["type"] == "ocr"
        assert steps[1]["type"] == "llm"

    def test_non_pipeline_reports_empty_pipeline_info(self, benchmark):
        # Les concurrents pipeline (LLM ou VLM) ont un pipeline_info non vide
        pipeline_engines = {"tesseract → gpt-4o", "gpt-4o-vision (zero-shot)"}
        for report in benchmark.engine_reports:
            if report.engine_name not in pipeline_engines:
                assert not report.is_pipeline
                assert report.pipeline_info == {}


# ---------------------------------------------------------------------------
# Intégration rapport HTML — pipeline dans les données JSON
# ---------------------------------------------------------------------------

class TestReportWithPipeline:

    @pytest.fixture(scope="class")
    def report_data(self):
        from picarones.fixtures import generate_sample_benchmark
        from picarones.report.generator import _build_report_data
        bm = generate_sample_benchmark(n_docs=3, seed=42)
        images_b64 = bm.metadata.get("_images_b64", {})
        return _build_report_data(bm, images_b64)

    def test_pipeline_engine_in_data(self, report_data):
        names = [e["name"] for e in report_data["engines"]]
        assert "tesseract → gpt-4o" in names

    def test_pipeline_engine_has_is_pipeline_flag(self, report_data):
        pipeline_e = next(e for e in report_data["engines"] if e["name"] == "tesseract → gpt-4o")
        assert pipeline_e["is_pipeline"] is True

    def test_non_pipeline_engines_not_flagged(self, report_data):
        # Les concurrents pipeline (LLM ou VLM zero-shot) sont correctement marqués is_pipeline=True
        pipeline_engines = {"tesseract → gpt-4o", "gpt-4o-vision (zero-shot)"}
        for e in report_data["engines"]:
            if e["name"] not in pipeline_engines:
                assert e["is_pipeline"] is False

    def test_pipeline_has_over_normalization_in_info(self, report_data):
        pipeline_e = next(e for e in report_data["engines"] if e["name"] == "tesseract → gpt-4o")
        pi = pipeline_e.get("pipeline_info", {})
        assert pi.get("over_normalization") is not None

    def test_document_results_have_ocr_intermediate(self, report_data):
        for doc in report_data["documents"]:
            pipeline_er = next(
                (er for er in doc["engine_results"] if er["engine"] == "tesseract → gpt-4o"),
                None,
            )
            assert pipeline_er is not None
            assert "ocr_intermediate" in pipeline_er
            assert "ocr_diff" in pipeline_er
            assert "llm_correction_diff" in pipeline_er

    def test_document_results_have_over_normalization(self, report_data):
        for doc in report_data["documents"]:
            pipeline_er = next(
                (er for er in doc["engine_results"] if er["engine"] == "tesseract → gpt-4o"),
                None,
            )
            assert pipeline_er is not None
            assert "over_normalization" in pipeline_er

    def test_html_contains_pipeline_tag(self, tmp_path):
        from picarones.fixtures import generate_sample_benchmark
        from picarones.report.generator import ReportGenerator
        bm = generate_sample_benchmark(n_docs=3, seed=42)
        out = tmp_path / "report.html"
        ReportGenerator(bm).generate(out)
        html = out.read_text(encoding="utf-8")
        assert "pipeline" in html.lower()
        assert "tesseract" in html
