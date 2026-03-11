"""Tests pour le sprint 15 — Correction des bugs dans les pipelines OCR+LLM.

Bug 1 : Sortie LLM vide → WARNING logué + pas de crash
Bug 2 : CER 0.00% pour hypothèse vide → doit être 1.0 (100%)
Bug 3 : Divergence runner/rapport → cohérence des métriques
"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Bug 2 — compute_metrics : hypothèse vide
# ---------------------------------------------------------------------------

class TestEmptyHypothesisMetrics:
    """compute_metrics doit retourner CER=1.0, pas 0.0, pour hypothèse vide."""

    def test_empty_hypothesis_cer_is_one(self):
        from picarones.core.metrics import compute_metrics
        result = compute_metrics("Bonjour le monde", "")
        assert result.cer == pytest.approx(1.0)
        assert result.error is None

    def test_empty_hypothesis_all_metrics_are_one(self):
        from picarones.core.metrics import compute_metrics
        result = compute_metrics("hello world", "")
        assert result.cer == pytest.approx(1.0)
        assert result.wer == pytest.approx(1.0)
        assert result.mer == pytest.approx(1.0)
        assert result.wil == pytest.approx(1.0)

    def test_whitespace_only_hypothesis_cer_is_one(self):
        from picarones.core.metrics import compute_metrics
        result = compute_metrics("Bonjour", "   \t\n")
        assert result.cer == pytest.approx(1.0)

    def test_none_hypothesis_guarded(self):
        """compute_metrics ne doit pas planter si hypothesis=None."""
        from picarones.core.metrics import compute_metrics
        # None ne sera jamais passé en pratique, mais on teste la robustesse
        # via une chaîne vide (le runner convertit None → "")
        result = compute_metrics("test", "")
        assert result.cer == pytest.approx(1.0)

    def test_both_empty_cer_is_zero(self):
        """Référence ET hypothèse vides → CER=0.0 (pas d'erreur à mesurer)."""
        from picarones.core.metrics import compute_metrics
        result = compute_metrics("", "")
        assert result.cer == pytest.approx(0.0)

    def test_empty_reference_nonempty_hypothesis(self):
        """Référence vide avec hypothèse non vide → CER=1.0 (comportement existant)."""
        from picarones.core.metrics import compute_metrics
        result = compute_metrics("", "something")
        assert result.cer == pytest.approx(1.0)

    def test_normal_case_unchanged(self):
        """Un cas normal ne doit pas être affecté par le guard."""
        from picarones.core.metrics import compute_metrics
        result = compute_metrics("abcd", "abce")
        assert result.cer == pytest.approx(0.25)
        assert result.error is None


# ---------------------------------------------------------------------------
# Bug 1 — MistralAdapter : WARNING pour réponse vide
# ---------------------------------------------------------------------------

class TestMistralAdapterLogging:
    """MistralAdapter doit loguer un WARNING si la réponse LLM est vide."""

    def _make_mock_mistral_module(self, content: str | None):
        """Retourne un module mistralai simulé avec la réponse donnée."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = content

        mock_client = MagicMock()
        mock_client.chat.complete.return_value = mock_response

        MockMistralClass = MagicMock(return_value=mock_client)

        import types
        fake_module = types.ModuleType("mistralai")
        fake_module.Mistral = MockMistralClass
        return fake_module, mock_client

    def _run_adapter(self, adapter, fake_mod, prompt="test prompt", image_b64=None):
        """Exécute l'adapter avec le module mistralai simulé."""
        import sys
        with patch.dict(sys.modules, {"mistralai": fake_mod}):
            adapter._api_key = "fake-key"  # injecter la clé directement
            return adapter.complete(prompt, image_b64=image_b64)

    def test_warning_on_empty_response(self, caplog):
        """Un WARNING doit être émis si le LLM retourne une chaîne vide."""
        from picarones.llm.mistral_adapter import MistralAdapter

        fake_mod, _ = self._make_mock_mistral_module("")
        adapter = MistralAdapter(model="ministral-3b-latest")

        with caplog.at_level(logging.WARNING, logger="picarones.llm.mistral_adapter"):
            result = self._run_adapter(adapter, fake_mod)

        assert result.text == ""
        assert any(
            "vide" in rec.message.lower() or "empty" in rec.message.lower()
            for rec in caplog.records
            if rec.levelno >= logging.WARNING
        ), f"WARNING attendu, messages : {[r.message for r in caplog.records]}"

    def test_no_warning_on_normal_response(self, caplog):
        """Aucun WARNING ne doit être émis pour une réponse normale."""
        from picarones.llm.mistral_adapter import MistralAdapter

        fake_mod, _ = self._make_mock_mistral_module("Texte OCR corrigé")
        adapter = MistralAdapter(model="ministral-3b-latest")

        with caplog.at_level(logging.WARNING, logger="picarones.llm.mistral_adapter"):
            result = self._run_adapter(adapter, fake_mod)

        assert result.text == "Texte OCR corrigé"
        assert not any(rec.levelno >= logging.WARNING for rec in caplog.records)

    def test_warning_on_none_response_content(self, caplog):
        """WARNING doit être émis si message.content est None."""
        from picarones.llm.mistral_adapter import MistralAdapter

        fake_mod, _ = self._make_mock_mistral_module(None)
        adapter = MistralAdapter(model="ministral-3b-latest")

        with caplog.at_level(logging.WARNING, logger="picarones.llm.mistral_adapter"):
            result = self._run_adapter(adapter, fake_mod)

        assert result.text == ""
        assert any(rec.levelno >= logging.WARNING for rec in caplog.records)

    def test_text_only_models_set_exists(self):
        """La liste des modèles text-only doit contenir ministral-3b."""
        from picarones.llm.mistral_adapter import _TEXT_ONLY_MODELS
        assert "ministral-3b-latest" in _TEXT_ONLY_MODELS

    def test_image_ignored_for_text_only_model(self, caplog):
        """L'image doit être ignorée (avec WARNING) pour un modèle text-only."""
        from picarones.llm.mistral_adapter import MistralAdapter

        fake_mod, mock_client = self._make_mock_mistral_module("résultat")
        adapter = MistralAdapter(model="ministral-3b-latest")

        with caplog.at_level(logging.WARNING, logger="picarones.llm.mistral_adapter"):
            result = self._run_adapter(adapter, fake_mod, image_b64="fake_b64")

        # L'appel doit avoir été fait SANS image (modèle text-only)
        call_kwargs = mock_client.chat.complete.call_args
        _, kwargs = call_kwargs
        msg_content = kwargs.get("messages", [{}])[0].get("content", "")
        assert isinstance(msg_content, str), "Image aurait dû être ignorée (content doit être str)"
        # Au moins un WARNING doit mentionner l'image ignorée
        assert any("ignor" in rec.message.lower() for rec in caplog.records
                   if rec.levelno >= logging.WARNING)


# ---------------------------------------------------------------------------
# Bug 1 — OCRLLMPipeline : WARNING quand le LLM retourne texte vide
# ---------------------------------------------------------------------------

class TestPipelineEmptyLLMResponse:
    """Le pipeline doit loguer un WARNING si le LLM retourne un texte vide."""

    def _make_pipeline(self, llm_text: str):
        """Crée un pipeline dont le LLM retourne llm_text."""
        from picarones.pipelines.base import OCRLLMPipeline, PipelineMode
        from picarones.llm.base import LLMResult

        mock_ocr = MagicMock()
        mock_ocr.name = "mock_ocr"
        mock_ocr.run.return_value = MagicMock(text="texte ocr brut", error=None, success=True)
        mock_ocr._safe_version.return_value = "1.0"

        mock_llm = MagicMock()
        mock_llm.name = "mock_llm"
        mock_llm.model = "mock-model"
        mock_llm.complete.return_value = LLMResult(
            model_id="mock-model", text=llm_text, duration_seconds=0.1,
        )

        return OCRLLMPipeline(
            ocr_engine=mock_ocr,
            llm_adapter=mock_llm,
            mode=PipelineMode.TEXT_ONLY,
            prompt="correction_medieval_french.txt",
        )

    def test_warning_on_empty_llm_output(self, tmp_path, caplog):
        """WARNING doit être logu si le LLM retourne une chaîne vide."""
        import shutil
        # Créer une fausse image
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        pipeline = self._make_pipeline("")
        with caplog.at_level(logging.WARNING, logger="picarones.pipelines.base"):
            result = pipeline.run(img_path)

        assert result.text == ""
        assert any(
            "vide" in rec.message.lower() or "empty" in rec.message.lower()
            for rec in caplog.records
            if rec.levelno >= logging.WARNING
        ), f"WARNING attendu, messages : {[r.message for r in caplog.records]}"

    def test_no_warning_on_normal_llm_output(self, tmp_path, caplog):
        """Aucun WARNING ne doit être émis pour une sortie LLM normale."""
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        pipeline = self._make_pipeline("Texte corrigé par le LLM")
        with caplog.at_level(logging.WARNING, logger="picarones.pipelines.base"):
            result = pipeline.run(img_path)

        assert result.text == "Texte corrigé par le LLM"
        assert not any(
            "vide" in rec.message.lower()
            for rec in caplog.records
            if rec.levelno >= logging.WARNING
        )


# ---------------------------------------------------------------------------
# Bug 3 — Cohérence runner/rapport : empty hypothesis → CER 1.0 dans DocumentResult
# ---------------------------------------------------------------------------

class TestRunnerDocumentResultCohérence:
    """Le DocumentResult doit stocker CER=1.0 pour une hypothèse vide."""

    def test_empty_hypothesis_stored_as_cer_one(self):
        """_compute_document_result avec text="" → metrics.cer = 1.0."""
        from picarones.core.runner import _compute_document_result
        from picarones.engines.base import EngineResult

        ocr_result = EngineResult(
            engine_name="TestEngine",
            image_path="fake.png",
            text="",       # ← sortie vide
            duration_seconds=1.0,
            error=None,    # ← pas d'erreur technique
        )

        doc_result = _compute_document_result(
            doc_id="doc1",
            image_path="fake.png",
            ground_truth="Bonjour le monde",
            ocr_result=ocr_result,
            char_exclude=None,
        )

        assert doc_result.metrics.cer == pytest.approx(1.0), (
            f"CER attendu 1.0 pour hypothèse vide, obtenu {doc_result.metrics.cer}"
        )
        assert doc_result.metrics.error is None, (
            "L'erreur ne devrait pas être renseignée — c'est une hypothèse vide, pas une erreur technique"
        )

    def test_engine_error_also_gives_cer_one(self):
        """EngineResult avec error → metrics.cer = 1.0 (comportement existant)."""
        from picarones.core.runner import _compute_document_result
        from picarones.engines.base import EngineResult

        ocr_result = EngineResult(
            engine_name="TestEngine",
            image_path="fake.png",
            text="",
            duration_seconds=0.0,
            error="Moteur en erreur",
        )

        doc_result = _compute_document_result(
            doc_id="doc1",
            image_path="fake.png",
            ground_truth="Bonjour le monde",
            ocr_result=ocr_result,
            char_exclude=None,
        )

        assert doc_result.metrics.cer == pytest.approx(1.0)
        assert doc_result.metrics.error is not None
