"""Tests Sprint 47 — adaptation Tesseract pour exposer token_confidences.

Couvre :

1. ``run()`` retourne ``EngineResult.token_confidences`` non-vide
   quand pytesseract est disponible et qu'``image_to_data`` produit
   des confidences.
2. Le ``text`` retourné reste **strictement identique** à ce que
   produit ``image_to_string`` (rétrocompat octet par octet —
   l'extraction des confidences n'altère jamais le texte).
3. ``expose_confidences=False`` désactive l'extraction (économie
   d'un appel Tesseract par image).
4. Si ``image_to_data`` lève, l'OCR continue : ``text`` retourné,
   ``token_confidences = None``, warning loggé.
5. Les non-mots (conf = -1) et tokens vides sont filtrés.
6. Les confidences passent le runner Sprint 42 et alimentent
   ``DocumentResult.calibration_metrics``.
7. Si pytesseract n'est pas installé, ``token_confidences = None``
   sans crash (fallback gracieux).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import picarones.engines.tesseract as tesseract_module
from picarones.engines.tesseract import TesseractEngine


# ──────────────────────────────────────────────────────────────────────────
# Mocks
# ──────────────────────────────────────────────────────────────────────────


class _MockPytesseract:
    """Mock minimal de pytesseract qui simule une réponse réaliste."""

    class Output:
        DICT = "DICT"

    class pytesseract:  # noqa: N801 (imite le namespace réel)
        tesseract_cmd: str = "tesseract"

    def __init__(
        self,
        text: str = "Bonjour le monde",
        data: dict | None = None,
        raise_on_data: bool = False,
        raise_on_string: bool = False,
    ) -> None:
        self._text = text
        self._data = data or {
            "text": ["Bonjour", "le", "monde"],
            "conf": [95.5, 88.0, 91.3],
        }
        self.raise_on_data = raise_on_data
        self.raise_on_string = raise_on_string

    def image_to_string(self, image, lang=None, config=None) -> str:
        if self.raise_on_string:
            raise RuntimeError("simulated OCR failure")
        return self._text

    def image_to_data(self, image, lang=None, config=None, output_type=None) -> dict:
        if self.raise_on_data:
            raise RuntimeError("simulated image_to_data failure")
        return self._data

    def get_tesseract_version(self):
        class _V:
            vstring = "5.0.0-mock"
        return _V()


class _MockImage:
    @staticmethod
    def open(path):
        return object()  # placeholder


@pytest.fixture
def patched_tesseract(monkeypatch: pytest.MonkeyPatch) -> _MockPytesseract:
    """Patche le module pour utiliser le mock."""
    mock = _MockPytesseract()
    monkeypatch.setattr(tesseract_module, "pytesseract", mock)
    monkeypatch.setattr(tesseract_module, "Image", _MockImage)
    monkeypatch.setattr(tesseract_module, "_PYTESSERACT_AVAILABLE", True)
    return mock


# ──────────────────────────────────────────────────────────────────────────
# 1-2. run() expose token_confidences sans modifier le texte
# ──────────────────────────────────────────────────────────────────────────


class TestRunExposesConfidences:
    def test_run_returns_token_confidences(
        self, patched_tesseract: _MockPytesseract, tmp_path: Path,
    ) -> None:
        img = tmp_path / "p.png"
        img.write_bytes(b"x")
        engine = TesseractEngine()
        result = engine.run(img)

        assert result.token_confidences is not None
        assert len(result.token_confidences) == 3
        assert result.token_confidences[0] == {
            "token": "Bonjour", "confidence": pytest.approx(95.5),
        }

    def test_text_matches_image_to_string(
        self, patched_tesseract: _MockPytesseract, tmp_path: Path,
    ) -> None:
        """Le texte de l'EngineResult doit être strictement celui de
        image_to_string, pas une reconstruction depuis image_to_data."""
        img = tmp_path / "p.png"
        img.write_bytes(b"x")
        engine = TesseractEngine()
        result = engine.run(img)

        assert result.text == "Bonjour le monde"


# ──────────────────────────────────────────────────────────────────────────
# 3. expose_confidences=False désactive
# ──────────────────────────────────────────────────────────────────────────


class TestExposeConfidencesFlag:
    def test_disabled_returns_no_confidences(
        self, patched_tesseract: _MockPytesseract, tmp_path: Path,
    ) -> None:
        img = tmp_path / "p.png"
        img.write_bytes(b"x")
        engine = TesseractEngine(config={"expose_confidences": False})
        result = engine.run(img)

        assert result.text == "Bonjour le monde"
        assert result.token_confidences is None


# ──────────────────────────────────────────────────────────────────────────
# 4. image_to_data échoue → fallback gracieux
# ──────────────────────────────────────────────────────────────────────────


class TestExtractionFailureFallback:
    def test_image_to_data_failure_returns_none_confidences(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        mock = _MockPytesseract(raise_on_data=True)
        monkeypatch.setattr(tesseract_module, "pytesseract", mock)
        monkeypatch.setattr(tesseract_module, "Image", _MockImage)
        monkeypatch.setattr(tesseract_module, "_PYTESSERACT_AVAILABLE", True)

        img = tmp_path / "p.png"
        img.write_bytes(b"x")
        engine = TesseractEngine()
        with caplog.at_level("WARNING", logger="picarones.engines.tesseract"):
            result = engine.run(img)

        # OCR a réussi sur le texte
        assert result.text == "Bonjour le monde"
        assert result.error is None
        # Mais les confidences sont None
        assert result.token_confidences is None
        # Et un warning explicite a été émis
        assert any("token_confidences" in rec.message for rec in caplog.records)

    def test_image_to_string_failure_keeps_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """Si l'OCR principal lève, on n'essaie même pas d'extraire les
        confidences (cohérent avec le contrat de BaseOCREngine.run)."""
        mock = _MockPytesseract(raise_on_string=True)
        monkeypatch.setattr(tesseract_module, "pytesseract", mock)
        monkeypatch.setattr(tesseract_module, "Image", _MockImage)
        monkeypatch.setattr(tesseract_module, "_PYTESSERACT_AVAILABLE", True)

        img = tmp_path / "p.png"
        img.write_bytes(b"x")
        engine = TesseractEngine()
        result = engine.run(img)

        assert result.error == "simulated OCR failure"
        assert result.text == ""
        assert result.token_confidences is None


# ──────────────────────────────────────────────────────────────────────────
# 5. Filtrage des non-mots et tokens vides
# ──────────────────────────────────────────────────────────────────────────


class TestTokenFiltering:
    def test_negative_conf_filtered(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        mock = _MockPytesseract(
            text="Bonjour monde",
            data={
                "text": ["Bonjour", "", "monde", "."],
                "conf": [95.0, -1.0, 88.0, -1.0],
            },
        )
        monkeypatch.setattr(tesseract_module, "pytesseract", mock)
        monkeypatch.setattr(tesseract_module, "Image", _MockImage)
        monkeypatch.setattr(tesseract_module, "_PYTESSERACT_AVAILABLE", True)

        img = tmp_path / "p.png"
        img.write_bytes(b"x")
        engine = TesseractEngine()
        result = engine.run(img)

        assert result.token_confidences is not None
        # Seuls "Bonjour" et "monde" sont retenus (conf > 0 et token non vide)
        tokens = [tc["token"] for tc in result.token_confidences]
        assert tokens == ["Bonjour", "monde"]

    def test_mismatched_lengths_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        # text et conf de longueurs différentes → format inattendu
        mock = _MockPytesseract(
            text="Bonjour",
            data={"text": ["Bonjour", "le"], "conf": [95.0]},
        )
        monkeypatch.setattr(tesseract_module, "pytesseract", mock)
        monkeypatch.setattr(tesseract_module, "Image", _MockImage)
        monkeypatch.setattr(tesseract_module, "_PYTESSERACT_AVAILABLE", True)

        img = tmp_path / "p.png"
        img.write_bytes(b"x")
        engine = TesseractEngine()
        result = engine.run(img)

        assert result.token_confidences is None


# ──────────────────────────────────────────────────────────────────────────
# 6. Bout-en-bout avec le runner : calibration_metrics calculée
# ──────────────────────────────────────────────────────────────────────────


class TestEndToEndWithRunner:
    def test_runner_picks_up_confidences_and_computes_calibration(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        from picarones.measurements.runner import _compute_document_result
        from picarones.evaluation.engines.base import EngineResult

        # Simulation : on appelle directement _compute_document_result
        # avec un EngineResult mocké qui porte des confidences. On
        # vérifie que la calibration_metrics est bien attachée.
        ocr = EngineResult(
            engine_name="tess",
            image_path="/tmp/x.png",
            text="alpha beta gamma",
            duration_seconds=0.1,
            token_confidences=[
                {"token": "alpha", "confidence": 95.0},
                {"token": "beta",  "confidence": 95.0},
                {"token": "gamma", "confidence": 95.0},
            ],
        )
        dr = _compute_document_result(
            doc_id="d1", image_path="/tmp/x.png",
            ground_truth="alpha beta gamma",
            ocr_result=ocr, char_exclude=None,
        )
        assert dr.calibration_metrics is not None
        # 3 tokens, tous corrects → accuracy = 1, conf = 0.95
        assert dr.calibration_metrics["overall_accuracy"] == 1.0
        assert dr.calibration_metrics["overall_confidence"] == pytest.approx(0.95)


# ──────────────────────────────────────────────────────────────────────────
# 7. pytesseract absent → fallback gracieux
# ──────────────────────────────────────────────────────────────────────────


class TestPytesseractAbsent:
    def test_extraction_returns_none_without_pytesseract(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(tesseract_module, "_PYTESSERACT_AVAILABLE", False)

        engine = TesseractEngine()
        result = engine.run(tmp_path / "p.png").token_confidences
        assert result is None
