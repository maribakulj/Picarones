"""Tests unitaires pour les adaptateurs moteurs OCR.

Les tests vérifient la structure et le comportement des adaptateurs
sans requérir que Tesseract ou Pero OCR soient réellement installés.
"""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from picarones.engines.base import BaseOCREngine, EngineResult
from picarones.engines.tesseract import TesseractEngine
from picarones.engines.pero_ocr import PeroOCREngine


# ---------------------------------------------------------------------------
# Tests BaseOCREngine
# ---------------------------------------------------------------------------

class ConcreteEngine(BaseOCREngine):
    """Implémentation minimale pour tester la classe de base."""

    @property
    def name(self) -> str:
        return "test_engine"

    def version(self) -> str:
        return "1.0.0"

    def _run_ocr(self, image_path: Path) -> str:
        return "Texte extrait par le moteur de test."


class FailingEngine(BaseOCREngine):
    """Moteur qui lève toujours une exception."""

    @property
    def name(self) -> str:
        return "failing_engine"

    def version(self) -> str:
        return "0.0.0"

    def _run_ocr(self, image_path: Path) -> str:
        raise RuntimeError("OCR échoué intentionnellement.")


class TestBaseOCREngine:
    def test_run_returns_engine_result(self, tmp_path):
        (tmp_path / "image.png").write_bytes(b"fake_image")
        engine = ConcreteEngine()
        result = engine.run(tmp_path / "image.png")
        assert isinstance(result, EngineResult)

    def test_run_success(self, tmp_path):
        (tmp_path / "image.png").write_bytes(b"fake_image")
        engine = ConcreteEngine()
        result = engine.run(tmp_path / "image.png")
        assert result.success is True
        assert result.error is None
        assert result.text == "Texte extrait par le moteur de test."

    def test_run_captures_exception(self, tmp_path):
        (tmp_path / "image.png").write_bytes(b"fake_image")
        engine = FailingEngine()
        result = engine.run(tmp_path / "image.png")
        assert result.success is False
        assert result.error is not None
        assert "OCR échoué" in result.error

    def test_run_measures_duration(self, tmp_path):
        (tmp_path / "image.png").write_bytes(b"fake_image")
        engine = ConcreteEngine()
        result = engine.run(tmp_path / "image.png")
        assert result.duration_seconds >= 0.0

    def test_engine_result_engine_name(self, tmp_path):
        (tmp_path / "image.png").write_bytes(b"fake_image")
        engine = ConcreteEngine()
        result = engine.run(tmp_path / "image.png")
        assert result.engine_name == "test_engine"

    def test_repr(self):
        engine = ConcreteEngine()
        assert "ConcreteEngine" in repr(engine)
        assert "test_engine" in repr(engine)

    def test_image_path_stored(self, tmp_path):
        img = tmp_path / "image.png"
        img.write_bytes(b"fake_image")
        engine = ConcreteEngine()
        result = engine.run(img)
        assert result.image_path == str(img)


# ---------------------------------------------------------------------------
# Tests TesseractEngine
# ---------------------------------------------------------------------------

class TestTesseractEngine:
    def test_name_default(self):
        engine = TesseractEngine()
        assert engine.name == "tesseract"

    def test_name_from_config(self):
        engine = TesseractEngine(config={"name": "tesseract_fra"})
        assert engine.name == "tesseract_fra"

    def test_from_config_factory(self):
        engine = TesseractEngine.from_config({"lang": "lat", "psm": 7})
        assert engine.config["lang"] == "lat"
        assert engine.config["psm"] == 7

    def test_run_with_pytesseract_mocked(self, tmp_path):
        """Vérifie que le moteur appelle pytesseract correctement."""
        img = tmp_path / "page.png"
        img.write_bytes(b"fake")

        with (
            patch("picarones.engines.tesseract._PYTESSERACT_AVAILABLE", True),
            patch("picarones.engines.tesseract.pytesseract") as mock_tess,
            patch("picarones.engines.tesseract.Image") as mock_pil,
        ):
            mock_tess.image_to_string.return_value = "Résultat OCR mock"
            mock_pil.open.return_value = MagicMock()

            engine = TesseractEngine(config={"lang": "fra", "psm": 6})
            result = engine.run(img)

        assert result.success is True
        assert result.text == "Résultat OCR mock"
        mock_tess.image_to_string.assert_called_once()

    def test_run_without_pytesseract_raises(self, tmp_path):
        """Sans pytesseract, le moteur doit retourner un EngineResult avec erreur."""
        img = tmp_path / "page.png"
        img.write_bytes(b"fake")

        with patch("picarones.engines.tesseract._PYTESSERACT_AVAILABLE", False):
            engine = TesseractEngine()
            result = engine.run(img)

        assert result.success is False
        assert "pytesseract" in result.error.lower()


# ---------------------------------------------------------------------------
# Tests PeroOCREngine
# ---------------------------------------------------------------------------

class TestPeroOCREngine:
    def test_name_default(self):
        engine = PeroOCREngine()
        assert engine.name == "pero_ocr"

    def test_name_from_config(self):
        engine = PeroOCREngine(config={"name": "pero_historique"})
        assert engine.name == "pero_historique"

    def test_from_config_factory(self):
        engine = PeroOCREngine.from_config({"config": "/path/to/pero.ini"})
        assert engine.config["config"] == "/path/to/pero.ini"

    def test_run_without_pero_raises(self, tmp_path):
        """Sans pero-ocr, le moteur doit retourner un EngineResult avec erreur."""
        img = tmp_path / "page.png"
        img.write_bytes(b"fake")

        with patch("picarones.engines.pero_ocr._PERO_AVAILABLE", False):
            engine = PeroOCREngine(config={"config": "/fake/config.ini"})
            result = engine.run(img)

        assert result.success is False

    def test_run_without_config_raises(self, tmp_path):
        """Sans paramètre 'config', le moteur doit signaler une erreur claire."""
        img = tmp_path / "page.png"
        img.write_bytes(b"fake")

        with patch("picarones.engines.pero_ocr._PERO_AVAILABLE", True):
            engine = PeroOCREngine()
            result = engine.run(img)

        assert result.success is False
        assert "config" in result.error.lower()


# ---------------------------------------------------------------------------
# Tests EngineResult
# ---------------------------------------------------------------------------

class TestEngineResult:
    def test_success_true_when_no_error(self):
        r = EngineResult(
            engine_name="test", image_path="/img.png",
            text="texte", duration_seconds=0.1
        )
        assert r.success is True

    def test_success_false_when_error(self):
        r = EngineResult(
            engine_name="test", image_path="/img.png",
            text="", duration_seconds=0.1, error="Erreur"
        )
        assert r.success is False

    def test_metadata_default_empty(self):
        r = EngineResult(
            engine_name="test", image_path="/img.png",
            text="", duration_seconds=0.0
        )
        assert r.metadata == {}
