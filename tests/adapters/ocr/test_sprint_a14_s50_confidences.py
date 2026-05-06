"""Sprint A14-S50 — sidecar de confidences OCR (fix audit #4).

Couvre :
1. ``filter_valid_tokens`` — normalisation et filtrage des tokens.
2. ``write_confidences_sidecar`` — fichier JSON canonique.
3. Intégration ``TesseractAdapter`` — sidecar produit en parallèle
   du fichier texte ; opt-out via ``expose_confidences=False``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from picarones.adapters.ocr import TesseractAdapter
from picarones.adapters.ocr.confidences import (
    filter_valid_tokens,
    write_confidences_sidecar,
)
from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.pipeline.types import RunContext


# ──────────────────────────────────────────────────────────────────────
# filter_valid_tokens
# ──────────────────────────────────────────────────────────────────────


class TestFilterValidTokens:
    def test_valid_tokens_passed_through(self) -> None:
        result = filter_valid_tokens([
            {"text": "Hello", "confidence": 0.95},
            {"text": "world", "confidence": 0.80},
        ])
        assert len(result) == 2
        assert result[0]["text"] == "Hello"
        assert result[0]["confidence"] == 0.95

    def test_empty_text_filtered(self) -> None:
        result = filter_valid_tokens([
            {"text": "", "confidence": 0.9},
            {"text": "  ", "confidence": 0.8},
            {"text": "ok", "confidence": 0.7},
        ])
        assert len(result) == 1
        assert result[0]["text"] == "ok"

    def test_negative_confidence_filtered(self) -> None:
        result = filter_valid_tokens([
            {"text": "ok", "confidence": -1},
            {"text": "good", "confidence": 0.5},
        ])
        assert len(result) == 1
        assert result[0]["text"] == "good"

    def test_none_confidence_filtered(self) -> None:
        result = filter_valid_tokens([
            {"text": "x", "confidence": None},
            {"text": "y", "confidence": 0.6},
        ])
        assert len(result) == 1
        assert result[0]["text"] == "y"

    def test_tesseract_format_normalized(self) -> None:
        """Tesseract retourne 0-100 ; on normalise à [0, 1]."""
        result = filter_valid_tokens([
            {"text": "Hello", "confidence": 95},
            {"text": "world", "confidence": 80.5},
        ])
        assert result[0]["confidence"] == 0.95
        assert result[1]["confidence"] == 0.805

    def test_out_of_range_filtered(self) -> None:
        result = filter_valid_tokens([
            {"text": "x", "confidence": 9999},  # > 100, ignoré
            {"text": "y", "confidence": 50},  # OK normalisé à 0.5
        ])
        assert len(result) == 1
        assert result[0]["text"] == "y"
        assert result[0]["confidence"] == 0.5

    def test_non_numeric_filtered(self) -> None:
        result = filter_valid_tokens([
            {"text": "x", "confidence": "not a number"},
            {"text": "y", "confidence": 0.5},
        ])
        assert len(result) == 1


# ──────────────────────────────────────────────────────────────────────
# write_confidences_sidecar
# ──────────────────────────────────────────────────────────────────────


class TestWriteSidecar:
    def test_writes_json_at_expected_path(self, tmp_path: Path) -> None:
        text_path = tmp_path / "doc.txt"
        text_path.write_text("Hello world", encoding="utf-8")
        artifact = write_confidences_sidecar(
            text_path=text_path,
            adapter_name="tesseract",
            tokens=[{"text": "Hello", "confidence": 0.9}],
            document_id="doc01",
            extractor="tesseract",
        )
        sidecar = tmp_path / "doc.tesseract.confidences.json"
        assert sidecar.exists()
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        assert payload["tokens"] == [
            {"text": "Hello", "confidence": 0.9},
        ]
        assert payload["extractor"] == "tesseract"
        assert payload["model_version"] is None
        # Artifact CONFIDENCES.
        assert artifact.type == ArtifactType.CONFIDENCES
        assert artifact.uri == str(sidecar)
        assert artifact.id == "doc01:tesseract:confidences"

    def test_unicode_preserved(self, tmp_path: Path) -> None:
        text_path = tmp_path / "doc.txt"
        text_path.write_text("ok", encoding="utf-8")
        write_confidences_sidecar(
            text_path=text_path,
            adapter_name="tesseract",
            tokens=[{"text": "français", "confidence": 0.9}],
            document_id="doc01",
        )
        sidecar = tmp_path / "doc.tesseract.confidences.json"
        # ensure_ascii=False → caractères Unicode bruts.
        assert "français" in sidecar.read_text(encoding="utf-8")

    def test_model_version_when_provided(self, tmp_path: Path) -> None:
        text_path = tmp_path / "doc.txt"
        text_path.write_text("ok", encoding="utf-8")
        write_confidences_sidecar(
            text_path=text_path,
            adapter_name="tesseract",
            tokens=[],
            document_id="doc01",
            model_version="5.3.0",
        )
        sidecar = tmp_path / "doc.tesseract.confidences.json"
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        assert payload["model_version"] == "5.3.0"


# ──────────────────────────────────────────────────────────────────────
# Intégration TesseractAdapter
# ──────────────────────────────────────────────────────────────────────


def _make_image_artifact(uri: str) -> Artifact:
    return Artifact(
        id="d1:img",
        document_id="d1",
        type=ArtifactType.IMAGE,
        uri=uri,
    )


def _make_context() -> RunContext:
    return RunContext(
        document_id="d1",
        code_version="1.0.0",
        pipeline_name="test",
    )


class TestTesseractConfidenceIntegration:
    def _create_dummy_image(self, tmp_path: Path) -> Path:
        path = tmp_path / "page.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n")
        return path

    @patch("PIL.Image.open")
    @patch("pytesseract.image_to_string")
    @patch("pytesseract.image_to_data")
    def test_sidecar_produced_by_default(
        self,
        mock_image_to_data: MagicMock,
        mock_image_to_string: MagicMock,
        mock_image_open: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_image_to_string.return_value = "Hello world"
        mock_image_to_data.return_value = {
            "text": ["Hello", "world"],
            "conf": [95, 88],
        }
        mock_image_open.return_value.__enter__.return_value = MagicMock()

        adapter = TesseractAdapter()  # expose_confidences=True par défaut
        image_path = self._create_dummy_image(tmp_path)
        result = adapter.execute(
            inputs={ArtifactType.IMAGE: _make_image_artifact(str(image_path))},
            params={},
            context=_make_context(),
        )
        # Outputs : RAW_TEXT + CONFIDENCES.
        assert ArtifactType.RAW_TEXT in result
        assert ArtifactType.CONFIDENCES in result
        sidecar_path = Path(result[ArtifactType.CONFIDENCES].uri)
        assert sidecar_path.exists()
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert payload["tokens"] == [
            {"text": "Hello", "confidence": 0.95},
            {"text": "world", "confidence": 0.88},
        ]
        assert payload["extractor"] == "tesseract"

    @patch("PIL.Image.open")
    @patch("pytesseract.image_to_string")
    def test_no_sidecar_when_expose_confidences_false(
        self,
        mock_image_to_string: MagicMock,
        mock_image_open: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_image_to_string.return_value = "Hello world"
        mock_image_open.return_value.__enter__.return_value = MagicMock()
        adapter = TesseractAdapter(expose_confidences=False)
        image_path = self._create_dummy_image(tmp_path)
        result = adapter.execute(
            inputs={ArtifactType.IMAGE: _make_image_artifact(str(image_path))},
            params={},
            context=_make_context(),
        )
        # Pas de CONFIDENCES dans les outputs.
        assert ArtifactType.RAW_TEXT in result
        assert ArtifactType.CONFIDENCES not in result
        # Pas de sidecar sur disque.
        sidecars = list(tmp_path.glob("*.confidences.json"))
        assert sidecars == []

    @patch("PIL.Image.open")
    @patch("pytesseract.image_to_string")
    @patch("pytesseract.image_to_data")
    def test_extraction_failure_is_graceful(
        self,
        mock_image_to_data: MagicMock,
        mock_image_to_string: MagicMock,
        mock_image_open: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Si image_to_data plante, l'OCR doit malgré tout produire
        RAW_TEXT — seule la calibration est sautée pour ce document."""
        mock_image_to_string.return_value = "Hello world"
        mock_image_to_data.side_effect = RuntimeError(
            "image_to_data crashed",
        )
        mock_image_open.return_value.__enter__.return_value = MagicMock()
        adapter = TesseractAdapter()
        image_path = self._create_dummy_image(tmp_path)
        result = adapter.execute(
            inputs={ArtifactType.IMAGE: _make_image_artifact(str(image_path))},
            params={},
            context=_make_context(),
        )
        assert ArtifactType.RAW_TEXT in result
        # CONFIDENCES absent — extraction a échoué silencieusement.
        assert ArtifactType.CONFIDENCES not in result
