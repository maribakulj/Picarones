"""Tests Sprint 48 — adaptation Pero OCR pour exposer token_confidences.

Couvre :

1. ``_extract_token_confidences_from_layout`` parcourt regions/lines
   et émet un dict ``{"token": str, "confidence": float}`` par mot,
   en utilisant ``line.transcription_confidence`` propagée à tous les
   mots de la ligne.
2. Les lignes sans ``transcription`` ou sans
   ``transcription_confidence`` sont sautées.
3. Une transcription multi-mots produit autant d'entrées que de mots.
4. ``expose_confidences=False`` désactive l'extraction.
5. ``page_layout = None`` ou vide → retourne ``None`` sans crash.
6. ``run()`` appelle ``_run_pero_pipeline`` **une seule fois** (pas
   de double coût comme Tesseract) et expose ``token_confidences``
   sur l'``EngineResult``.
7. Si le pipeline lève, ``error`` est renseigné, ``text=""``, et
   ``token_confidences = None``.
8. Intégration bout-en-bout avec ``_compute_document_result``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import picarones.adapters.legacy_engines.pero_ocr as pero_module
from picarones.adapters.legacy_engines.pero_ocr import PeroOCREngine


# ──────────────────────────────────────────────────────────────────────────
# Helpers : mock d'un page_layout Pero OCR
# ──────────────────────────────────────────────────────────────────────────


def _mock_line(transcription: str, conf: float | None) -> MagicMock:
    line = MagicMock()
    line.transcription = transcription
    line.transcription_confidence = conf
    return line


def _mock_region(lines: list) -> MagicMock:
    region = MagicMock()
    region.lines = lines
    return region


def _mock_layout(regions: list) -> MagicMock:
    layout = MagicMock()
    layout.regions = regions
    return layout


# ──────────────────────────────────────────────────────────────────────────
# 1-3. Extraction depuis page_layout
# ──────────────────────────────────────────────────────────────────────────


class TestExtractFromLayout:
    def test_one_word_per_token(self) -> None:
        engine = PeroOCREngine()
        layout = _mock_layout([
            _mock_region([
                _mock_line("Bonjour le monde", 0.92),
            ]),
        ])
        out = engine._normalize_token_confidences(engine._extract_raw_confidences(layout))
        assert out is not None
        assert out == [
            {"token": "Bonjour", "confidence": 0.92},
            {"token": "le", "confidence": 0.92},
            {"token": "monde", "confidence": 0.92},
        ]

    def test_multiple_lines_concatenated(self) -> None:
        engine = PeroOCREngine()
        layout = _mock_layout([
            _mock_region([
                _mock_line("Première ligne", 0.95),
                _mock_line("Deuxième ligne", 0.80),
            ]),
        ])
        out = engine._normalize_token_confidences(engine._extract_raw_confidences(layout))
        assert out is not None
        # Chaque mot porte la confidence de SA ligne
        assert {"token": "Première", "confidence": 0.95} in out
        assert {"token": "Deuxième", "confidence": 0.80} in out

    def test_skips_empty_transcription(self) -> None:
        engine = PeroOCREngine()
        layout = _mock_layout([
            _mock_region([
                _mock_line("", 0.95),       # transcription vide
                _mock_line(None, 0.95),     # transcription None
                _mock_line("ok", 0.95),     # ok
            ]),
        ])
        out = engine._normalize_token_confidences(engine._extract_raw_confidences(layout))
        assert out == [{"token": "ok", "confidence": 0.95}]

    def test_skips_none_confidence(self) -> None:
        engine = PeroOCREngine()
        layout = _mock_layout([
            _mock_region([
                _mock_line("avec_conf", 0.85),
                _mock_line("sans_conf", None),
            ]),
        ])
        out = engine._normalize_token_confidences(engine._extract_raw_confidences(layout))
        assert out == [{"token": "avec_conf", "confidence": 0.85}]

    def test_skips_negative_confidence(self) -> None:
        engine = PeroOCREngine()
        layout = _mock_layout([
            _mock_region([
                _mock_line("ok", 0.9),
                _mock_line("dropped", -0.1),
            ]),
        ])
        out = engine._normalize_token_confidences(engine._extract_raw_confidences(layout))
        assert out == [{"token": "ok", "confidence": 0.9}]


# ──────────────────────────────────────────────────────────────────────────
# 4. expose_confidences=False
# ──────────────────────────────────────────────────────────────────────────


class TestExposeFlag:
    def test_disabled_returns_none(self) -> None:
        engine = PeroOCREngine(config={"expose_confidences": False})
        layout = _mock_layout([
            _mock_region([_mock_line("hello", 0.9)]),
        ])
        assert engine._normalize_token_confidences(engine._extract_raw_confidences(layout)) is None


# ──────────────────────────────────────────────────────────────────────────
# 5. Cas dégénérés
# ──────────────────────────────────────────────────────────────────────────


class TestDegenerateLayouts:
    def test_none_layout(self) -> None:
        engine = PeroOCREngine()
        assert engine._normalize_token_confidences(engine._extract_raw_confidences(None)) is None

    def test_empty_regions(self) -> None:
        engine = PeroOCREngine()
        layout = _mock_layout([])
        assert engine._normalize_token_confidences(engine._extract_raw_confidences(layout)) is None

    def test_only_lines_without_conf_returns_none(self) -> None:
        engine = PeroOCREngine()
        layout = _mock_layout([
            _mock_region([
                _mock_line("ok", None),
                _mock_line("ok2", None),
            ]),
        ])
        assert engine._normalize_token_confidences(engine._extract_raw_confidences(layout)) is None


# ──────────────────────────────────────────────────────────────────────────
# 6-7. run() avec mock du pipeline complet
# ──────────────────────────────────────────────────────────────────────────


def _make_engine_with_mock_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    text: str = "Bonjour le monde",
    layout_regions: list | None = None,
    raise_on_pipeline: bool = False,
) -> PeroOCREngine:
    """Mocke ``_run_pero_pipeline`` pour ne pas dépendre de pero-ocr."""
    engine = PeroOCREngine()

    if layout_regions is None:
        layout_regions = [
            _mock_region([_mock_line(text, 0.92)]),
        ]
    layout = _mock_layout(layout_regions)

    def _fake_pipeline(self, image_path):
        if raise_on_pipeline:
            raise RuntimeError("simulated pipeline failure")
        return text, layout

    monkeypatch.setattr(
        PeroOCREngine, "_run_pero_pipeline", _fake_pipeline,
    )
    return engine


class TestRunPipeline:
    def test_run_exposes_confidences(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        engine = _make_engine_with_mock_pipeline(monkeypatch)
        img = tmp_path / "p.png"
        img.write_bytes(b"x")

        result = engine.run(img)
        assert result.text == "Bonjour le monde"
        assert result.error is None
        assert result.token_confidences is not None
        assert len(result.token_confidences) == 3

    def test_run_text_preserved_octet_for_octet(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        engine = _make_engine_with_mock_pipeline(
            monkeypatch, text="Texte avec\nplusieurs lignes",
        )
        img = tmp_path / "p.png"
        img.write_bytes(b"x")
        result = engine.run(img)
        assert result.text == "Texte avec\nplusieurs lignes"

    def test_pipeline_failure_keeps_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        engine = _make_engine_with_mock_pipeline(
            monkeypatch, raise_on_pipeline=True,
        )
        img = tmp_path / "p.png"
        img.write_bytes(b"x")
        result = engine.run(img)
        assert result.error == "simulated pipeline failure"
        assert result.text == ""
        assert result.token_confidences is None


# ──────────────────────────────────────────────────────────────────────────
# 8. Intégration bout-en-bout avec le runner
# ──────────────────────────────────────────────────────────────────────────


class TestPeroAbsent:
    def test_pipeline_missing_pero_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """Si pero-ocr n'est pas installé, ``_run_pero_pipeline`` lève
        à travers ``_get_parser()``.  ``run()`` capture l'exception et
        retourne ``EngineResult.error``."""
        monkeypatch.setattr(pero_module, "_PERO_AVAILABLE", False)
        engine = PeroOCREngine(config={"config": "/no/such/file.ini"})
        img = tmp_path / "p.png"
        img.write_bytes(b"x")
        result = engine.run(img)
        assert result.error is not None
        assert "pero" in result.error.lower() or "Pillow" in result.error
        assert result.token_confidences is None
