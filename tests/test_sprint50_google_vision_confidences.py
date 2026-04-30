"""Tests Sprint 50 — adaptation Google Vision pour exposer token_confidences.

Couvre :

1. ``_extract_token_confidences_from_full_text`` reconstruit chaque mot
   par concaténation des ``word.symbols[i].text`` et associe la
   ``word.confidence``.
2. Hiérarchie pages → blocks → paragraphs → words est traversée
   correctement (multi-pages, multi-blocks).
3. Mots sans confidence, conf négative, symboles vides → ignorés.
4. ``expose_confidences=False`` désactive l'extraction.
5. ``full_text_annotation = None`` (cas TEXT_DETECTION) → retourne
   ``None``.
6. ``run()`` orchestre les deux chemins :
   - SDK : ``response.full_text_annotation`` proto converti en dict
   - REST : ``r["fullTextAnnotation"]`` directement utilisé
   Le texte reste celui de ``full_text_annotation.text``
   (rétrocompat).
7. Échec API → ``error`` renseigné, ``token_confidences = None``.
8. Conversion SDK → dict normalisé : un mock proto est correctement
   sérialisé.
9. Intégration runner : ``calibration_metrics`` calculée bout-en-bout.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import picarones.engines.google_vision as gv_module
from picarones.engines.google_vision import GoogleVisionEngine


# ──────────────────────────────────────────────────────────────────────────
# Helpers : construire un fullTextAnnotation au format dict normalisé
# ──────────────────────────────────────────────────────────────────────────


def _word(text: str, conf: float) -> dict:
    return {
        "confidence": conf,
        "symbols": [{"text": c} for c in text],
    }


def _full_text(words: list[dict]) -> dict:
    return {
        "pages": [{
            "blocks": [{
                "paragraphs": [{"words": words}],
            }],
        }],
    }


# ──────────────────────────────────────────────────────────────────────────
# 1-3. Extraction depuis full_text_annotation
# ──────────────────────────────────────────────────────────────────────────


class TestExtractFromFullText:
    def test_reconstructs_word_from_symbols(self) -> None:
        engine = GoogleVisionEngine()
        full = _full_text([_word("Bonjour", 0.95)])
        assert engine._extract_token_confidences_from_full_text(full) == [
            {"token": "Bonjour", "confidence": 0.95},
        ]

    def test_multiple_words(self) -> None:
        engine = GoogleVisionEngine()
        full = _full_text([
            _word("Bonjour", 0.95),
            _word("monde", 0.88),
        ])
        out = engine._extract_token_confidences_from_full_text(full)
        assert out == [
            {"token": "Bonjour", "confidence": 0.95},
            {"token": "monde",   "confidence": 0.88},
        ]

    def test_skips_word_without_confidence(self) -> None:
        engine = GoogleVisionEngine()
        full = _full_text([
            {"confidence": 0.95, "symbols": [{"text": "ok"}]},
            {"symbols": [{"text": "nope"}]},          # pas de confidence
            {"confidence": None, "symbols": [{"text": "nope"}]},  # None
        ])
        out = engine._extract_token_confidences_from_full_text(full)
        assert out == [{"token": "ok", "confidence": 0.95}]

    def test_skips_negative_confidence(self) -> None:
        engine = GoogleVisionEngine()
        full = _full_text([
            _word("ok", 0.9),
            _word("dropped", -0.1),
        ])
        out = engine._extract_token_confidences_from_full_text(full)
        assert out == [{"token": "ok", "confidence": 0.9}]

    def test_skips_empty_text(self) -> None:
        engine = GoogleVisionEngine()
        full = _full_text([
            _word("", 0.95),
            _word("ok", 0.9),
        ])
        out = engine._extract_token_confidences_from_full_text(full)
        assert out == [{"token": "ok", "confidence": 0.9}]

    def test_traverses_multiple_pages_and_blocks(self) -> None:
        engine = GoogleVisionEngine()
        full = {
            "pages": [
                {"blocks": [
                    {"paragraphs": [{"words": [_word("alpha", 0.9)]}]},
                    {"paragraphs": [{"words": [_word("beta", 0.85)]}]},
                ]},
                {"blocks": [
                    {"paragraphs": [{"words": [_word("gamma", 0.8)]}]},
                ]},
            ],
        }
        out = engine._extract_token_confidences_from_full_text(full)
        assert out is not None
        tokens = [tc["token"] for tc in out]
        assert tokens == ["alpha", "beta", "gamma"]


# ──────────────────────────────────────────────────────────────────────────
# 4. expose_confidences=False
# ──────────────────────────────────────────────────────────────────────────


class TestExposeFlag:
    def test_disabled_returns_none(self) -> None:
        engine = GoogleVisionEngine(config={"expose_confidences": False})
        full = _full_text([_word("ok", 0.95)])
        assert engine._extract_token_confidences_from_full_text(full) is None


# ──────────────────────────────────────────────────────────────────────────
# 5. Cas dégénérés
# ──────────────────────────────────────────────────────────────────────────


class TestDegenerateInputs:
    def test_none(self) -> None:
        engine = GoogleVisionEngine()
        assert engine._extract_token_confidences_from_full_text(None) is None

    def test_empty_dict(self) -> None:
        engine = GoogleVisionEngine()
        assert engine._extract_token_confidences_from_full_text({}) is None

    def test_no_pages(self) -> None:
        engine = GoogleVisionEngine()
        assert engine._extract_token_confidences_from_full_text(
            {"pages": []},
        ) is None

    def test_pages_without_blocks(self) -> None:
        engine = GoogleVisionEngine()
        assert engine._extract_token_confidences_from_full_text(
            {"pages": [{"text": "raw text only"}]},
        ) is None


# ──────────────────────────────────────────────────────────────────────────
# 6. Conversion SDK → dict
# ──────────────────────────────────────────────────────────────────────────


class TestSdkConversion:
    def test_sdk_proto_to_dict(self) -> None:
        # Simule un proto SDK avec des objets attribut-based
        word_mock = MagicMock()
        word_mock.confidence = 0.92
        sym_b = MagicMock()
        sym_b.text = "B"
        sym_o = MagicMock()
        sym_o.text = "o"
        sym_n = MagicMock()
        sym_n.text = "n"
        word_mock.symbols = [sym_b, sym_o, sym_n]
        para_mock = MagicMock()
        para_mock.words = [word_mock]
        block_mock = MagicMock()
        block_mock.paragraphs = [para_mock]
        page_mock = MagicMock()
        page_mock.blocks = [block_mock]
        full_mock = MagicMock()
        full_mock.pages = [page_mock]

        result = GoogleVisionEngine._sdk_full_text_to_dict(full_mock)

        assert "pages" in result
        assert len(result["pages"]) == 1
        word = result["pages"][0]["blocks"][0]["paragraphs"][0]["words"][0]
        assert word["confidence"] == pytest.approx(0.92)
        assert "".join(s["text"] for s in word["symbols"]) == "Bon"


# ──────────────────────────────────────────────────────────────────────────
# 7. run() bout-en-bout via mock du chemin réseau
# ──────────────────────────────────────────────────────────────────────────


def _patch_run_with_full(
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    full: dict | None,
    *,
    raise_exc: Exception | None = None,
) -> GoogleVisionEngine:
    engine = GoogleVisionEngine()
    engine._api_key = "test"  # bypass auth check

    def _fake(self, image_path):
        if raise_exc is not None:
            raise raise_exc
        return text, full

    monkeypatch.setattr(
        GoogleVisionEngine, "_run_ocr_with_full_annotation", _fake,
    )
    return engine


class TestRunOverride:
    def test_run_exposes_confidences(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        engine = _patch_run_with_full(
            monkeypatch,
            text="Bonjour monde",
            full=_full_text([_word("Bonjour", 0.95), _word("monde", 0.88)]),
        )
        img = tmp_path / "p.png"
        img.write_bytes(b"x")
        result = engine.run(img)
        assert result.text == "Bonjour monde"
        assert result.error is None
        assert result.token_confidences is not None
        assert len(result.token_confidences) == 2

    def test_run_text_detection_no_confidences(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """TEXT_DETECTION : full = None → token_confidences = None."""
        engine = _patch_run_with_full(monkeypatch, text="Texte court", full=None)
        img = tmp_path / "p.png"
        img.write_bytes(b"x")
        result = engine.run(img)
        assert result.text == "Texte court"
        assert result.token_confidences is None

    def test_run_api_failure_keeps_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        engine = _patch_run_with_full(
            monkeypatch, text="", full=None,
            raise_exc=RuntimeError("Quota exceeded"),
        )
        img = tmp_path / "p.png"
        img.write_bytes(b"x")
        result = engine.run(img)
        assert result.error == "Quota exceeded"
        assert result.text == ""
        assert result.token_confidences is None


# ──────────────────────────────────────────────────────────────────────────
# 8. REST direct : parsing du JSON complet
# ──────────────────────────────────────────────────────────────────────────


class TestRESTPath:
    def test_rest_passes_full_text_through(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """Le chemin REST renvoie tel quel le ``fullTextAnnotation``
        du JSON, qui est un dict directement consommable par
        ``_extract_token_confidences_from_full_text``."""
        engine = GoogleVisionEngine()
        engine._api_key = "test-key"
        engine._credentials_path = None

        # Mock urllib.request.urlopen pour retourner une réponse REST
        # contenant un fullTextAnnotation complet.
        fake_response = json.dumps({
            "responses": [{
                "fullTextAnnotation": {
                    "text": "Bonjour",
                    **_full_text([_word("Bonjour", 0.97)]),
                },
            }],
        }).encode("utf-8")

        class FakeResp:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def read(self):
                return fake_response

        monkeypatch.setattr(
            gv_module.urllib.request, "urlopen",
            lambda req, timeout=60: FakeResp(),
        )

        img = tmp_path / "p.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")
        text, full = engine._run_via_rest(img)

        assert text == "Bonjour"
        assert full is not None
        assert "pages" in full

        # L'extraction passe ensuite normalement
        out = engine._extract_token_confidences_from_full_text(full)
        assert out == [{"token": "Bonjour", "confidence": 0.97}]


# ──────────────────────────────────────────────────────────────────────────
# 9. Intégration runner
# ──────────────────────────────────────────────────────────────────────────


class TestEndToEndWithRunner:
    def test_runner_picks_up_google_vision_confidences(self) -> None:
        from picarones.core.runner import _compute_document_result
        from picarones.engines.base import EngineResult

        ocr = EngineResult(
            engine_name="google_vision",
            image_path="/tmp/x.png",
            text="alpha beta gamma",
            duration_seconds=0.1,
            token_confidences=[
                {"token": "alpha", "confidence": 0.95},
                {"token": "beta",  "confidence": 0.92},
                {"token": "gamma", "confidence": 0.97},
            ],
        )
        dr = _compute_document_result(
            doc_id="d1", image_path="/tmp/x.png",
            ground_truth="alpha beta gamma",
            ocr_result=ocr, char_exclude=None,
        )
        assert dr.calibration_metrics is not None
        assert dr.calibration_metrics["overall_accuracy"] == 1.0
        assert dr.calibration_metrics["overall_confidence"] == pytest.approx(
            (0.95 + 0.92 + 0.97) / 3,
        )
