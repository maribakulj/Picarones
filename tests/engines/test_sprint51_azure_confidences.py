"""Tests Sprint 51 — adaptation Azure Document Intelligence pour exposer
token_confidences.

Couvre :

1. ``_extract_token_confidences_from_result`` parcourt
   ``pages[].words[]`` et émet ``{"token": content, "confidence": float}``
   par mot.
2. Filtrage des mots sans confidence, conf négative, contenu vide.
3. ``expose_confidences=False`` désactive l'extraction.
4. ``analyze_result = None`` ou structures invalides → retourne ``None``.
5. ``_sdk_result_to_dict`` convertit un objet SDK proto en dict
   normalisé compatible avec le chemin REST.
6. ``run()`` orchestre les deux chemins (SDK + REST) et expose les
   confidences sur l'``EngineResult``.
7. Échec API → ``error`` renseigné, ``token_confidences = None``.
8. Intégration runner : ``calibration_metrics`` calculée bout-en-bout.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from picarones.adapters.legacy_engines.azure_doc_intel import AzureDocIntelEngine


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _word(content: str, conf: float | None) -> dict:
    return {"content": content, "confidence": conf}


def _result(words: list[dict]) -> dict:
    return {"pages": [{"words": words}]}


# ──────────────────────────────────────────────────────────────────────────
# 1-2. Extraction depuis analyze_result
# ──────────────────────────────────────────────────────────────────────────


class TestExtractFromResult:
    def test_emits_one_entry_per_word(self) -> None:
        engine = AzureDocIntelEngine()
        result = _result([
            _word("Bonjour", 0.97),
            _word("monde", 0.93),
        ])
        out = engine._normalize_token_confidences(engine._extract_raw_confidences(result))
        assert out == [
            {"token": "Bonjour", "confidence": 0.97},
            {"token": "monde",   "confidence": 0.93},
        ]

    def test_skips_word_without_confidence(self) -> None:
        engine = AzureDocIntelEngine()
        result = _result([
            _word("ok", 0.95),
            {"content": "no_conf"},      # pas de confidence
            _word("none_conf", None),
        ])
        out = engine._normalize_token_confidences(engine._extract_raw_confidences(result))
        assert out == [{"token": "ok", "confidence": 0.95}]

    def test_skips_negative_confidence(self) -> None:
        engine = AzureDocIntelEngine()
        result = _result([
            _word("ok", 0.9),
            _word("dropped", -0.1),
        ])
        out = engine._normalize_token_confidences(engine._extract_raw_confidences(result))
        assert out == [{"token": "ok", "confidence": 0.9}]

    def test_skips_empty_content(self) -> None:
        engine = AzureDocIntelEngine()
        result = _result([
            _word("", 0.95),
            _word("ok", 0.9),
        ])
        out = engine._normalize_token_confidences(engine._extract_raw_confidences(result))
        assert out == [{"token": "ok", "confidence": 0.9}]

    def test_traverses_multiple_pages(self) -> None:
        engine = AzureDocIntelEngine()
        result = {
            "pages": [
                {"words": [_word("alpha", 0.9), _word("beta", 0.85)]},
                {"words": [_word("gamma", 0.8)]},
            ],
        }
        out = engine._normalize_token_confidences(engine._extract_raw_confidences(result))
        assert [tc["token"] for tc in (out or [])] == ["alpha", "beta", "gamma"]


# ──────────────────────────────────────────────────────────────────────────
# 3. expose_confidences=False
# ──────────────────────────────────────────────────────────────────────────


class TestExposeFlag:
    def test_disabled_returns_none(self) -> None:
        engine = AzureDocIntelEngine(config={"expose_confidences": False})
        assert engine._normalize_token_confidences(
            engine._extract_raw_confidences(_result([_word("ok", 0.9)])),
        ) is None


# ──────────────────────────────────────────────────────────────────────────
# 4. Cas dégénérés
# ──────────────────────────────────────────────────────────────────────────


class TestDegenerateInputs:
    def test_none(self) -> None:
        engine = AzureDocIntelEngine()
        assert engine._normalize_token_confidences(engine._extract_raw_confidences(None)) is None

    def test_empty_dict(self) -> None:
        engine = AzureDocIntelEngine()
        assert engine._normalize_token_confidences(engine._extract_raw_confidences({})) is None

    def test_no_pages(self) -> None:
        engine = AzureDocIntelEngine()
        assert engine._normalize_token_confidences(engine._extract_raw_confidences(
            {"pages": []},
        )) is None

    def test_pages_without_words(self) -> None:
        engine = AzureDocIntelEngine()
        assert engine._normalize_token_confidences(engine._extract_raw_confidences(
            {"pages": [{"lines": [{"content": "no words"}]}]},
        )) is None


# ──────────────────────────────────────────────────────────────────────────
# 5. Conversion SDK → dict
# ──────────────────────────────────────────────────────────────────────────


class TestSdkConversion:
    def test_sdk_to_dict(self) -> None:
        # Mock du proto SDK
        word_mock = MagicMock()
        word_mock.content = "Bonjour"
        word_mock.confidence = 0.97
        page_mock = MagicMock()
        page_mock.words = [word_mock]
        result_mock = MagicMock()
        result_mock.pages = [page_mock]

        out = AzureDocIntelEngine._sdk_result_to_dict(result_mock)
        assert "pages" in out
        assert out["pages"][0]["words"][0]["content"] == "Bonjour"
        assert out["pages"][0]["words"][0]["confidence"] == pytest.approx(0.97)

    def test_sdk_word_with_none_confidence(self) -> None:
        word_mock = MagicMock()
        word_mock.content = "ok"
        word_mock.confidence = None
        page_mock = MagicMock()
        page_mock.words = [word_mock]
        result_mock = MagicMock()
        result_mock.pages = [page_mock]

        out = AzureDocIntelEngine._sdk_result_to_dict(result_mock)
        assert out["pages"][0]["words"][0]["confidence"] is None


# ──────────────────────────────────────────────────────────────────────────
# 6-7. run() avec mock
# ──────────────────────────────────────────────────────────────────────────


def _patch_run_with_result(
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    analyze_result: dict | None,
    *,
    raise_exc: Exception | None = None,
) -> AzureDocIntelEngine:
    engine = AzureDocIntelEngine()
    engine._api_key = "test-key"
    engine._endpoint = "https://test.cognitiveservices.azure.com"

    def _fake(self, image_path):
        if raise_exc is not None:
            raise raise_exc
        return text, analyze_result

    monkeypatch.setattr(
        AzureDocIntelEngine, "_run_with_native", _fake,
    )
    return engine


class TestRunOverride:
    def test_run_exposes_confidences(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        engine = _patch_run_with_result(
            monkeypatch,
            text="Bonjour\nmonde",
            analyze_result=_result([
                _word("Bonjour", 0.97),
                _word("monde", 0.93),
            ]),
        )
        img = tmp_path / "p.png"
        img.write_bytes(b"x")
        result = engine.run(img)
        assert result.text == "Bonjour\nmonde"
        assert result.error is None
        assert result.token_confidences is not None
        assert len(result.token_confidences) == 2

    def test_run_no_result_no_confidences(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        engine = _patch_run_with_result(
            monkeypatch, text="Texte", analyze_result=None,
        )
        img = tmp_path / "p.png"
        img.write_bytes(b"x")
        result = engine.run(img)
        assert result.text == "Texte"
        assert result.token_confidences is None

    def test_run_api_failure_keeps_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        engine = _patch_run_with_result(
            monkeypatch, text="", analyze_result=None,
            raise_exc=RuntimeError("Azure timeout"),
        )
        img = tmp_path / "p.png"
        img.write_bytes(b"x")
        result = engine.run(img)
        assert result.error == "Azure timeout"
        assert result.token_confidences is None


# ──────────────────────────────────────────────────────────────────────────
# 8. Intégration runner
# ──────────────────────────────────────────────────────────────────────────


