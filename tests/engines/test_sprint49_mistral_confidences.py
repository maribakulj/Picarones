"""Tests Sprint 49 — adaptation Mistral OCR pour exposer token_confidences.

Couvre :

1. ``_extract_token_confidences_from_response`` :
   - extrait les words explicites avec ``{"text", "confidence"}``
   - propage la confidence d'une ligne / bloc à chaque mot
   - ignore les entrées sans confidence ou avec confidence négative
2. Réponse vide / None / sans pages → retourne ``None``.
3. ``expose_confidences=False`` désactive l'extraction.
4. ``run()`` appelle ``_run_ocr_with_response`` et stocke les
   confidences dans ``EngineResult.token_confidences``.
5. Le chemin chat/vision (``pixtral-*``) renvoie
   ``raw_response = None`` → ``token_confidences = None``.
6. Si l'API échoue, ``error`` renseigné, ``text=""``,
   ``token_confidences = None``.
7. Intégration bout-en-bout avec ``_compute_document_result``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from picarones.adapters.legacy_engines.mistral_ocr import MistralOCREngine


# ──────────────────────────────────────────────────────────────────────────
# 1. Extraction depuis une réponse JSON Mistral
# ──────────────────────────────────────────────────────────────────────────


class TestExtractFromResponse:
    def test_extract_words_explicit(self) -> None:
        engine = MistralOCREngine()
        response = {
            "pages": [{
                "words": [
                    {"text": "Bonjour", "confidence": 0.95},
                    {"text": "monde",   "confidence": 0.90},
                ],
            }],
        }
        out = engine._normalize_token_confidences(engine._extract_raw_confidences(response))
        assert out == [
            {"token": "Bonjour", "confidence": 0.95},
            {"token": "monde",   "confidence": 0.90},
        ]

    def test_lines_propagate_confidence_to_words(self) -> None:
        engine = MistralOCREngine()
        response = {
            "pages": [{
                "lines": [
                    {"text": "première ligne", "confidence": 0.88},
                    {"text": "seconde",        "confidence": 0.75},
                ],
            }],
        }
        out = engine._normalize_token_confidences(engine._extract_raw_confidences(response))
        assert out is not None
        # 3 tokens (2 mots + 1 mot), avec leurs confidences respectives
        assert {"token": "première", "confidence": 0.88} in out
        assert {"token": "ligne",    "confidence": 0.88} in out
        assert {"token": "seconde",  "confidence": 0.75} in out

    def test_blocks_propagate_confidence(self) -> None:
        engine = MistralOCREngine()
        response = {
            "pages": [{
                "blocks": [
                    {"text": "bloc1 mot2", "confidence": 0.82},
                ],
            }],
        }
        out = engine._normalize_token_confidences(engine._extract_raw_confidences(response))
        assert out == [
            {"token": "bloc1", "confidence": 0.82},
            {"token": "mot2",  "confidence": 0.82},
        ]

    def test_skips_empty_text(self) -> None:
        engine = MistralOCREngine()
        response = {
            "pages": [{
                "words": [
                    {"text": "", "confidence": 0.9},
                    {"text": "ok", "confidence": 0.9},
                ],
            }],
        }
        out = engine._normalize_token_confidences(engine._extract_raw_confidences(response))
        assert out == [{"token": "ok", "confidence": 0.9}]

    def test_skips_none_confidence(self) -> None:
        engine = MistralOCREngine()
        response = {
            "pages": [{
                "words": [
                    {"text": "avec_conf", "confidence": 0.85},
                    {"text": "sans_conf"},
                    {"text": "explicit_none", "confidence": None},
                ],
            }],
        }
        out = engine._normalize_token_confidences(engine._extract_raw_confidences(response))
        assert out == [{"token": "avec_conf", "confidence": 0.85}]

    def test_skips_negative_confidence(self) -> None:
        engine = MistralOCREngine()
        response = {
            "pages": [{
                "words": [
                    {"text": "ok", "confidence": 0.9},
                    {"text": "neg", "confidence": -0.1},
                ],
            }],
        }
        out = engine._normalize_token_confidences(engine._extract_raw_confidences(response))
        assert out == [{"token": "ok", "confidence": 0.9}]

    def test_combines_words_and_lines(self) -> None:
        engine = MistralOCREngine()
        response = {
            "pages": [{
                "words": [{"text": "explicit", "confidence": 0.99}],
                "lines": [{"text": "ligne mots", "confidence": 0.7}],
            }],
        }
        out = engine._normalize_token_confidences(engine._extract_raw_confidences(response))
        assert out is not None
        assert len(out) == 3  # 1 word explicit + 2 mots de la ligne


# ──────────────────────────────────────────────────────────────────────────
# 2. Cas dégénérés
# ──────────────────────────────────────────────────────────────────────────


class TestDegenerateResponses:
    def test_none_response(self) -> None:
        engine = MistralOCREngine()
        assert engine._normalize_token_confidences(engine._extract_raw_confidences(None)) is None

    def test_empty_dict(self) -> None:
        engine = MistralOCREngine()
        assert engine._normalize_token_confidences(engine._extract_raw_confidences({})) is None

    def test_no_pages(self) -> None:
        engine = MistralOCREngine()
        assert engine._normalize_token_confidences(engine._extract_raw_confidences(
            {"pages": []},
        )) is None

    def test_pages_without_confidences(self) -> None:
        engine = MistralOCREngine()
        response = {
            "pages": [
                {"markdown": "Texte sans annotation de confidence"},
            ],
        }
        assert engine._normalize_token_confidences(engine._extract_raw_confidences(response)) is None

    def test_non_dict_input(self) -> None:
        engine = MistralOCREngine()
        assert engine._normalize_token_confidences(engine._extract_raw_confidences("not a dict")) is None
        assert engine._normalize_token_confidences(engine._extract_raw_confidences([1, 2, 3])) is None


# ──────────────────────────────────────────────────────────────────────────
# 3. expose_confidences=False
# ──────────────────────────────────────────────────────────────────────────


class TestExposeFlag:
    def test_disabled_returns_none(self) -> None:
        engine = MistralOCREngine(config={"expose_confidences": False})
        response = {
            "pages": [{
                "words": [{"text": "ok", "confidence": 0.9}],
            }],
        }
        assert engine._normalize_token_confidences(engine._extract_raw_confidences(response)) is None


# ──────────────────────────────────────────────────────────────────────────
# 4-6. run() avec mock du chemin réseau
# ──────────────────────────────────────────────────────────────────────────


def _mock_run_with_response(
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    raw_response: dict | None,
    *,
    raise_exc: Exception | None = None,
) -> MistralOCREngine:
    """Patche ``_run_ocr_with_response`` pour ne pas appeler l'API."""
    engine = MistralOCREngine()
    # On évite la vérification de la clé API (set artificiellement)
    engine._api_key = "test-key"

    def _fake(self, image_path):
        if raise_exc is not None:
            raise raise_exc
        return text, raw_response

    monkeypatch.setattr(
        MistralOCREngine, "_run_with_native", _fake,
    )
    return engine


class TestRunOverride:
    def test_run_exposes_confidences_when_response_has_them(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        engine = _mock_run_with_response(
            monkeypatch,
            "Bonjour le monde",
            {"pages": [{
                "words": [
                    {"text": "Bonjour", "confidence": 0.95},
                    {"text": "le",      "confidence": 0.92},
                    {"text": "monde",   "confidence": 0.90},
                ],
            }]},
        )
        img = tmp_path / "p.png"
        img.write_bytes(b"x")
        result = engine.run(img)
        assert result.text == "Bonjour le monde"
        assert result.error is None
        assert result.token_confidences is not None
        assert len(result.token_confidences) == 3

    def test_run_no_confidences_when_chat_vision(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """Chemin pixtral : raw_response = None → token_confidences = None."""
        engine = _mock_run_with_response(
            monkeypatch,
            "Texte produit par pixtral",
            None,  # le chemin chat/vision ne fournit pas de raw_response
        )
        img = tmp_path / "p.png"
        img.write_bytes(b"x")
        result = engine.run(img)
        assert result.text == "Texte produit par pixtral"
        assert result.token_confidences is None

    def test_run_api_failure_keeps_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        engine = _mock_run_with_response(
            monkeypatch,
            "",
            None,
            raise_exc=RuntimeError("API timeout"),
        )
        img = tmp_path / "p.png"
        img.write_bytes(b"x")
        result = engine.run(img)
        assert result.error == "API timeout"
        assert result.text == ""
        assert result.token_confidences is None


# ──────────────────────────────────────────────────────────────────────────
# 7. Intégration runner
# ──────────────────────────────────────────────────────────────────────────


class TestEndToEndWithRunner:
    def test_runner_picks_up_mistral_confidences(self) -> None:
        from picarones.measurements.runner import _compute_document_result
        from picarones.adapters.legacy_engines.base import EngineResult

        ocr = EngineResult(
            engine_name="mistral_ocr",
            image_path="/tmp/x.png",
            text="alpha beta gamma",
            duration_seconds=0.1,
            token_confidences=[
                {"token": "alpha", "confidence": 0.95},
                {"token": "beta",  "confidence": 0.85},
                {"token": "gamma", "confidence": 0.95},
            ],
        )
        dr = _compute_document_result(
            doc_id="d1", image_path="/tmp/x.png",
            ground_truth="alpha beta gamma",
            ocr_result=ocr, char_exclude=None,
        )
        assert dr.calibration_metrics is not None
        assert dr.calibration_metrics["overall_accuracy"] == 1.0
        # confidence moyenne = (0.95 + 0.85 + 0.95) / 3
        assert dr.calibration_metrics["overall_confidence"] == pytest.approx(
            (0.95 + 0.85 + 0.95) / 3,
        )
