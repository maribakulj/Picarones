"""Tests Sprint 31 — couverture dédiée des moteurs OCR cloud.

Avant Sprint 31, ``picarones/engines/{mistral_ocr,google_vision,
azure_doc_intel}.py`` n'étaient testés que via les fixtures du runner —
ce qui signifiait qu'on ne déclenchait jamais leurs branches d'erreur
(clé manquante, endpoint manquant, HTTP 4xx/5xx, format de réponse
inattendu). Ce fichier mocke ``urllib.request.urlopen`` pour les trois
moteurs et vérifie :

- la **création** réussie sans clef API ne plante pas (clés sont lues
  paresseusement dans ``_run_ocr``) ;
- l'**absence de clé** lève ``RuntimeError`` avec un message qui
  pointe vers la bonne variable d'environnement ;
- le **happy path REST** retourne le texte attendu d'une réponse JSON
  fictive ;
- les **erreurs HTTP** sont remontées en ``RuntimeError`` lisibles ;
- les **propriétés** ``name``, ``version`` et ``execution_mode``
  sont déclarées correctement (Sprint 31 — moteurs cloud doivent
  hériter de ``execution_mode='io'`` du parent).
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

import pytest


# ---------------------------------------------------------------------------
# Fixture utilitaire — image PNG minimale
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_image(tmp_path: Path) -> Path:
    """Crée un PNG 10x10 décodable par Pillow."""
    from PIL import Image
    p = tmp_path / "test.png"
    Image.new("RGB", (10, 10), color=(120, 120, 120)).save(p, format="PNG")
    return p


def _mock_urlopen_response(json_body: dict, headers: dict | None = None) -> MagicMock:
    """Construit un faux ``urlopen`` context manager qui retourne ``json_body``."""
    raw = json.dumps(json_body).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = raw
    mock_resp.headers = headers or {}
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_resp
    mock_cm.__exit__.return_value = False
    return mock_cm


# ---------------------------------------------------------------------------
# 1. MistralOCREngine
# ---------------------------------------------------------------------------

class TestMistralOCREngine:
    def test_class_metadata(self, monkeypatch):
        from picarones.engines.mistral_ocr import MistralOCREngine
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        eng = MistralOCREngine()
        assert eng.name == "mistral_ocr"
        assert eng.version()  # retourne un str non vide
        # Sprint 24/31 — execution_mode hérite de la valeur 'io' du parent
        assert eng.execution_mode == "io"

    def test_missing_api_key_raises(self, monkeypatch, fake_image):
        from picarones.engines.mistral_ocr import MistralOCREngine
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        eng = MistralOCREngine()
        with pytest.raises(RuntimeError, match="MISTRAL_API_KEY"):
            eng._run_ocr(fake_image)

    def test_native_ocr_endpoint_parses_pages(self, monkeypatch, fake_image):
        """``mistral-ocr-latest`` route vers ``/v1/ocr`` et concatène les pages."""
        from picarones.engines.mistral_ocr import MistralOCREngine
        monkeypatch.setenv("MISTRAL_API_KEY", "fake-key")
        eng = MistralOCREngine(config={"model": "mistral-ocr-latest"})

        body = {
            "pages": [
                {"markdown": "Page 1 — Lorem ipsum"},
                {"markdown": "Page 2 — dolor sit amet"},
            ],
        }
        with patch("urllib.request.urlopen", return_value=_mock_urlopen_response(body)):
            text = eng._run_ocr(fake_image)
        assert "Page 1" in text
        assert "Page 2" in text
        # Concaténation par double saut de ligne
        assert "\n\n" in text

    def test_native_endpoint_handles_empty_pages(self, monkeypatch, fake_image):
        from picarones.engines.mistral_ocr import MistralOCREngine
        monkeypatch.setenv("MISTRAL_API_KEY", "fake-key")
        eng = MistralOCREngine(config={"model": "mistral-ocr-latest"})

        with patch("urllib.request.urlopen",
                   return_value=_mock_urlopen_response({"pages": []})):
            text = eng._run_ocr(fake_image)
        assert text == ""


# ---------------------------------------------------------------------------
# 2. GoogleVisionEngine
# ---------------------------------------------------------------------------

class TestGoogleVisionEngine:
    def test_class_metadata(self, monkeypatch):
        from picarones.engines.google_vision import GoogleVisionEngine
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        eng = GoogleVisionEngine()
        assert eng.name == "google_vision"
        assert eng.version() == "v1"
        assert eng.execution_mode == "io"

    def test_missing_credentials_raises(self, monkeypatch, fake_image):
        from picarones.engines.google_vision import GoogleVisionEngine
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        eng = GoogleVisionEngine()
        with pytest.raises(RuntimeError, match="(?i)Authentification"):
            eng._run_ocr(fake_image)

    def test_rest_happy_path_extracts_text(self, monkeypatch, fake_image):
        from picarones.engines.google_vision import GoogleVisionEngine
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        eng = GoogleVisionEngine()

        body = {
            "responses": [
                {"fullTextAnnotation": {"text": "Texte reconstitué de Gallica"}},
            ],
        }
        with patch("urllib.request.urlopen", return_value=_mock_urlopen_response(body)):
            text = eng._run_ocr(fake_image)
        assert text == "Texte reconstitué de Gallica"

    def test_rest_response_with_error_field_raises(self, monkeypatch, fake_image):
        from picarones.engines.google_vision import GoogleVisionEngine
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        eng = GoogleVisionEngine()

        body = {"responses": [{"error": {"message": "Quota exhausted"}}]}
        with patch("urllib.request.urlopen", return_value=_mock_urlopen_response(body)):
            with pytest.raises(RuntimeError, match="(?i)Quota"):
                eng._run_ocr(fake_image)

    def test_http_error_remontes_lisible(self, monkeypatch, fake_image):
        from picarones.engines.google_vision import GoogleVisionEngine
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        eng = GoogleVisionEngine()

        err = HTTPError(
            url="https://vision.googleapis.com/v1/images:annotate",
            code=400,
            msg="Bad Request",
            hdrs=None,  # type: ignore[arg-type]
            fp=io.BytesIO(b'{"error": "bad image"}'),
        )
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(RuntimeError, match="(?i)400"):
                eng._run_ocr(fake_image)

    def test_text_detection_extracts_first_annotation(self, monkeypatch, fake_image):
        from picarones.engines.google_vision import GoogleVisionEngine
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        eng = GoogleVisionEngine(config={"feature_type": "TEXT_DETECTION"})

        body = {
            "responses": [{
                "textAnnotations": [
                    {"description": "Premier annot"},
                    {"description": "Second annot"},
                ],
            }],
        }
        with patch("urllib.request.urlopen", return_value=_mock_urlopen_response(body)):
            text = eng._run_ocr(fake_image)
        assert text == "Premier annot"


# ---------------------------------------------------------------------------
# 3. AzureDocIntelEngine
# ---------------------------------------------------------------------------

class TestAzureDocIntelEngine:
    def test_class_metadata(self, monkeypatch):
        from picarones.engines.azure_doc_intel import AzureDocIntelEngine
        monkeypatch.delenv("AZURE_DOC_INTEL_KEY", raising=False)
        monkeypatch.delenv("AZURE_DOC_INTEL_ENDPOINT", raising=False)
        eng = AzureDocIntelEngine()
        assert eng.name == "azure_doc_intel"
        assert eng.version()  # date string non vide
        assert eng.execution_mode == "io"

    def test_missing_key_raises(self, monkeypatch, fake_image):
        from picarones.engines.azure_doc_intel import AzureDocIntelEngine
        monkeypatch.delenv("AZURE_DOC_INTEL_KEY", raising=False)
        monkeypatch.setenv("AZURE_DOC_INTEL_ENDPOINT", "https://x.cognitiveservices.azure.com")
        eng = AzureDocIntelEngine()
        with pytest.raises(RuntimeError, match="AZURE_DOC_INTEL_KEY"):
            eng._run_ocr(fake_image)

    def test_missing_endpoint_raises(self, monkeypatch, fake_image):
        from picarones.engines.azure_doc_intel import AzureDocIntelEngine
        monkeypatch.setenv("AZURE_DOC_INTEL_KEY", "k")
        monkeypatch.delenv("AZURE_DOC_INTEL_ENDPOINT", raising=False)
        eng = AzureDocIntelEngine()
        with pytest.raises(RuntimeError, match="AZURE_DOC_INTEL_ENDPOINT"):
            eng._run_ocr(fake_image)

    def test_extract_text_pure_function(self):
        # Méthode statique — testable sans réseau ni mocks.
        from picarones.engines.azure_doc_intel import AzureDocIntelEngine
        result = {
            "analyzeResult": {
                "pages": [
                    {"lines": [
                        {"content": "Première ligne"},
                        {"content": "Deuxième ligne"},
                        {"content": ""},  # ignoré
                    ]},
                    {"lines": [{"content": "Page 2 — texte"}]},
                ],
            },
        }
        text = AzureDocIntelEngine._extract_text_from_result(result)
        assert "Première ligne" in text
        assert "Deuxième ligne" in text
        assert "Page 2 — texte" in text

    def test_extract_text_handles_empty_result(self):
        from picarones.engines.azure_doc_intel import AzureDocIntelEngine
        assert AzureDocIntelEngine._extract_text_from_result({}) == ""
        assert AzureDocIntelEngine._extract_text_from_result(
            {"analyzeResult": {"pages": []}}
        ) == ""


# ---------------------------------------------------------------------------
# 4. Cohérence inter-moteurs cloud — Sprint 24/31
# ---------------------------------------------------------------------------

class TestCloudEngineExecutionMode:
    """Sprint 24 documente que les moteurs cloud sont en mode IO. Le test
    vérifie cette invariance — si un futur sprint passe l'un d'eux en
    'cpu', le runner ne le mettrait plus dans le ThreadPool, ce qui
    serait une régression silencieuse de performance."""

    def test_all_cloud_engines_are_io_bound(self, monkeypatch):
        # Nettoyer les env vars pour ne pas tenter d'init clients cloud.
        for v in ("MISTRAL_API_KEY", "GOOGLE_API_KEY",
                  "GOOGLE_APPLICATION_CREDENTIALS",
                  "AZURE_DOC_INTEL_KEY", "AZURE_DOC_INTEL_ENDPOINT"):
            monkeypatch.delenv(v, raising=False)

        from picarones.engines.azure_doc_intel import AzureDocIntelEngine
        from picarones.engines.google_vision import GoogleVisionEngine
        from picarones.engines.mistral_ocr import MistralOCREngine

        for cls in (MistralOCREngine, GoogleVisionEngine, AzureDocIntelEngine):
            eng = cls()
            assert eng.execution_mode == "io", (
                f"{cls.__name__} doit rester IO-bound (utilisé en ThreadPool)"
            )
