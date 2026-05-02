"""Tests Sprint A5 — robustesse des adapters cloud face aux erreurs HTTP.

Item m-10 de l'audit institutional-readiness-2026-05.

**Contrat testé** : si l'API cloud renvoie une erreur HTTP (401, 429,
500, 503) ou un body mal formé, l'adapter doit produire un
``EngineResult`` dont :

1. ``text == ""`` (pas de transcription fictive),
2. ``error`` est non vide et **contient le code HTTP** (pour que
   l'utilisateur sache si c'est un rate limit, une clé invalide, une
   indispo, etc.),
3. ``engine_name`` est correctement renseigné.

Ce contrat est crucial : sans ces tests, une régression où un adapter
retournerait silencieusement ``text=""`` sans ``error`` ferait croire
à un crash du moteur OCR alors que c'est l'API qui était indisponible
— pire scénario possible pour un benchmark institutionnel.

NB : le pattern ``BaseOCREngine.run()`` capture les exceptions et les
stocke dans ``EngineResult.error`` (décision architecturale Sprint 14
pour que le runner continue avec les autres docs). Donc ce test
vérifie ``result.error``, pas ``pytest.raises``.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_image_path(tmp_path: Path) -> Path:
    """Crée un PNG minimal pour satisfaire les checks de présence."""
    p = tmp_path / "page.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n")
    return p


def _http_error(code: int, body: str = '{"error": "test"}') -> HTTPError:
    return HTTPError(
        url="https://api.example/test",
        code=code,
        msg="Test",
        hdrs=None,  # type: ignore[arg-type]
        fp=io.BytesIO(body.encode("utf-8")),
    )


def _assert_error_propagated(result, expected_code: int) -> None:
    """Vérifie le contrat de propagation d'erreur HTTP."""
    assert result is not None, "EngineResult ne doit jamais être None"
    assert result.text == "", (
        f"Sur erreur HTTP, l'adapter doit retourner text='', pas "
        f"une chaîne fictive. Obtenu : {result.text!r}"
    )
    assert result.error, (
        "Sur erreur HTTP, EngineResult.error doit être renseigné. "
        "Avaler silencieusement une erreur API est le pire scénario."
    )
    assert str(expected_code) in result.error, (
        f"EngineResult.error doit contenir le code HTTP {expected_code} ; "
        f"obtenu : {result.error!r}"
    )


# ---------------------------------------------------------------------------
# Google Vision
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", [401, 403, 429, 500, 503])
def test_google_vision_propagates_http_error(
    fake_image_path: Path, code: int, monkeypatch
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    from picarones.engines.google_vision import GoogleVisionEngine

    engine = GoogleVisionEngine()

    with patch("picarones.engines.google_vision.urllib.request.urlopen") as mock_open:
        mock_open.side_effect = _http_error(code)
        result = engine.run(fake_image_path)

    _assert_error_propagated(result, code)
    assert result.engine_name == "google_vision"


def test_google_vision_propagates_network_failure(
    fake_image_path: Path, monkeypatch
) -> None:
    """``URLError`` (DNS, timeout TCP) doit aussi remplir ``result.error``."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    from picarones.engines.google_vision import GoogleVisionEngine

    engine = GoogleVisionEngine()

    with patch("picarones.engines.google_vision.urllib.request.urlopen") as mock_open:
        mock_open.side_effect = URLError("Name or service not known")
        result = engine.run(fake_image_path)

    assert result.text == ""
    assert result.error, "URLError doit être propagée via result.error"


# ---------------------------------------------------------------------------
# Azure Document Intelligence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", [401, 403, 429, 500, 503])
def test_azure_doc_intel_propagates_http_error(
    fake_image_path: Path, code: int, monkeypatch
) -> None:
    monkeypatch.setenv(
        "AZURE_DOC_INTEL_ENDPOINT", "https://test.cognitiveservices.azure.com"
    )
    monkeypatch.setenv("AZURE_DOC_INTEL_KEY", "fake")
    from picarones.engines.azure_doc_intel import AzureDocIntelEngine

    engine = AzureDocIntelEngine()

    with patch("picarones.engines.azure_doc_intel.urllib.request.urlopen") as mock_open:
        mock_open.side_effect = _http_error(code)
        result = engine.run(fake_image_path)

    _assert_error_propagated(result, code)


def test_azure_doc_intel_handles_missing_operation_location(
    fake_image_path: Path, monkeypatch
) -> None:
    """Réponse 202 sans en-tête ``Operation-Location`` → l'engine doit
    remplir ``result.error`` plutôt que de boucler indéfiniment ou
    de retourner du vide silencieux."""
    monkeypatch.setenv(
        "AZURE_DOC_INTEL_ENDPOINT", "https://test.cognitiveservices.azure.com"
    )
    monkeypatch.setenv("AZURE_DOC_INTEL_KEY", "fake")
    from picarones.engines.azure_doc_intel import AzureDocIntelEngine

    engine = AzureDocIntelEngine()

    fake_response = MagicMock()
    fake_response.status = 202
    fake_response.headers = {}  # pas d'Operation-Location
    fake_response.__enter__ = lambda self: self
    fake_response.__exit__ = lambda self, *a: False
    fake_response.read = lambda: b""

    with patch(
        "picarones.engines.azure_doc_intel.urllib.request.urlopen",
        return_value=fake_response,
    ):
        result = engine.run(fake_image_path)

    assert result.text == ""
    assert result.error and "Operation-Location" in result.error


# ---------------------------------------------------------------------------
# Mistral OCR
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", [401, 429, 500, 503])
def test_mistral_ocr_propagates_http_error(
    fake_image_path: Path, code: int, monkeypatch
) -> None:
    """Le chemin natif Mistral OCR fait ``import urllib.request`` à
    l'intérieur de ``_run_ocr_native_api`` (pas au top-level), donc
    on patch ``urllib.request.urlopen`` global."""
    monkeypatch.setenv("MISTRAL_API_KEY", "fake")
    from picarones.engines.mistral_ocr import MistralOCREngine

    engine = MistralOCREngine()

    with patch("urllib.request.urlopen") as mock_open:
        mock_open.side_effect = _http_error(code)
        result = engine.run(fake_image_path)

    # Mistral peut tomber en fallback Vision API ; on accepte donc soit
    # propagation propre du code HTTP, soit propagation d'un message
    # générique mais non vide. Le contrat minimal : pas de silence.
    assert result.text == ""
    assert result.error, (
        f"Mistral OCR a avalé l'erreur HTTP {code} silencieusement. "
        "Mauvais signal pour un benchmark institutionnel."
    )


# ---------------------------------------------------------------------------
# Garde-fou transverse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "engine_cls_path,env_vars,patch_target",
    [
        (
            "picarones.engines.google_vision.GoogleVisionEngine",
            {"GOOGLE_API_KEY": "x"},
            "picarones.engines.google_vision.urllib.request.urlopen",
        ),
        (
            "picarones.engines.azure_doc_intel.AzureDocIntelEngine",
            {
                "AZURE_DOC_INTEL_ENDPOINT": "https://test.cognitiveservices.azure.com",
                "AZURE_DOC_INTEL_KEY": "x",
            },
            "picarones.engines.azure_doc_intel.urllib.request.urlopen",
        ),
        (
            "picarones.engines.mistral_ocr.MistralOCREngine",
            {"MISTRAL_API_KEY": "x"},
            "urllib.request.urlopen",
        ),
    ],
)
def test_no_silent_empty_on_5xx(
    fake_image_path: Path,
    engine_cls_path: str,
    env_vars: dict,
    patch_target: str,
    monkeypatch,
) -> None:
    """Garantit transverse : aucun adapter cloud ne doit retourner un
    ``EngineResult`` avec ``text=""`` et ``error=None`` sur 503.

    C'est le pire scénario : un benchmark qui rapporte CER=100 % et
    fait croire à un crash du moteur OCR alors que c'est l'API qui
    était indisponible (impact direct sur les conclusions éditoriales)."""
    for k, v in env_vars.items():
        monkeypatch.setenv(k, v)

    module_path, cls_name = engine_cls_path.rsplit(".", 1)
    import importlib

    mod = importlib.import_module(module_path)
    engine_cls = getattr(mod, cls_name)
    engine = engine_cls()

    with patch(patch_target) as mock_open:
        mock_open.side_effect = _http_error(503)
        result = engine.run(fake_image_path)

    assert result.text == "", (
        f"{cls_name} a inventé du texte sur erreur 503 : {result.text!r}"
    )
    assert result.error, (
        f"{cls_name} a avalé l'erreur 503 silencieusement (text='', "
        f"error=None). Régression critique pour un benchmark BnF."
    )
