"""Sprint S8.7 — couverture des branches HTTP error des adapters
OCR cloud (Azure Document Intelligence + Google Vision REST).

Cible (avant) :
- ``azure_doc_intel.py`` 91% — lignes 305-322 (HTTPError sur POST
  initial), 342-343 (Exception sur polling).
- ``google_vision.py`` 88% — lignes 272-288 (HTTPError sur call),
  295 (responses vides).

Pourquoi tester ces branches
----------------------------
Les adapters cloud parlent à des APIs distantes — un 400 (key
invalide), un 429 (quota), un 500 (panne serveur), une exception
réseau pendant le polling : tous doivent être transformés en
``OCRAdapterError`` lisible (avec code HTTP + body) plutôt que de
remonter une ``HTTPError`` brute qui confond le caller du
``CorpusRunner``.

Le mock cible ``urllib.request.urlopen`` (la lib stdlib utilisée
en mode REST quand le SDK officiel n'est pas installé).  Pas de
mock du SDK — on teste explicitement le chemin REST direct.
"""

from __future__ import annotations

import io
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

from picarones.adapters.ocr.base import OCRAdapterError


def _png_bytes() -> bytes:
    """1×1 PNG transparent valide."""
    import base64
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "AAIAAAoAAv/lpgAAAABJRU5ErkJggg=="
    )


def _make_image(tmp_path) -> Path:
    img = tmp_path / "test.png"
    img.write_bytes(_png_bytes())
    return img


# ──────────────────────────────────────────────────────────────────────
# Azure Document Intelligence — HTTP errors via REST
# ──────────────────────────────────────────────────────────────────────


class TestAzureDocIntelHTTPErrors:
    def _adapter(self):
        from picarones.adapters.ocr.azure_doc_intel import (
            AzureDocIntelAdapter,
        )

        return AzureDocIntelAdapter(
            endpoint="https://fake.azure.com",
            api_key="fake-key",
            timeout_seconds=1.0,
            max_polling_attempts=2,
            polling_interval_base=0.01,
        )

    def test_http_error_with_readable_body_raises_with_code(
        self, tmp_path,
    ) -> None:
        """Un HTTP 401 (Unauthorized) sur le POST initial doit être
        transformé en ``OCRAdapterError`` qui inclut le code et le
        body (utile pour debug : "key invalide" vs "endpoint inconnu").
        Couvre lignes 305-320."""
        adapter = self._adapter()
        image = _make_image(tmp_path)

        body = b'{"error": {"code": "Unauthorized"}}'
        http_err = urllib.error.HTTPError(
            url="https://fake.azure.com/x",
            code=401, msg="Unauthorized",
            hdrs=None, fp=io.BytesIO(body),
        )

        with patch(
            "urllib.request.urlopen", side_effect=http_err,
        ):
            with pytest.raises(OCRAdapterError) as exc_info:
                adapter._call_via_rest(
                    image, "https://fake.azure.com", "fake-key",
                )
        msg = str(exc_info.value)
        assert "401" in msg
        assert "Unauthorized" in msg or "Azure" in msg

    def test_http_error_with_unreadable_body_still_raises(
        self, tmp_path,
    ) -> None:
        """Si lire le body de l'HTTPError échoue (fp cassé,
        encoding inattendu), on lève quand même avec le code seul.
        Couvre la branche ``except Exception as read_exc`` lignes
        309-316.  Code 404 non-retryable pour éviter le backoff
        retry (test rapide)."""
        adapter = self._adapter()
        image = _make_image(tmp_path)

        class BrokenFp:
            def read(self):
                raise UnicodeDecodeError("utf-8", b"", 0, 1, "broken")

            def close(self):  # __del__ de TemporaryFileCloser appelle close()
                pass

        http_err = urllib.error.HTTPError(
            url="https://fake.azure.com/x",
            code=404, msg="Not Found",
            hdrs=None, fp=BrokenFp(),
        )

        with patch(
            "urllib.request.urlopen", side_effect=http_err,
        ):
            with pytest.raises(OCRAdapterError, match="404"):
                adapter._call_via_rest(
                    image, "https://fake.azure.com", "fake-key",
                )

    def test_generic_exception_wrapped_with_type_name(
        self, tmp_path,
    ) -> None:
        """Une exception non-HTTP non-retryable doit être
        transformée en ``OCRAdapterError`` avec le type d'origine.
        Couvre lignes 321-325.  ``ValueError`` non-retryable pour
        éviter le backoff."""
        adapter = self._adapter()
        image = _make_image(tmp_path)

        with patch(
            "urllib.request.urlopen",
            side_effect=ValueError("malformed URL"),
        ):
            with pytest.raises(OCRAdapterError) as exc_info:
                adapter._call_via_rest(
                    image, "https://fake.azure.com", "fake-key",
                )
        # Le wrapping doit nommer le type d'exception d'origine.
        assert "ValueError" in str(exc_info.value)


# ──────────────────────────────────────────────────────────────────────
# Google Vision — HTTP errors via REST
# ──────────────────────────────────────────────────────────────────────


class TestGoogleVisionHTTPErrors:
    def _adapter(self):
        from picarones.adapters.ocr.google_vision import (
            GoogleVisionAdapter,
        )

        return GoogleVisionAdapter(
            api_key="fake-key",
            timeout_seconds=1.0,
        )

    def test_http_error_with_body_raises_with_code(
        self, tmp_path,
    ) -> None:
        """Couvre lignes 272-286."""
        adapter = self._adapter()
        image = _make_image(tmp_path)

        body = b'{"error": {"message": "API key invalid"}}'
        http_err = urllib.error.HTTPError(
            url="https://vision.googleapis.com/x",
            code=403, msg="Forbidden",
            hdrs=None, fp=io.BytesIO(body),
        )

        with patch(
            "urllib.request.urlopen", side_effect=http_err,
        ):
            with pytest.raises(OCRAdapterError) as exc_info:
                adapter._call_via_rest(image, "fake-key")
        msg = str(exc_info.value)
        assert "403" in msg

    def test_http_error_with_unreadable_body(self, tmp_path) -> None:
        """Couvre la branche ``except Exception as read_exc``
        lignes 276-283.  Code 404 non-retryable pour éviter le
        backoff (test rapide)."""
        adapter = self._adapter()
        image = _make_image(tmp_path)

        class BrokenFp:
            def read(self):
                raise OSError("disk read error")

            def close(self):
                pass

        http_err = urllib.error.HTTPError(
            url="https://vision.googleapis.com/x",
            code=404, msg="Not Found",
            hdrs=None, fp=BrokenFp(),
        )

        with patch(
            "urllib.request.urlopen", side_effect=http_err,
        ):
            with pytest.raises(OCRAdapterError, match="404"):
                adapter._call_via_rest(image, "fake-key")

    def test_generic_exception_wrapped(self, tmp_path) -> None:
        """Couvre lignes 287-291.  ``ValueError`` non-retryable."""
        adapter = self._adapter()
        image = _make_image(tmp_path)

        with patch(
            "urllib.request.urlopen",
            side_effect=ValueError("malformed payload"),
        ):
            with pytest.raises(OCRAdapterError) as exc_info:
                adapter._call_via_rest(image, "fake-key")
        assert "ValueError" in str(exc_info.value)
