"""Sprint S8.7 — derniers wins coverage (lignes 1-2 manquantes
dans plusieurs petits fichiers de la patch).

Cibles :
- ``adapters/corpus/_http.validate_http_url`` : ``hostname`` vide.
- ``adapters/corpus/_http.download_url`` : header custom via
  ``extra_headers``.
- ``adapters/corpus/gallica`` : politesse ``time.sleep(delay)``
  quand ``delay > 0``.
- ``app/services/dependencies.capture_system_binaries_lock`` :
  vrai retour avec binaire dispo (ligne 110) + fallback empty
  output (lignes 81-84).
- ``adapters/llm/base.execute`` : retry exponentiel sur erreur
  retryable (lignes 353-358) — testé via une fake fonction
  ``complete`` qui lève deux fois puis réussit.
"""

from __future__ import annotations

import time
from unittest.mock import patch, MagicMock

import pytest


# ──────────────────────────────────────────────────────────────────────
# _http.py — validate_http_url + download_url avec extra_headers
# ──────────────────────────────────────────────────────────────────────


class TestValidateHTTPUrl:
    def test_empty_hostname_rejected(self) -> None:
        """Couvre ligne 120 — ``http:///path`` (sans host) → rejet."""
        from picarones.adapters.corpus._http import validate_http_url

        with pytest.raises(ValueError, match="hostname"):
            validate_http_url("http:///some/path")

    def test_unsupported_scheme_rejected(self) -> None:
        from picarones.adapters.corpus._http import validate_http_url

        with pytest.raises(ValueError, match="Schéma"):
            validate_http_url("ftp://example.com/file")

    def test_blocked_loopback_rejected(self) -> None:
        from picarones.adapters.corpus._http import validate_http_url

        with pytest.raises(ValueError, match="refusé|loopback|interne"):
            validate_http_url("http://127.0.0.1/admin")

    def test_legitimate_url_accepted(self) -> None:
        """Contrôle positif : un URL public valide ne lève pas."""
        from picarones.adapters.corpus._http import validate_http_url

        validate_http_url("https://gallica.bnf.fr/ark:/12148/foo")
        # no raise


class TestDownloadURLWithExtraHeaders:
    def test_extra_headers_merged_into_request(self) -> None:
        """Couvre ligne 165 — ``extra_headers`` doit être mergé
        dans le dict de headers de la requête."""
        from picarones.adapters.corpus import _http as http_mod

        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["headers"] = dict(req.headers)
            mock_resp = MagicMock()
            mock_resp.read.return_value = b"ok"
            mock_resp.__enter__ = lambda self: mock_resp
            mock_resp.__exit__ = lambda *a: None
            return mock_resp

        with patch.object(http_mod.urllib.request, "urlopen", fake_urlopen):
            http_mod.download_url(
                "https://gallica.bnf.fr/foo",
                user_agent="picarones-test/1.0",
                extra_headers={"X-Custom": "value-42"},
                retries=1,
            )
        assert captured["headers"].get("X-custom") == "value-42", (
            f"extra_headers pas mergé : {captured['headers']!r}"
        )


# ──────────────────────────────────────────────────────────────────────
# gallica.py — politesse delay
# ──────────────────────────────────────────────────────────────────────


class TestGallicaDelay:
    def test_delay_triggers_sleep_when_positive(self, monkeypatch) -> None:
        """Couvre ligne 161 — ``time.sleep(self.delay)`` quand
        ``delay > 0`` (politesse anti rate-limit Gallica)."""
        from picarones.adapters.corpus.gallica import GallicaClient

        sleep_calls: list[float] = []

        def fake_sleep(duration):
            sleep_calls.append(duration)

        monkeypatch.setattr(time, "sleep", fake_sleep)

        client = GallicaClient(delay_between_requests=0.5)

        # ``_fetch_url`` importe ``download_url`` au call-time depuis
        # ``_http`` — c'est cette référence qu'on patche.
        def fake_download(url, **kwargs):
            return b'<srw:searchRetrieveResponse xmlns:srw="x"></srw:searchRetrieveResponse>'

        monkeypatch.setattr(
            "picarones.adapters.corpus._http.download_url",
            fake_download,
        )

        client.search(title="hugo", max_results=1)
        assert 0.5 in sleep_calls, (
            f"sleep(0.5) attendu, appels : {sleep_calls}"
        )


# ──────────────────────────────────────────────────────────────────────
# dependencies.py — capture_system_binaries_lock
# ──────────────────────────────────────────────────────────────────────


class TestSystemBinariesLock:
    def test_returns_dict_with_available_binary(self) -> None:
        """Couvre ligne 110 — quand un binaire est dans PATH et
        retourne une version, il est ajouté au lock dict."""
        from picarones.app.services.dependencies import (
            capture_system_binaries_lock,
        )

        lock = capture_system_binaries_lock()
        # Python est garanti d'être dispo (on est en train de
        # l'exécuter).  ``tesseract`` peut être absent localement.
        assert isinstance(lock, dict)
        # Tous les binaires détectés ont une version non vide.
        for binary, version in lock.items():
            assert version, f"{binary} a une version vide dans le lock"

    def test_safe_capture_returns_none_when_binary_absent(self) -> None:
        from picarones.app.services.dependencies import (
            _safe_capture_binary_version,
        )

        result = _safe_capture_binary_version(
            "definitely-not-a-real-binary-xyz-12345",
        )
        assert result is None

    def test_safe_capture_returns_none_on_empty_output(
        self, monkeypatch, tmp_path,
    ) -> None:
        """Couvre lignes 81-84 — si le binaire répond avec une
        chaîne vide, on retourne ``None`` (pas une chaîne vide)."""
        from picarones.app.services import dependencies as deps_mod

        # Mock shutil.which pour faire croire que le binaire existe.
        monkeypatch.setattr(
            "shutil.which", lambda b: "/usr/bin/fake-empty-output"
            if b == "fake-empty-output" else None,
        )

        # Mock subprocess.run pour retourner stdout vide.
        class FakeResult:
            stdout = "   "  # whitespace only
            stderr = ""

        monkeypatch.setattr(
            "subprocess.run", lambda *args, **kwargs: FakeResult(),
        )

        result = deps_mod._safe_capture_binary_version("fake-empty-output")
        assert result is None

    def test_safe_capture_subprocess_error_returns_none(
        self, monkeypatch, caplog,
    ) -> None:
        """Couvre la branche ``except OSError`` lignes 76-80."""
        import subprocess

        from picarones.app.services import dependencies as deps_mod

        monkeypatch.setattr(
            "shutil.which", lambda b: "/usr/bin/fake-crashing",
        )

        def raising_run(*args, **kwargs):
            raise subprocess.SubprocessError("simulated crash")

        monkeypatch.setattr("subprocess.run", raising_run)

        with caplog.at_level("DEBUG"):
            result = deps_mod._safe_capture_binary_version(
                "fake-crashing",
            )
        assert result is None
        assert any("échouée" in rec.message for rec in caplog.records)


# ──────────────────────────────────────────────────────────────────────
# llm/base.py — retry exponentiel sur erreur retryable
# ──────────────────────────────────────────────────────────────────────


class TestBaseLLMAdapterRetry:
    """``BaseLLMAdapter.complete()`` (lignes 329-371) gère le retry
    interne : sur erreur retryable (TimeoutError, 5xx, rate-limit),
    backoff exponentiel ; sur erreur non-retryable, sortie immédiate.
    Le helper ``_call(prompt, image_b64)`` est le point d'extension
    SDK-specific qu'on mocke ici."""

    def test_retry_on_retryable_error_then_success(
        self, monkeypatch, caplog,
    ) -> None:
        """Couvre lignes 352-358 — ``_call`` lève ``TimeoutError``
        (retryable), backoff exponentiel, puis réussit."""
        from picarones.adapters.llm.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter(model="gpt-4o")
        # Désactive le sleep réel sinon le test dure ~4s (backoff 2^1
        # + 2^2).
        monkeypatch.setattr(time, "sleep", lambda d: None)

        call_count = {"n": 0}

        def fake_call(prompt, image_b64=None):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise TimeoutError("simulated timeout")
            return "recovered"

        adapter._call = fake_call
        # 3 retries = 4 tentatives totales possibles.
        adapter.config["max_retries"] = 3
        adapter.config["retry_backoff"] = 2.0

        with caplog.at_level("WARNING"):
            result = adapter.complete("dummy prompt")
        assert result.text == "recovered"
        # 3 tentatives : 1 initiale + 2 retries.
        assert call_count["n"] == 3
        # 2 warnings émis (1 par retry).
        retry_warnings = [
            rec for rec in caplog.records if "retryable" in rec.message
        ]
        assert len(retry_warnings) >= 2

    def test_non_retryable_error_breaks_immediately(
        self, monkeypatch,
    ) -> None:
        """Une exception non-retryable (``ValueError`` par ex.) sort
        de la boucle au 1er échec (ligne 360 — ``else: break``)."""
        from picarones.adapters.llm.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter(model="gpt-4o")
        monkeypatch.setattr(time, "sleep", lambda d: None)

        call_count = {"n": 0}

        def fake_call(prompt, image_b64=None):
            call_count["n"] += 1
            raise ValueError("non-retryable error")

        adapter._call = fake_call
        adapter.config["max_retries"] = 3

        result = adapter.complete("dummy")
        # ValueError n'est pas retryable → 1 seule tentative.
        assert call_count["n"] == 1
        # Le LLMResult retourné a ``error`` renseigné.
        assert result.error is not None
        assert "non-retryable" in result.error
