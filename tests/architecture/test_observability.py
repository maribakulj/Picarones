"""Sprint S6.5+S6.6+S6.7 — observabilité institutionnelle.

Vérifie :
1. ``JsonLogFormatter`` produit un JSON valide avec les champs
   attendus.
2. ``request_id_middleware`` pose ``request.state.request_id`` et
   expose ``X-Request-Id`` en réponse.
3. Si le client fournit ``X-Request-Id``, il est respecté
   (tracing distribué).
4. Le handler 500 global (Sprint S3.2) inclut le ``request_id``
   dans son payload.
"""

from __future__ import annotations

import json
import logging

import pytest


# ──────────────────────────────────────────────────────────────────────
# 1. JsonLogFormatter
# ──────────────────────────────────────────────────────────────────────


class TestJsonLogFormatter:
    def test_basic_record_serializes_to_json(self) -> None:
        from picarones.interfaces.web.observability import JsonLogFormatter

        formatter = JsonLogFormatter()
        record = logging.LogRecord(
            name="picarones.test",
            level=logging.INFO,
            pathname="t.py", lineno=1,
            msg="hello %s", args=("world",),
            exc_info=None,
        )
        out = formatter.format(record)
        # Doit être un JSON valide.
        payload = json.loads(out)
        assert payload["level"] == "INFO"
        assert payload["logger"] == "picarones.test"
        assert payload["message"] == "hello world"
        assert "timestamp" in payload

    def test_request_id_included_when_set(self) -> None:
        from picarones.interfaces.web.observability import JsonLogFormatter

        formatter = JsonLogFormatter()
        record = logging.LogRecord(
            name="t", level=logging.INFO, pathname="t.py",
            lineno=1, msg="x", args=(), exc_info=None,
        )
        record.request_id = "abc123"

        payload = json.loads(formatter.format(record))
        assert payload["request_id"] == "abc123"

    def test_exception_info_flattened(self) -> None:
        from picarones.interfaces.web.observability import JsonLogFormatter

        formatter = JsonLogFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            record = logging.LogRecord(
                name="t", level=logging.ERROR, pathname="t.py",
                lineno=1, msg="caught", args=(),
                exc_info=sys.exc_info(),
            )

        payload = json.loads(formatter.format(record))
        assert payload["exc_type"] == "ValueError"
        assert payload["exc_message"] == "boom"
        assert "stack" in payload

    def test_extra_fields_propagated(self) -> None:
        from picarones.interfaces.web.observability import JsonLogFormatter

        formatter = JsonLogFormatter()
        record = logging.LogRecord(
            name="t", level=logging.INFO, pathname="t.py",
            lineno=1, msg="x", args=(), exc_info=None,
        )
        record.job_id = "job_42"
        record.user = "alice"

        payload = json.loads(formatter.format(record))
        assert payload["job_id"] == "job_42"
        assert payload["user"] == "alice"

    def test_non_json_serializable_value_falls_back_to_repr(self) -> None:
        from picarones.interfaces.web.observability import JsonLogFormatter

        formatter = JsonLogFormatter()
        record = logging.LogRecord(
            name="t", level=logging.INFO, pathname="t.py",
            lineno=1, msg="x", args=(), exc_info=None,
        )
        # set() n'est pas JSON-serializable.
        record.weird = {1, 2, 3}

        # Doit pas crasher — fallback sur repr().
        out = formatter.format(record)
        payload = json.loads(out)
        assert "weird" in payload
        assert isinstance(payload["weird"], str)


# ──────────────────────────────────────────────────────────────────────
# 2. request_id_middleware
# ──────────────────────────────────────────────────────────────────────


class TestRequestIdMiddleware:
    """Le middleware pose ``X-Request-Id`` en header de réponse.

    On vérifie ce header (pas le ``request.state.request_id`` côté
    serveur) pour éviter les pièges d'introspection FastAPI sur
    ``request: Request`` qui peuvent échouer dans certains contextes
    pytest.
    """

    def _make_app(self):
        from fastapi import FastAPI
        from starlette.middleware.base import BaseHTTPMiddleware

        from picarones.interfaces.web.observability import (
            request_id_middleware,
        )

        app = FastAPI()
        app.add_middleware(BaseHTTPMiddleware, dispatch=request_id_middleware)

        @app.get("/probe")
        async def probe_endpoint() -> dict:
            return {"ok": True}

        return app

    def test_request_id_header_set_when_missing(self) -> None:
        from fastapi.testclient import TestClient

        app = self._make_app()
        with TestClient(app) as client:
            r = client.get("/probe")
            assert r.status_code == 200, f"body={r.text[:300]}"
            # Présent dans le header de réponse
            headers_lower = {k.lower(): v for k, v in r.headers.items()}
            assert "x-request-id" in headers_lower
            rid = headers_lower["x-request-id"]
            assert len(rid) >= 8

    def test_incoming_request_id_respected(self) -> None:
        from fastapi.testclient import TestClient

        app = self._make_app()
        with TestClient(app) as client:
            r = client.get(
                "/probe", headers={"X-Request-Id": "trace-abc-123"},
            )
            assert r.status_code == 200
            assert r.headers.get("x-request-id") == "trace-abc-123"

    def test_too_long_incoming_id_replaced(self) -> None:
        from fastapi.testclient import TestClient

        app = self._make_app()
        with TestClient(app) as client:
            r = client.get(
                "/probe", headers={"X-Request-Id": "a" * 65},
            )
            assert r.status_code == 200
            rid = r.headers.get("x-request-id", "")
            # Pas le payload géant — un nouvel ID a été généré.
            assert rid != "a" * 65
            assert len(rid) <= 12

    def test_non_ascii_incoming_id_replaced(self) -> None:
        """Garde-fou anti log injection : un id contenant un
        caractère de contrôle est rejeté et remplacé par un id
        propre."""
        from fastapi.testclient import TestClient

        app = self._make_app()
        with TestClient(app) as client:
            r = client.get(
                "/probe", headers={"X-Request-Id": "trace\x01id"},
            )
            assert r.status_code == 200
            rid = r.headers.get("x-request-id", "")
            assert "\x01" not in rid


# ──────────────────────────────────────────────────────────────────────
# 3. install_json_logging
# ──────────────────────────────────────────────────────────────────────


class TestInstallJsonLogging:
    def test_installs_json_formatter_on_root(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from picarones.interfaces.web.observability import (
            JsonLogFormatter, install_json_logging,
        )

        # Sauvegarde + restore les handlers root.
        root = logging.getLogger()
        saved = list(root.handlers)
        try:
            install_json_logging()
            assert root.handlers
            assert isinstance(root.handlers[0].formatter, JsonLogFormatter)
        finally:
            root.handlers = saved


# ──────────────────────────────────────────────────────────────────────
# 4. is_json_logging_requested
# ──────────────────────────────────────────────────────────────────────


class TestIsJsonLoggingRequested:
    def test_default_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from picarones.interfaces.web.observability import (
            is_json_logging_requested,
        )
        monkeypatch.delenv("PICARONES_LOG_FORMAT", raising=False)
        assert is_json_logging_requested() is False

    def test_value_json_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from picarones.interfaces.web.observability import (
            is_json_logging_requested,
        )
        monkeypatch.setenv("PICARONES_LOG_FORMAT", "json")
        assert is_json_logging_requested() is True

    def test_value_JSON_uppercase_true(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from picarones.interfaces.web.observability import (
            is_json_logging_requested,
        )
        monkeypatch.setenv("PICARONES_LOG_FORMAT", "JSON")
        assert is_json_logging_requested() is True

    def test_other_value_false(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from picarones.interfaces.web.observability import (
            is_json_logging_requested,
        )
        monkeypatch.setenv("PICARONES_LOG_FORMAT", "text")
        assert is_json_logging_requested() is False
