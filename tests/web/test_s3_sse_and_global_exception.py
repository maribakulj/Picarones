"""Sprint S3 — bugs latents : SSE NoneType + handler exception global.

Tests
-----

S3.1 — SSE pour un job inexistant : retourne 404 propre, pas
d'AttributeError sur ``None.status``.

S3.2 — handler exception global FastAPI : route qui throw → 500
JSON propre avec ``request_id``, sans stack trace exposée.
"""

from __future__ import annotations

import pytest


# ──────────────────────────────────────────────────────────────────────
# S3.1 — SSE sur job inexistant ne crash pas
# ──────────────────────────────────────────────────────────────────────


class TestSSEPurgedJob:
    """Une connexion SSE sur un job qui n'existe pas (jamais créé,
    purgé, ou dans un autre worker) doit retourner 404 propre."""

    def test_sse_unknown_job_returns_404(self) -> None:
        from fastapi.testclient import TestClient
        from picarones.interfaces.web.app import app

        with TestClient(app) as client:
            r = client.get("/api/benchmark/job_does_not_exist/stream")
            # Soit 404 (le bon comportement, déjà implémenté ligne 263),
            # soit 200 SSE (cas dégradé acceptable mais pas idéal).
            # Refus net : NE PAS crasher en 500.
            assert r.status_code != 500, (
                f"SSE sur job inexistant a crashé en 500 — bug NoneType "
                f"reproduit.  Reçu : {r.status_code}, body : "
                f"{r.text[:200]}"
            )
            assert r.status_code in (404,), (
                f"SSE sur job inconnu doit retourner 404, reçu "
                f"{r.status_code}."
            )


# ──────────────────────────────────────────────────────────────────────
# S3.2 — Handler exception global
# ──────────────────────────────────────────────────────────────────────


class TestGlobalExceptionHandler:
    """Une route qui lève une exception non capturée doit retourner
    un 500 JSON propre, pas la stack trace au client."""

    def _make_app_with_throwing_route(self):
        """Construit une mini app FastAPI qui inclut le handler
        global de Picarones + une route qui throw."""
        from fastapi import FastAPI

        from picarones.interfaces.web.app import (
            register_global_exception_handler,
        )

        app = FastAPI()
        register_global_exception_handler(app)

        @app.get("/api/throw")
        async def throw_route() -> dict:
            raise RuntimeError("boom — détail interne sensible")

        return app

    def test_unhandled_exception_returns_500_json(self) -> None:
        from fastapi.testclient import TestClient

        app = self._make_app_with_throwing_route()
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.get("/api/throw")
            assert r.status_code == 500

            # Body doit être JSON, pas HTML / stack trace
            body = r.json()
            assert "error" in body or "detail" in body

    def test_unhandled_exception_does_not_leak_stack_trace(self) -> None:
        from fastapi.testclient import TestClient

        app = self._make_app_with_throwing_route()
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.get("/api/throw")
            text = r.text

            # Le détail interne ne doit PAS fuiter dans la réponse.
            forbidden_strings = (
                "boom — détail interne sensible",
                "Traceback",
                "RuntimeError",
                "throw_route",
                'File "',
                "line ",
            )
            for forbidden in forbidden_strings:
                assert forbidden not in text, (
                    f"La réponse 500 contient ``{forbidden!r}`` — "
                    f"fuite d'information interne.\nBody complet :\n"
                    f"{text[:500]}"
                )

    def test_unhandled_exception_logs_at_error_level(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Le détail technique doit aller dans les logs côté serveur,
        pas au client.  Une équipe ops doit pouvoir corréler la
        500 à un log d'erreur."""
        import logging

        from fastapi.testclient import TestClient

        app = self._make_app_with_throwing_route()
        with caplog.at_level(logging.ERROR):
            with TestClient(app, raise_server_exceptions=False) as client:
                client.get("/api/throw")

        # Au moins un log ERROR a été émis avec le contexte.
        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert error_records, (
            "Aucun log ERROR émis pour l'exception non capturée — "
            "l'équipe ops ne pourra pas diagnostiquer."
        )
