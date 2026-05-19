"""Sprint S1.7 — Tests CSRF token requis en mode institutionnel.

Le middleware ``picarones.interfaces.web.security.csrf_middleware``
implémente le pattern « double-submit cookie » + signature HMAC.

Vérifications
-------------
1. Mode public (``PICARONES_CSRF_REQUIRED`` non set) : bypass total
   (rétrocompat HuggingFace Space).
2. Mode institutionnel (``PICARONES_CSRF_REQUIRED=1``) :
   - GET sur endpoint exempt → pas de token requis.
   - POST/PUT/DELETE/PATCH sur endpoint protégé sans token → 403.
   - POST avec token cookie mais pas header → 403.
   - POST avec cookie + header divergents → 403.
   - POST avec token mal signé → 403.
   - POST avec token valide → 2xx.
3. ``generate_csrf_token`` produit des tokens uniques par appel.
4. ``verify_csrf_token`` accepte un token valide et rejette les
   variantes (vide, mal formé, signature invalide).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture
def csrf_required(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Active ``PICARONES_CSRF_REQUIRED=1`` et fournit un secret
    déterministe pour les tests."""
    monkeypatch.setenv("PICARONES_CSRF_REQUIRED", "1")
    monkeypatch.setenv(
        "PICARONES_CSRF_SECRET", "test-secret-deterministic-do-not-use-in-prod",
    )
    # Force le re-cache du secret runtime
    import picarones.interfaces.web.security as sec
    sec._csrf_secret_runtime = None  # type: ignore[attr-defined]
    yield


@pytest.fixture
def csrf_disabled(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("PICARONES_CSRF_REQUIRED", raising=False)
    yield


# ──────────────────────────────────────────────────────────────────────
# 1. Token generation + verification (unit)
# ──────────────────────────────────────────────────────────────────────


class TestCSRFTokenPrimitives:
    def test_generate_returns_signed_token(self, csrf_required: None) -> None:
        from picarones.interfaces.web.security import generate_csrf_token

        token = generate_csrf_token()
        assert "." in token, "Format attendu nonce_hex.signature_hex"
        nonce_hex, sig_hex = token.split(".", 1)
        # nonce = 16 bytes → 32 hex chars
        assert len(nonce_hex) == 32
        # HMAC-SHA256 = 32 bytes → 64 hex chars
        assert len(sig_hex) == 64

    def test_generate_returns_unique_token_per_call(
        self, csrf_required: None,
    ) -> None:
        from picarones.interfaces.web.security import generate_csrf_token

        tokens = {generate_csrf_token() for _ in range(50)}
        assert len(tokens) == 50, "Collision détectée — nonce non unique"

    def test_verify_accepts_valid_token(self, csrf_required: None) -> None:
        from picarones.interfaces.web.security import (
            generate_csrf_token,
            verify_csrf_token,
        )

        token = generate_csrf_token()
        assert verify_csrf_token(token) is True

    def test_verify_rejects_none(self, csrf_required: None) -> None:
        from picarones.interfaces.web.security import verify_csrf_token

        assert verify_csrf_token(None) is False

    def test_verify_rejects_empty(self, csrf_required: None) -> None:
        from picarones.interfaces.web.security import verify_csrf_token

        assert verify_csrf_token("") is False

    def test_verify_rejects_malformed(self, csrf_required: None) -> None:
        from picarones.interfaces.web.security import verify_csrf_token

        assert verify_csrf_token("notatoken") is False
        assert verify_csrf_token("abc.def") is False  # not hex
        assert verify_csrf_token("a" * 32 + "." + "z" * 64) is False  # bad hex

    def test_verify_rejects_tampered_signature(
        self, csrf_required: None,
    ) -> None:
        from picarones.interfaces.web.security import (
            generate_csrf_token,
            verify_csrf_token,
        )

        token = generate_csrf_token()
        nonce, sig = token.split(".", 1)
        # Flip un bit de la signature
        tampered_sig = ("0" if sig[0] != "0" else "1") + sig[1:]
        assert verify_csrf_token(f"{nonce}.{tampered_sig}") is False


# ──────────────────────────────────────────────────────────────────────
# 2. Middleware end-to-end via TestClient
# ──────────────────────────────────────────────────────────────────────


class TestCSRFMiddlewareEnforcement:
    """Tests d'intégration via FastAPI TestClient."""

    def _make_app(self):
        """Crée une mini app FastAPI avec le middleware CSRF + un
        POST factice pour le tester."""
        from fastapi import FastAPI
        from starlette.middleware.base import BaseHTTPMiddleware

        from picarones.interfaces.web.security import csrf_middleware

        app = FastAPI()
        app.add_middleware(BaseHTTPMiddleware, dispatch=csrf_middleware)

        @app.get("/api/csrf/token")
        async def get_token():
            from picarones.interfaces.web.security import generate_csrf_token
            return {"token": generate_csrf_token()}

        @app.post("/api/test")
        async def post_endpoint():
            return {"ok": True}

        @app.get("/api/test")
        async def get_endpoint():
            return {"ok": True}

        return app

    def test_get_does_not_require_token(
        self, csrf_required: None,
    ) -> None:
        from fastapi.testclient import TestClient

        app = self._make_app()
        with TestClient(app) as client:
            r = client.get("/api/test")
            assert r.status_code == 200

    def test_post_without_token_returns_403(
        self, csrf_required: None,
    ) -> None:
        from fastapi.testclient import TestClient

        app = self._make_app()
        with TestClient(app) as client:
            r = client.post("/api/test")
            assert r.status_code == 403, (
                f"POST sans CSRF token doit être refusé en mode "
                f"institutionnel.  Reçu : {r.status_code}, body : "
                f"{r.text[:200]}"
            )
            assert "CSRF" in r.text or "csrf" in r.text

    def test_post_with_cookie_only_returns_403(
        self, csrf_required: None,
    ) -> None:
        from fastapi.testclient import TestClient
        from picarones.interfaces.web.security import (
            CSRF_COOKIE,
            generate_csrf_token,
        )

        app = self._make_app()
        with TestClient(app) as client:
            token = generate_csrf_token()
            r = client.post(
                "/api/test",
                cookies={CSRF_COOKIE: token},
                # Pas de header — protection double-submit doit refuser.
            )
            assert r.status_code == 403

    def test_post_with_divergent_cookie_and_header_returns_403(
        self, csrf_required: None,
    ) -> None:
        from fastapi.testclient import TestClient
        from picarones.interfaces.web.security import (
            CSRF_COOKIE,
            CSRF_HEADER,
            generate_csrf_token,
        )

        app = self._make_app()
        with TestClient(app) as client:
            cookie_token = generate_csrf_token()
            header_token = generate_csrf_token()  # différent
            r = client.post(
                "/api/test",
                cookies={CSRF_COOKIE: cookie_token},
                headers={CSRF_HEADER: header_token},
            )
            assert r.status_code == 403, (
                "Cookie ≠ header doit être refusé (double-submit)."
            )

    def test_post_with_invalid_signature_returns_403(
        self, csrf_required: None,
    ) -> None:
        from fastapi.testclient import TestClient
        from picarones.interfaces.web.security import CSRF_COOKIE, CSRF_HEADER

        app = self._make_app()
        with TestClient(app) as client:
            fake_token = "a" * 32 + "." + "f" * 64
            r = client.post(
                "/api/test",
                cookies={CSRF_COOKIE: fake_token},
                headers={CSRF_HEADER: fake_token},
            )
            assert r.status_code == 403

    def test_post_with_valid_token_succeeds(
        self, csrf_required: None,
    ) -> None:
        from fastapi.testclient import TestClient
        from picarones.interfaces.web.security import (
            CSRF_COOKIE,
            CSRF_HEADER,
            generate_csrf_token,
        )

        app = self._make_app()
        with TestClient(app) as client:
            token = generate_csrf_token()
            r = client.post(
                "/api/test",
                cookies={CSRF_COOKIE: token},
                headers={CSRF_HEADER: token},
            )
            assert r.status_code == 200, (
                f"POST avec token valide refusé (status={r.status_code}, "
                f"body={r.text[:200]})"
            )

    def test_public_mode_bypasses_csrf(self, csrf_disabled: None) -> None:
        """En l'absence de ``PICARONES_CSRF_REQUIRED``, le middleware
        laisse passer toutes les requêtes (rétrocompat HF Space)."""
        from fastapi.testclient import TestClient

        app = self._make_app()
        with TestClient(app) as client:
            r = client.post("/api/test")
            assert r.status_code == 200, (
                "Mode public : POST doit passer sans token."
            )
