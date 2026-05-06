"""Sprint A14-S49 — middlewares de sécurité (fix audit #3).

Couvre les 4 middlewares :
1. SecurityHeadersMiddleware — CSP, X-Frame-Options, etc.
2. BodySizeLimitMiddleware — rejet 413 si Content-Length trop gros.
3. RateLimitMiddleware — 429 si dépassement de la fenêtre.
4. AuthenticationMiddleware — 401 si pas authentifié + bypass health/version.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from fastapi import HTTPException, Request, status
from fastapi.testclient import TestClient

from picarones.app.services import RegistryService, WorkspaceManager
from picarones.interfaces.web import WebAppState, create_app


def _state(tmp_path: Path) -> WebAppState:
    return WebAppState(
        workspace=WorkspaceManager(base_dir=tmp_path, session_id="s49"),
        registry=RegistryService.bootstrap_defaults(),
        corpus=MagicMock(),
        benchmark=MagicMock(),
        orchestrator=MagicMock(),
    )


# ──────────────────────────────────────────────────────────────────────
# SecurityHeadersMiddleware
# ──────────────────────────────────────────────────────────────────────


class TestSecurityHeaders:
    def test_csp_present_by_default(self, tmp_path: Path) -> None:
        app = create_app(_state(tmp_path))
        client = TestClient(app)
        r = client.get("/health")
        csp = r.headers.get("content-security-policy", "")
        assert "default-src 'self'" in csp
        assert "frame-ancestors 'none'" in csp
        # Pas d'unsafe-inline.
        assert "unsafe-inline" not in csp

    def test_x_frame_options_deny(self, tmp_path: Path) -> None:
        app = create_app(_state(tmp_path))
        r = TestClient(app).get("/health")
        assert r.headers.get("x-frame-options") == "DENY"

    def test_nosniff_and_referrer_policy(self, tmp_path: Path) -> None:
        app = create_app(_state(tmp_path))
        r = TestClient(app).get("/health")
        assert r.headers.get("x-content-type-options") == "nosniff"
        assert "strict-origin" in r.headers.get("referrer-policy", "")

    def test_can_be_disabled(self, tmp_path: Path) -> None:
        app = create_app(_state(tmp_path), enable_security_headers=False)
        r = TestClient(app).get("/health")
        # Si désactivé, headers absents.
        assert "content-security-policy" not in (h.lower() for h in r.headers)


# ──────────────────────────────────────────────────────────────────────
# BodySizeLimitMiddleware
# ──────────────────────────────────────────────────────────────────────


class TestBodySizeLimit:
    def test_request_with_large_content_length_rejected(self, tmp_path: Path) -> None:
        # Limite à 1 KB.
        app = create_app(_state(tmp_path), max_body_bytes=1024)
        client = TestClient(app)
        # Content-Length annoncé > 1024 → 413 immédiat.
        r = client.post(
            "/api/corpus/import?corpus_name=x",
            content=b"x" * 2048,
            headers={"content-type": "application/zip"},
        )
        assert r.status_code == 413

    def test_request_within_limit_accepted(self, tmp_path: Path) -> None:
        app = create_app(_state(tmp_path), max_body_bytes=1024 * 1024)
        # GET sans body — passe le check.
        r = TestClient(app).get("/health")
        assert r.status_code == 200

    def test_can_be_disabled(self, tmp_path: Path) -> None:
        app = create_app(_state(tmp_path), max_body_bytes=None)
        # Sans middleware → pas de 413 pour gros body.
        r = TestClient(app).get("/health")
        assert r.status_code == 200


# ──────────────────────────────────────────────────────────────────────
# RateLimitMiddleware
# ──────────────────────────────────────────────────────────────────────


class TestRateLimit:
    def test_within_window_passes(self, tmp_path: Path) -> None:
        app = create_app(_state(tmp_path), rate_limit_per_minute=10)
        client = TestClient(app)
        for _ in range(5):
            assert client.get("/health").status_code == 200

    def test_exceeding_returns_429(self, tmp_path: Path) -> None:
        app = create_app(_state(tmp_path), rate_limit_per_minute=3)
        client = TestClient(app)
        # 3 OK puis 1 KO.
        for _ in range(3):
            assert client.get("/health").status_code == 200
        r = client.get("/health")
        assert r.status_code == 429
        assert "Rate limit" in r.json()["detail"]

    def test_can_be_disabled(self, tmp_path: Path) -> None:
        app = create_app(_state(tmp_path), rate_limit_per_minute=None)
        client = TestClient(app)
        # 100 requêtes sans limite.
        for _ in range(100):
            assert client.get("/health").status_code == 200


# ──────────────────────────────────────────────────────────────────────
# AuthenticationMiddleware
# ──────────────────────────────────────────────────────────────────────


class _BearerBackend:
    """Backend qui exige un Bearer token == 'secret'."""

    async def authenticate(self, request: Request) -> None:
        auth = request.headers.get("authorization", "")
        if auth != "Bearer secret":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Bearer token requis.",
            )


class TestAuthentication:
    def test_no_backend_means_public(self, tmp_path: Path) -> None:
        # Mode public par défaut (auth_backend=None).
        app = create_app(_state(tmp_path))
        r = TestClient(app).get("/health")
        assert r.status_code == 200

    def test_protected_endpoint_requires_auth(self, tmp_path: Path) -> None:
        app = create_app(_state(tmp_path), auth_backend=_BearerBackend())
        client = TestClient(app)
        r = client.get("/")  # endpoint home, pas dans la allowlist
        assert r.status_code == 401

    def test_health_bypasses_auth(self, tmp_path: Path) -> None:
        # /health doit toujours répondre (sondes Docker/k8s).
        app = create_app(_state(tmp_path), auth_backend=_BearerBackend())
        r = TestClient(app).get("/health")
        assert r.status_code == 200

    def test_version_bypasses_auth(self, tmp_path: Path) -> None:
        app = create_app(_state(tmp_path), auth_backend=_BearerBackend())
        r = TestClient(app).get("/version")
        assert r.status_code == 200

    def test_valid_token_grants_access(self, tmp_path: Path) -> None:
        app = create_app(_state(tmp_path), auth_backend=_BearerBackend())
        r = TestClient(app).get(
            "/", headers={"authorization": "Bearer secret"},
        )
        assert r.status_code == 200
