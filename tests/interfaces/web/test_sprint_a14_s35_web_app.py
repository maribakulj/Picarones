"""Sprint A14-S35 — squelette FastAPI ``interfaces/web``.

Tests du squelette FastAPI natif qui consomme les services
applicatifs du Sprint S17+.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from picarones.app.services import (
    BenchmarkService,
    CorpusService,
    RegistryService,
    RunOrchestrator,
    WorkspaceManager,
)
from picarones.interfaces.web import (
    HealthResponse,
    VersionResponse,
    WebAppState,
    create_app,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _make_state(tmp_path: Path) -> WebAppState:
    """Construit un ``WebAppState`` avec services réels (registres
    bootstrappés, workspace temporaire)."""
    workspace = WorkspaceManager(
        base_dir=tmp_path,
        session_id="test_session",
    )
    registry = RegistryService.bootstrap_defaults()

    # Pour les tests S35 squelette, on n'a pas besoin de services
    # complètement fonctionnels — des MagicMock conviennent puisque
    # les endpoints squelette ne les invoquent pas.
    corpus = MagicMock(spec=CorpusService)
    benchmark = MagicMock(spec=BenchmarkService)
    orchestrator = MagicMock(spec=RunOrchestrator)

    return WebAppState(
        workspace=workspace,
        registry=registry,
        corpus=corpus,
        benchmark=benchmark,
        orchestrator=orchestrator,
        version="1.0.0-s35-test",
    )


# ──────────────────────────────────────────────────────────────────────
# WebAppState dataclass
# ──────────────────────────────────────────────────────────────────────


class TestWebAppStateDataclass:
    def test_frozen(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        with pytest.raises(Exception):  # FrozenInstanceError
            state.version = "modified"  # type: ignore[misc]

    def test_default_version(self, tmp_path: Path) -> None:
        workspace = WorkspaceManager(base_dir=tmp_path, session_id="test")
        registry = RegistryService.bootstrap_defaults()
        state = WebAppState(
            workspace=workspace,
            registry=registry,
            corpus=MagicMock(),
            benchmark=MagicMock(),
            orchestrator=MagicMock(),
        )
        assert state.version == "1.0.0"


# ──────────────────────────────────────────────────────────────────────
# create_app factory
# ──────────────────────────────────────────────────────────────────────


class TestCreateApp:
    def test_returns_fastapi_instance(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        app = create_app(state)
        assert isinstance(app, FastAPI)

    def test_state_attached_to_app(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        app = create_app(state)
        assert app.state.picarones is state

    def test_rejects_non_state_input(self) -> None:
        with pytest.raises(TypeError, match="WebAppState"):
            create_app("not a state")  # type: ignore[arg-type]

    def test_each_call_yields_new_app(self, tmp_path: Path) -> None:
        """Pas de singleton global — chaque create_app produit une
        instance indépendante."""
        state = _make_state(tmp_path)
        app1 = create_app(state)
        app2 = create_app(state)
        assert app1 is not app2

    def test_app_has_title_and_version(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        app = create_app(state)
        assert app.title == "Picarones"
        assert app.version == state.version

    def test_openapi_doc_endpoints_available(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        app = create_app(state)
        client = TestClient(app)
        # /api/docs et /api/redoc doivent exister.
        r_docs = client.get("/api/docs")
        r_redoc = client.get("/api/redoc")
        assert r_docs.status_code == 200
        assert r_redoc.status_code == 200


# ──────────────────────────────────────────────────────────────────────
# /health endpoint
# ──────────────────────────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_returns_200_ok(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        app = create_app(state)
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body == {"status": "ok"}

    def test_health_response_schema(self, tmp_path: Path) -> None:
        # Le schéma HealthResponse doit valider {"status": "ok"}.
        h = HealthResponse(status="ok")
        assert h.status == "ok"


# ──────────────────────────────────────────────────────────────────────
# /version endpoint
# ──────────────────────────────────────────────────────────────────────


class TestVersionEndpoint:
    def test_version_returns_200_ok(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        app = create_app(state)
        client = TestClient(app)
        response = client.get("/version")
        assert response.status_code == 200

    def test_version_includes_workspace_root(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        app = create_app(state)
        client = TestClient(app)
        response = client.get("/version")
        body = response.json()
        assert "workspace_root" in body
        # Le root doit pointer dans tmp_path.
        assert tmp_path.name in body["workspace_root"]

    def test_version_includes_n_metrics_and_projectors(
        self, tmp_path: Path,
    ) -> None:
        state = _make_state(tmp_path)
        app = create_app(state)
        client = TestClient(app)
        response = client.get("/version")
        body = response.json()
        # Bootstrap par défaut enregistre cer/wer/mer/wil + autres → > 0.
        assert body["n_metrics"] > 0
        assert body["n_projectors"] > 0

    def test_version_string_matches_state(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        app = create_app(state)
        client = TestClient(app)
        response = client.get("/version")
        body = response.json()
        assert body["version"] == "1.0.0-s35-test"

    def test_version_response_schema(self) -> None:
        v = VersionResponse(
            version="1.0.0",
            workspace_root="/tmp/test",
            n_metrics=5,
            n_projectors=3,
        )
        assert v.version == "1.0.0"
        assert v.n_metrics == 5


# ──────────────────────────────────────────────────────────────────────
# Pas de routers métier en S35
# ──────────────────────────────────────────────────────────────────────


class TestSkeletonScope:
    def test_no_corpus_endpoint_yet(self, tmp_path: Path) -> None:
        """S35 ne contient pas encore les endpoints métier (S36+)."""
        state = _make_state(tmp_path)
        app = create_app(state)
        client = TestClient(app)
        # Aucun endpoint corpus / benchmark / jobs n'est attendu en S35.
        for path in ("/api/corpus", "/api/benchmark", "/api/jobs"):
            response = client.get(path)
            assert response.status_code == 404, (
                f"Endpoint {path!r} ne devrait pas exister en S35 — "
                "viendra en S36-S37."
            )

    def test_no_static_mount_yet(self, tmp_path: Path) -> None:
        """S35 ne sert pas encore de fichiers statiques (S38)."""
        state = _make_state(tmp_path)
        app = create_app(state)
        client = TestClient(app)
        response = client.get("/static/css/main.css")
        assert response.status_code == 404
