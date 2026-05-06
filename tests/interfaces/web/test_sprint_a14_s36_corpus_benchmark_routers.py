"""Sprint A14-S36 — routers corpus + benchmark.

Tests des endpoints livrés au S36 dans le squelette FastAPI natif :

- ``POST /api/corpus/import`` : upload ZIP + ``CorpusService``.
- ``GET  /api/runs``           : liste manifests dans le workspace.
- ``GET  /api/runs/{run_id}``  : lit un manifest individuel.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from picarones.app.services import (
    CorpusService,
    RegistryService,
    WorkspaceManager,
)
from picarones.interfaces.web import WebAppState, create_app


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _make_zip_bytes(entries: dict[str, bytes]) -> bytes:
    """Construit un ZIP en mémoire avec les entrées données."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in entries.items():
            zf.writestr(name, content)
    return buf.getvalue()


def _make_state(tmp_path: Path) -> WebAppState:
    workspace = WorkspaceManager(
        base_dir=tmp_path,
        session_id="s36_test",
    )
    registry = RegistryService.bootstrap_defaults()
    corpus = CorpusService(workspace=workspace)
    return WebAppState(
        workspace=workspace,
        registry=registry,
        corpus=corpus,
        benchmark=MagicMock(),
        orchestrator=MagicMock(),
        version="1.0.0-s36-test",
    )


def _make_minimal_image_bytes() -> bytes:
    """Crée une image PNG 10x10 valide pour passer la validation
    d'image de CorpusService (qui peut faire du sniffing)."""
    try:
        from PIL import Image
        import numpy as np
        buf = io.BytesIO()
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        # Si PIL absent, on tente avec des bytes PNG simples ; certains
        # tests vont skip alors.
        return b"\x89PNG\r\n\x1a\n"


# ──────────────────────────────────────────────────────────────────────
# Corpus router
# ──────────────────────────────────────────────────────────────────────


class TestCorpusImportEndpoint:
    def test_import_minimal_corpus(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        app = create_app(state)
        client = TestClient(app)

        zip_bytes = _make_zip_bytes({
            "doc1.png": _make_minimal_image_bytes(),
            "doc1.gt.txt": "Bonjour".encode("utf-8"),
        })
        response = client.post(
            "/api/corpus/import?corpus_name=test_corpus",
            files={"file": ("upload.zip", zip_bytes, "application/zip")},
        )
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["corpus_name"] == "test_corpus"
        assert body["n_documents"] >= 1

    def test_import_rejects_empty_corpus_name(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        app = create_app(state)
        client = TestClient(app)

        zip_bytes = _make_zip_bytes({"x.txt": b"x"})
        response = client.post(
            "/api/corpus/import?corpus_name=",
            files={"file": ("upload.zip", zip_bytes, "application/zip")},
        )
        # FastAPI peut rejeter en 422 si le query param vide n'est pas
        # accepté ; sinon notre code répond 400.
        assert response.status_code in (400, 422)

    def test_import_rejects_empty_file(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        app = create_app(state)
        client = TestClient(app)

        response = client.post(
            "/api/corpus/import?corpus_name=test",
            files={"file": ("upload.zip", b"", "application/zip")},
        )
        assert response.status_code == 400
        assert "vide" in response.json()["detail"].lower()

    def test_import_rejects_garbage_zip(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        app = create_app(state)
        client = TestClient(app)

        # Bytes qui ne sont pas un ZIP valide.
        response = client.post(
            "/api/corpus/import?corpus_name=test",
            files={"file": ("upload.zip", b"not a zip", "application/zip")},
        )
        assert response.status_code == 400


# ──────────────────────────────────────────────────────────────────────
# Benchmark router — list / get
# ──────────────────────────────────────────────────────────────────────


class TestBenchmarkListRunsEndpoint:
    def test_list_empty_runs_returns_empty(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        app = create_app(state)
        client = TestClient(app)
        response = client.get("/api/runs")
        assert response.status_code == 200
        assert response.json() == {"runs": []}

    def test_list_returns_runs_after_persist(self, tmp_path: Path) -> None:
        """Si le workspace contient des runs, le listing les remonte."""
        state = _make_state(tmp_path)
        app = create_app(state)
        client = TestClient(app)

        # Simule un run persisté manuellement (à terme ce sera fait
        # par le RunOrchestrator, ici on teste juste le router).
        runs_dir = Path(state.workspace.root) / "runs"
        runs_dir.mkdir(exist_ok=True)
        run_dir = runs_dir / "run_001"
        run_dir.mkdir()
        manifest = {
            "run_id": "run_001",
            "corpus_name": "demo_corpus",
            "n_documents": 10,
            "pipeline_names": ["tess", "pero"],
            "started_at": "2026-05-06T10:00:00Z",
            "completed_at": "2026-05-06T10:05:00Z",
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )

        response = client.get("/api/runs")
        assert response.status_code == 200
        body = response.json()
        assert len(body["runs"]) == 1
        run = body["runs"][0]
        assert run["run_id"] == "run_001"
        assert run["corpus_name"] == "demo_corpus"
        assert run["n_documents"] == 10
        assert run["pipeline_names"] == ["tess", "pero"]

    def test_list_skips_dirs_without_manifest(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        app = create_app(state)
        client = TestClient(app)

        runs_dir = Path(state.workspace.root) / "runs"
        runs_dir.mkdir(exist_ok=True)
        # Un dir sans manifest est ignoré.
        (runs_dir / "incomplete_run").mkdir()
        # Un dir avec manifest valide est listé.
        valid_dir = runs_dir / "valid_run"
        valid_dir.mkdir()
        (valid_dir / "run_manifest.json").write_text(
            json.dumps({"run_id": "valid_run", "corpus_name": "x"}),
            encoding="utf-8",
        )

        response = client.get("/api/runs")
        body = response.json()
        run_ids = [r["run_id"] for r in body["runs"]]
        assert "valid_run" in run_ids
        assert "incomplete_run" not in run_ids

    def test_list_skips_corrupted_manifest(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        app = create_app(state)
        client = TestClient(app)

        runs_dir = Path(state.workspace.root) / "runs"
        runs_dir.mkdir(exist_ok=True)
        bad = runs_dir / "bad_run"
        bad.mkdir()
        (bad / "run_manifest.json").write_text(
            "this is not json",
            encoding="utf-8",
        )

        response = client.get("/api/runs")
        body = response.json()
        # Le run corrompu est silencieusement ignoré (warning loggé).
        assert all(r["run_id"] != "bad_run" for r in body["runs"])


class TestBenchmarkGetRunEndpoint:
    def test_get_returns_full_manifest(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        app = create_app(state)
        client = TestClient(app)

        runs_dir = Path(state.workspace.root) / "runs"
        runs_dir.mkdir()
        run_dir = runs_dir / "test_run"
        run_dir.mkdir()
        manifest = {
            "run_id": "test_run",
            "corpus_name": "demo",
            "view_specs": [],
            "metadata": {"key": "value"},
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )

        response = client.get("/api/runs/test_run")
        assert response.status_code == 200
        body = response.json()
        assert body["run_id"] == "test_run"
        assert body["raw"] == manifest

    def test_get_unknown_run_returns_404(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        app = create_app(state)
        client = TestClient(app)
        response = client.get("/api/runs/missing_run")
        assert response.status_code == 404

    def test_get_rejects_path_traversal(self, tmp_path: Path) -> None:
        """Un run_id avec '../' ne doit pas pouvoir s'évader du
        workspace."""
        state = _make_state(tmp_path)
        app = create_app(state)
        client = TestClient(app)
        # FastAPI/Starlette résout les .. dans l'URL avant d'arriver au
        # router ; on teste donc la robustesse côté code.
        response = client.get("/api/runs/..%2F..%2Fetc")
        assert response.status_code in (400, 404)

    def test_get_returns_500_on_corrupted_manifest(
        self, tmp_path: Path,
    ) -> None:
        state = _make_state(tmp_path)
        app = create_app(state)
        client = TestClient(app)

        runs_dir = Path(state.workspace.root) / "runs"
        runs_dir.mkdir()
        bad = runs_dir / "bad_run"
        bad.mkdir()
        (bad / "run_manifest.json").write_text(
            "garbage", encoding="utf-8",
        )

        response = client.get("/api/runs/bad_run")
        assert response.status_code == 500
