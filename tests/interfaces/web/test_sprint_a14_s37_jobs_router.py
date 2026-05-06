"""Sprint A14-S37 — JobStore + router jobs.

Tests de la persistance SQLite des jobs et des endpoints
``/api/jobs``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from picarones.adapters.storage import (
    JobRecord,
    JobStore,
    JobStoreError,
)
from picarones.app.services import (
    RegistryService,
    WorkspaceManager,
)
from picarones.interfaces.web import WebAppState, create_app


# ──────────────────────────────────────────────────────────────────────
# JobStore unitaires
# ──────────────────────────────────────────────────────────────────────


class TestJobStoreLifecycle:
    def test_create_then_get(self, tmp_path: Path) -> None:
        store = JobStore(tmp_path / "jobs.db")
        rec = store.create("job_1", payload={"key": "value"}, total_docs=10)
        assert rec.job_id == "job_1"
        assert rec.status == "pending"
        assert rec.progress == 0.0
        assert rec.total_docs == 10
        assert rec.payload == {"key": "value"}
        assert rec.is_live
        assert not rec.is_terminal
        assert rec.finished_at is None

        # get retourne le même snapshot.
        again = store.get("job_1")
        assert again is not None
        assert again.job_id == "job_1"

    def test_get_unknown_returns_none(self, tmp_path: Path) -> None:
        store = JobStore(tmp_path / "jobs.db")
        assert store.get("missing") is None

    def test_create_duplicate_raises(self, tmp_path: Path) -> None:
        store = JobStore(tmp_path / "jobs.db")
        store.create("job_dup")
        with pytest.raises(JobStoreError, match="déjà existant"):
            store.create("job_dup")

    def test_create_empty_id_raises(self, tmp_path: Path) -> None:
        store = JobStore(tmp_path / "jobs.db")
        with pytest.raises(JobStoreError, match="vide"):
            store.create("")

    def test_list_orders_by_created_at_desc(self, tmp_path: Path) -> None:
        import time

        store = JobStore(tmp_path / "jobs.db")
        store.create("a")
        time.sleep(0.01)
        store.create("b")
        time.sleep(0.01)
        store.create("c")
        rows = store.list()
        ids = [r.job_id for r in rows]
        assert ids == ["c", "b", "a"]

    def test_list_with_limit(self, tmp_path: Path) -> None:
        store = JobStore(tmp_path / "jobs.db")
        for i in range(5):
            store.create(f"job_{i}")
        rows = store.list(limit=2)
        assert len(rows) == 2


class TestJobStoreMutations:
    def test_update_progress_clamps(self, tmp_path: Path) -> None:
        store = JobStore(tmp_path / "jobs.db")
        store.create("j1", total_docs=10)
        store.update_progress("j1", progress=2.0, processed_docs=5)
        rec = store.get("j1")
        assert rec.progress == 1.0  # clamped to [0, 1]
        assert rec.processed_docs == 5

    def test_update_progress_negative_clamps_to_zero(
        self, tmp_path: Path,
    ) -> None:
        store = JobStore(tmp_path / "jobs.db")
        store.create("j1")
        store.update_progress("j1", progress=-0.5)
        assert store.get("j1").progress == 0.0

    def test_mark_running(self, tmp_path: Path) -> None:
        store = JobStore(tmp_path / "jobs.db")
        store.create("j1")
        store.mark_running("j1")
        rec = store.get("j1")
        assert rec.status == "running"
        assert rec.is_live
        assert rec.finished_at is None

    def test_mark_complete(self, tmp_path: Path) -> None:
        store = JobStore(tmp_path / "jobs.db")
        store.create("j1")
        store.mark_complete("j1", output_path="/tmp/out.html")
        rec = store.get("j1")
        assert rec.status == "complete"
        assert rec.output_path == "/tmp/out.html"
        assert rec.is_terminal
        assert rec.finished_at is not None

    def test_mark_error(self, tmp_path: Path) -> None:
        store = JobStore(tmp_path / "jobs.db")
        store.create("j1")
        store.mark_error("j1", "something broke")
        rec = store.get("j1")
        assert rec.status == "error"
        assert rec.error == "something broke"
        assert rec.is_terminal

    def test_mark_cancelled(self, tmp_path: Path) -> None:
        store = JobStore(tmp_path / "jobs.db")
        store.create("j1")
        store.mark_cancelled("j1")
        rec = store.get("j1")
        assert rec.status == "cancelled"
        assert rec.is_terminal


class TestJobStoreOrphanRecovery:
    def test_marks_pending_and_running_as_interrupted(
        self, tmp_path: Path,
    ) -> None:
        store = JobStore(tmp_path / "jobs.db")
        store.create("pending_one")
        store.create("running_one")
        store.mark_running("running_one")
        store.create("complete_one")
        store.mark_complete("complete_one")

        n = store.mark_orphaned_jobs_interrupted()
        assert n == 2

        assert store.get("pending_one").status == "interrupted"
        assert store.get("running_one").status == "interrupted"
        assert store.get("complete_one").status == "complete"

    def test_idempotent_after_no_live_jobs(self, tmp_path: Path) -> None:
        store = JobStore(tmp_path / "jobs.db")
        assert store.mark_orphaned_jobs_interrupted() == 0


class TestJobStorePersistence:
    def test_data_survives_reopen(self, tmp_path: Path) -> None:
        db_path = tmp_path / "jobs.db"
        store1 = JobStore(db_path)
        store1.create("j_persistent", payload={"foo": "bar"})

        # Réouvre une nouvelle instance.
        store2 = JobStore(db_path)
        rec = store2.get("j_persistent")
        assert rec is not None
        assert rec.payload == {"foo": "bar"}


class TestJobRecordDataclass:
    def test_frozen(self) -> None:
        rec = JobRecord(
            job_id="x", status="pending", progress=0.0,
            current_engine="", total_docs=0, processed_docs=0,
            output_path="", error="", payload={},
            created_at=0.0, updated_at=0.0, finished_at=None,
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            rec.status = "running"  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────────────
# Router /api/jobs (intégré dans create_app)
# ──────────────────────────────────────────────────────────────────────


def _make_state_with_store(tmp_path: Path) -> WebAppState:
    workspace = WorkspaceManager(
        base_dir=tmp_path,
        session_id="s37_test",
    )
    registry = RegistryService.bootstrap_defaults()
    job_store = JobStore(tmp_path / "jobs.db")
    return WebAppState(
        workspace=workspace,
        registry=registry,
        corpus=MagicMock(),
        benchmark=MagicMock(),
        orchestrator=MagicMock(),
        job_store=job_store,
        version="1.0.0-s37-test",
    )


class TestJobsListEndpoint:
    def test_empty_returns_empty_list(self, tmp_path: Path) -> None:
        state = _make_state_with_store(tmp_path)
        app = create_app(state)
        client = TestClient(app)
        response = client.get("/api/jobs")
        assert response.status_code == 200
        assert response.json() == {"jobs": []}

    def test_lists_existing_jobs(self, tmp_path: Path) -> None:
        state = _make_state_with_store(tmp_path)
        state.job_store.create("job_1", total_docs=5)
        state.job_store.mark_running("job_1")
        state.job_store.update_progress("job_1", 0.4, processed_docs=2)
        state.job_store.create("job_2", total_docs=10)

        app = create_app(state)
        client = TestClient(app)
        response = client.get("/api/jobs")
        assert response.status_code == 200
        body = response.json()
        ids = [j["job_id"] for j in body["jobs"]]
        assert "job_1" in ids
        assert "job_2" in ids

    def test_503_when_no_store(self, tmp_path: Path) -> None:
        """Sans WebAppState.job_store, /api/jobs doit retourner 503."""
        workspace = WorkspaceManager(base_dir=tmp_path, session_id="x")
        registry = RegistryService.bootstrap_defaults()
        state = WebAppState(
            workspace=workspace, registry=registry,
            corpus=MagicMock(), benchmark=MagicMock(),
            orchestrator=MagicMock(),
            # job_store=None par défaut
        )
        app = create_app(state)
        client = TestClient(app)
        response = client.get("/api/jobs")
        assert response.status_code == 503


class TestJobsGetEndpoint:
    def test_get_existing_job(self, tmp_path: Path) -> None:
        state = _make_state_with_store(tmp_path)
        state.job_store.create("job_1", payload={"k": "v"}, total_docs=3)
        app = create_app(state)
        client = TestClient(app)
        response = client.get("/api/jobs/job_1")
        assert response.status_code == 200
        body = response.json()
        assert body["job_id"] == "job_1"
        assert body["payload"] == {"k": "v"}
        assert body["total_docs"] == 3
        assert body["status"] == "pending"

    def test_get_unknown_returns_404(self, tmp_path: Path) -> None:
        state = _make_state_with_store(tmp_path)
        app = create_app(state)
        client = TestClient(app)
        response = client.get("/api/jobs/missing")
        assert response.status_code == 404


class TestJobsCancelEndpoint:
    def test_cancel_pending_job(self, tmp_path: Path) -> None:
        state = _make_state_with_store(tmp_path)
        state.job_store.create("job_1")
        app = create_app(state)
        client = TestClient(app)
        response = client.delete("/api/jobs/job_1")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "cancelled"
        # Vérifie en base.
        assert state.job_store.get("job_1").status == "cancelled"

    def test_cancel_running_job(self, tmp_path: Path) -> None:
        state = _make_state_with_store(tmp_path)
        state.job_store.create("job_1")
        state.job_store.mark_running("job_1")
        app = create_app(state)
        client = TestClient(app)
        response = client.delete("/api/jobs/job_1")
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"

    def test_cancel_terminal_job_idempotent(self, tmp_path: Path) -> None:
        """Annuler un job déjà terminal retourne le statut sans erreur."""
        state = _make_state_with_store(tmp_path)
        state.job_store.create("job_1")
        state.job_store.mark_complete("job_1")
        app = create_app(state)
        client = TestClient(app)
        response = client.delete("/api/jobs/job_1")
        assert response.status_code == 200
        # Statut inchangé.
        assert response.json()["status"] == "complete"
        assert state.job_store.get("job_1").status == "complete"

    def test_cancel_unknown_returns_404(self, tmp_path: Path) -> None:
        state = _make_state_with_store(tmp_path)
        app = create_app(state)
        client = TestClient(app)
        response = client.delete("/api/jobs/missing")
        assert response.status_code == 404
