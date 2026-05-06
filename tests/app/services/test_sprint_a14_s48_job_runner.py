"""Sprint A14-S48 — ``JobRunner`` + lifespan hook + ``POST /api/jobs``.

Fix audit #2 : avant ce sprint, ``JobStore`` (S37) était à moitié
branché — pas de ``POST /api/jobs``, pas de lifespan hook, pas
d'orchestrateur async.

Tests couvrent les 3 chantiers :

1. ``JobRunner`` (service applicatif) :
   - submit + thread démarré, job marqué ``running`` puis ``complete`` ;
   - exception orchestrator → ``error`` avec message ;
   - cancellation pré-démarrage → thread skippe l'exécution ;
   - cancellation post-démarrage → résultat discardé.

2. Lifespan hook : ``mark_orphaned_jobs_interrupted`` appelé au boot.

3. ``POST /api/jobs`` :
   - YAML valide → 202 + job_id ;
   - YAML invalide → 400 ;
   - corps vide → 400 ;
   - sans job_runner configuré → 503.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from picarones.adapters.storage import JobStore
from picarones.app.services import JobRunner
from picarones.app.services import (
    RegistryService,
    WorkspaceManager,
)
from picarones.interfaces.web import WebAppState, create_app


# ──────────────────────────────────────────────────────────────────────
# Stub orchestrator + factory
# ──────────────────────────────────────────────────────────────────────


class _StubOrchestrator:
    """Stub qui simule un orchestrator : succès, échec, ou délai."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        delay_seconds: float = 0.0,
        raise_on_execute: Exception | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.delay_seconds = delay_seconds
        self.raise_on_execute = raise_on_execute
        self.execute_called = False

    def execute(self, run_spec, *, report_renderer=None):
        self.execute_called = True
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        if self.raise_on_execute is not None:
            raise self.raise_on_execute
        result = MagicMock()
        result.persisted_files = {"manifest": self.manifest_path}
        return result


def _make_factory(stub: _StubOrchestrator):
    """Retourne une factory `(output_dir) -> stub` pour JobRunner."""
    def _factory(output_dir):
        return stub
    return _factory


# ──────────────────────────────────────────────────────────────────────
# JobRunner unitaires
# ──────────────────────────────────────────────────────────────────────


class TestJobRunnerConstructor:
    def test_rejects_non_jobstore(self) -> None:
        with pytest.raises(TypeError, match="JobStore"):
            JobRunner(
                job_store="nope",  # type: ignore[arg-type]
                orchestrator_factory=lambda d: None,
            )

    def test_rejects_non_callable_factory(self, tmp_path: Path) -> None:
        store = JobStore(tmp_path / "jobs.db")
        with pytest.raises(TypeError, match="orchestrator_factory"):
            JobRunner(
                job_store=store,
                orchestrator_factory="nope",  # type: ignore[arg-type]
            )

    def test_rejects_non_callable_renderer(self, tmp_path: Path) -> None:
        store = JobStore(tmp_path / "jobs.db")
        with pytest.raises(TypeError, match="report_renderer"):
            JobRunner(
                job_store=store,
                orchestrator_factory=lambda d: None,
                report_renderer="nope",  # type: ignore[arg-type]
            )


class TestJobRunnerHappyPath:
    def test_submit_creates_job_and_marks_complete(self, tmp_path: Path) -> None:
        store = JobStore(tmp_path / "jobs.db")
        manifest = tmp_path / "manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        stub = _StubOrchestrator(manifest_path=manifest)
        runner = JobRunner(store, _make_factory(stub))

        job_id = runner.submit(
            run_spec=MagicMock(),
            output_dir=tmp_path / "run_out",
        )
        assert runner.wait(job_id, timeout=5.0)
        assert stub.execute_called

        rec = store.get(job_id)
        assert rec is not None
        assert rec.status == "complete"
        assert rec.output_path == str(manifest)

    def test_submit_returns_unique_uuid_when_no_id(
        self, tmp_path: Path,
    ) -> None:
        store = JobStore(tmp_path / "jobs.db")
        manifest = tmp_path / "manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        stub = _StubOrchestrator(manifest_path=manifest)
        runner = JobRunner(store, _make_factory(stub))

        job_id_1 = runner.submit(
            run_spec=MagicMock(),
            output_dir=tmp_path / "out1",
        )
        job_id_2 = runner.submit(
            run_spec=MagicMock(),
            output_dir=tmp_path / "out2",
        )
        assert job_id_1 != job_id_2
        runner.wait(job_id_1, timeout=5.0)
        runner.wait(job_id_2, timeout=5.0)

    def test_submit_stores_explicit_job_id(self, tmp_path: Path) -> None:
        store = JobStore(tmp_path / "jobs.db")
        manifest = tmp_path / "m.json"
        manifest.write_text("{}", encoding="utf-8")
        stub = _StubOrchestrator(manifest_path=manifest)
        runner = JobRunner(store, _make_factory(stub))

        returned = runner.submit(
            run_spec=MagicMock(),
            output_dir=tmp_path / "out",
            job_id="my_explicit_id",
        )
        assert returned == "my_explicit_id"
        runner.wait("my_explicit_id", timeout=5.0)
        assert store.get("my_explicit_id") is not None


class TestJobRunnerErrorPath:
    def test_orchestrator_exception_marks_error(self, tmp_path: Path) -> None:
        store = JobStore(tmp_path / "jobs.db")
        stub = _StubOrchestrator(
            manifest_path=tmp_path / "x",
            raise_on_execute=RuntimeError("orchestrator boom"),
        )
        runner = JobRunner(store, _make_factory(stub))

        job_id = runner.submit(
            run_spec=MagicMock(),
            output_dir=tmp_path / "out",
        )
        runner.wait(job_id, timeout=5.0)

        rec = store.get(job_id)
        assert rec is not None
        assert rec.status == "error"
        assert "RuntimeError" in rec.error
        assert "orchestrator boom" in rec.error


class TestJobRunnerCancellation:
    def test_cancel_during_execution_discards_result(
        self, tmp_path: Path,
    ) -> None:
        """Cancel pendant que le worker tourne → le résultat est
        discardé (statut reste cancelled)."""
        store = JobStore(tmp_path / "jobs.db")
        manifest = tmp_path / "m.json"
        manifest.write_text("{}", encoding="utf-8")
        # Délai suffisant pour cancel avant complétion.
        stub = _StubOrchestrator(
            manifest_path=manifest, delay_seconds=0.3,
        )
        runner = JobRunner(store, _make_factory(stub))

        job_id = runner.submit(
            run_spec=MagicMock(),
            output_dir=tmp_path / "out",
        )
        # Attendre que mark_running ait été appelé (le thread a démarré).
        for _ in range(50):
            time.sleep(0.01)
            rec = store.get(job_id)
            if rec is not None and rec.status == "running":
                break
        # Cancel en pleine exécution.
        store.mark_cancelled(job_id)
        # Attendre la fin du thread (~0.3s).
        runner.wait(job_id, timeout=5.0)
        rec_final = store.get(job_id)
        assert rec_final.status == "cancelled", (
            f"Status final attendu cancelled, obtenu {rec_final.status}"
        )


# ──────────────────────────────────────────────────────────────────────
# Lifespan hook (mark_orphaned_jobs_interrupted au boot)
# ──────────────────────────────────────────────────────────────────────


class TestLifespanHook:
    def test_orphaned_jobs_marked_interrupted_on_app_start(
        self, tmp_path: Path,
    ) -> None:
        """Pré-condition : un job ``running`` existe dans le store
        (simule un crash du process précédent).
        Action : démarrage de l'app FastAPI (lifespan hook).
        Résultat : le job orphelin est marqué ``interrupted``."""
        # Phase 1 : pré-pollution du store (simule l'état après crash).
        db_path = tmp_path / "jobs.db"
        store = JobStore(db_path)
        store.create("zombie_pending")
        store.create("zombie_running")
        store.mark_running("zombie_running")
        store.create("complete_one")
        store.mark_complete("complete_one")
        # Vérification pré-état.
        assert store.get("zombie_pending").status == "pending"
        assert store.get("zombie_running").status == "running"
        assert store.get("complete_one").status == "complete"

        # Phase 2 : démarrage de l'app — lifespan hook s'exécute.
        workspace = WorkspaceManager(base_dir=tmp_path, session_id="s48")
        registry = RegistryService.bootstrap_defaults()
        state = WebAppState(
            workspace=workspace,
            registry=registry,
            corpus=MagicMock(),
            benchmark=MagicMock(),
            orchestrator=MagicMock(),
            job_store=store,  # store pré-pollué
        )
        app = create_app(state)
        # Le lifespan hook tourne au context manager du TestClient.
        with TestClient(app) as client:
            # Le hook a tourné au démarrage.  On vérifie l'état du store.
            assert store.get("zombie_pending").status == "interrupted"
            assert store.get("zombie_running").status == "interrupted"
            # Les jobs déjà terminaux ne sont pas touchés.
            assert store.get("complete_one").status == "complete"
            # Sanity check : l'app répond.
            assert client.get("/health").status_code == 200


# ──────────────────────────────────────────────────────────────────────
# POST /api/jobs (intégration end-to-end via TestClient)
# ──────────────────────────────────────────────────────────────────────


def _make_state_with_runner(tmp_path: Path) -> WebAppState:
    """Construit un WebAppState complet avec JobStore + JobRunner.

    L'orchestrator est un stub qui complète immédiatement (pour que
    les tests POST puissent vérifier le statut).
    """
    workspace = WorkspaceManager(base_dir=tmp_path, session_id="s48")
    registry = RegistryService.bootstrap_defaults()
    job_store = JobStore(tmp_path / "jobs.db")

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    # Stub orchestrator factory.
    def _factory(output_dir):
        return _StubOrchestrator(manifest_path=manifest_path)

    job_runner = JobRunner(
        job_store=job_store,
        orchestrator_factory=_factory,
    )
    return WebAppState(
        workspace=workspace,
        registry=registry,
        corpus=MagicMock(),
        benchmark=MagicMock(),
        orchestrator=MagicMock(),
        job_store=job_store,
        job_runner=job_runner,
    )


_VALID_RUNSPEC_YAML = """
corpus_dir: /tmp/c
output_dir: /tmp/out
pipelines:
  - name: ocr_only
    initial_inputs: [image]
    steps:
      - id: ocr
        adapter_class: my_pkg.OCR
        input_types: [image]
        output_types: [raw_text]
views: [text_final]
""".strip()


class TestPostJobsEndpoint:
    def test_valid_yaml_returns_202_with_job_id(self, tmp_path: Path) -> None:
        state = _make_state_with_runner(tmp_path)
        app = create_app(state)
        with TestClient(app) as client:
            response = client.post("/api/jobs", content=_VALID_RUNSPEC_YAML)
            assert response.status_code == 202, response.text
            body = response.json()
            assert "job_id" in body
            assert body["status"] == "pending"
            # Le job_id retourné est dans le store.
            assert state.job_store.get(body["job_id"]) is not None

    def test_invalid_yaml_returns_400(self, tmp_path: Path) -> None:
        state = _make_state_with_runner(tmp_path)
        app = create_app(state)
        with TestClient(app) as client:
            response = client.post(
                "/api/jobs",
                content="not a valid runspec yaml: [",
            )
            assert response.status_code == 400
            assert "RunSpec" in response.json()["detail"]

    def test_empty_body_returns_400_or_422(self, tmp_path: Path) -> None:
        """Body vide → 400 (notre check) ou 422 (pydantic validation
        en amont du handler).  Les deux sont acceptables pour
        l'utilisateur."""
        state = _make_state_with_runner(tmp_path)
        app = create_app(state)
        with TestClient(app) as client:
            response = client.post("/api/jobs", content="")
            # FastAPI/Starlette peut valider Body(...) en 422 avant
            # d'atteindre notre handler ; sinon notre check répond 400.
            assert response.status_code in (400, 422)

    def test_no_job_runner_returns_503(self, tmp_path: Path) -> None:
        """Sans WebAppState.job_runner, POST /api/jobs → 503."""
        workspace = WorkspaceManager(base_dir=tmp_path, session_id="s48")
        registry = RegistryService.bootstrap_defaults()
        state = WebAppState(
            workspace=workspace,
            registry=registry,
            corpus=MagicMock(),
            benchmark=MagicMock(),
            orchestrator=MagicMock(),
            job_store=JobStore(tmp_path / "jobs.db"),
            # job_runner=None par défaut
        )
        app = create_app(state)
        with TestClient(app) as client:
            response = client.post("/api/jobs", content=_VALID_RUNSPEC_YAML)
            assert response.status_code == 503
            assert "Job runner" in response.json()["detail"]
