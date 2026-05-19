"""Sprint S4.10 — couverture directe de ``JobRunner``.

Avant S4 : 0% direct (des tests transitifs existaient avant H.4
mais les chemins canoniques étaient peu couverts).

Cible : 85%+ — vérifie le contrat ``submit`` / ``wait`` avec un
orchestrator factice qui n'a pas besoin de Tesseract ni de réseau.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from picarones.adapters.storage.job_store import JobStore
from picarones.app.services.job_runner import JobRunner


# ──────────────────────────────────────────────────────────────────────
# Stub orchestrator
# ──────────────────────────────────────────────────────────────────────


class _StubOrchestrator:
    """Orchestrator de test : ne fait rien, retourne un manifest
    fictif."""

    def __init__(self, output_dir: Path, *, raise_on_execute: Exception | None = None,
                 delay: float = 0.0) -> None:
        self.output_dir = output_dir
        self.execute_called = False
        self._raise = raise_on_execute
        self._delay = delay
        self.manifest_path = output_dir / "run_manifest.json"

    def execute(self, run_spec: Any, *, report_renderer: Any = None) -> Any:
        import time
        if self._delay:
            time.sleep(self._delay)
        if self._raise:
            raise self._raise
        self.execute_called = True
        return type("FakeResult", (), {
            "manifest_path": self.manifest_path,
            "report_path": None,
        })()


def _factory_with_stub(*, raise_on_execute: Exception | None = None,
                       delay: float = 0.0):
    def _factory(output_dir: Path) -> _StubOrchestrator:
        return _StubOrchestrator(
            output_dir, raise_on_execute=raise_on_execute, delay=delay,
        )
    return _factory


@pytest.fixture
def store(tmp_path: Path) -> JobStore:
    return JobStore(db_path=tmp_path / "jobs.sqlite")


# ──────────────────────────────────────────────────────────────────────
# 1. submit + wait flow normal
# ──────────────────────────────────────────────────────────────────────


class TestSubmitNormalFlow:
    def test_submit_returns_job_id(
        self, store: JobStore, tmp_path: Path,
    ) -> None:
        runner = JobRunner(
            job_store=store,
            orchestrator_factory=_factory_with_stub(),
        )
        job_id = runner.submit(
            run_spec={},
            output_dir=tmp_path / "out",
        )
        assert isinstance(job_id, str)
        assert len(job_id) >= 8

    def test_wait_completes(
        self, store: JobStore, tmp_path: Path,
    ) -> None:
        runner = JobRunner(
            job_store=store,
            orchestrator_factory=_factory_with_stub(),
        )
        job_id = runner.submit(run_spec={}, output_dir=tmp_path / "out")
        finished = runner.wait(job_id, timeout=10.0)
        assert finished is True
        # Le statut DB doit être ``complete`` ou similaire
        rec = store.get(job_id)
        assert rec is not None
        assert rec.status in ("complete", "running", "pending")

    def test_explicit_job_id_is_respected(
        self, store: JobStore, tmp_path: Path,
    ) -> None:
        runner = JobRunner(
            job_store=store,
            orchestrator_factory=_factory_with_stub(),
        )
        job_id = runner.submit(
            run_spec={},
            output_dir=tmp_path / "out",
            job_id="explicit_id",
        )
        assert job_id == "explicit_id"
        runner.wait(job_id, timeout=5.0)

    def test_payload_persisted_in_store(
        self, store: JobStore, tmp_path: Path,
    ) -> None:
        runner = JobRunner(
            job_store=store,
            orchestrator_factory=_factory_with_stub(),
        )
        job_id = runner.submit(
            run_spec={},
            output_dir=tmp_path / "out",
            payload={"corpus": "test"},
        )
        runner.wait(job_id, timeout=5.0)
        rec = store.get(job_id)
        assert rec is not None
        assert rec.payload.get("corpus") == "test"
        assert rec.payload.get("output_dir")  # auto-injecté


# ──────────────────────────────────────────────────────────────────────
# 2. Exception dans l'orchestrator → status=error
# ──────────────────────────────────────────────────────────────────────


class TestOrchestratorFailure:
    def test_exception_marks_job_error(
        self, store: JobStore, tmp_path: Path,
    ) -> None:
        runner = JobRunner(
            job_store=store,
            orchestrator_factory=_factory_with_stub(
                raise_on_execute=RuntimeError("orchestrator boom"),
            ),
        )
        job_id = runner.submit(run_spec={}, output_dir=tmp_path / "out")
        runner.wait(job_id, timeout=5.0)
        rec = store.get(job_id)
        assert rec is not None
        assert rec.status == "error"
        assert "boom" in rec.error or "error" in rec.error.lower()


# ──────────────────────────────────────────────────────────────────────
# 3. Validation des paramètres au constructeur
# ──────────────────────────────────────────────────────────────────────


class TestConstructorValidation:
    def test_invalid_job_store_raises(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="JobStore"):
            JobRunner(
                job_store="not a store",  # type: ignore[arg-type]
                orchestrator_factory=_factory_with_stub(),
            )

    def test_invalid_orchestrator_factory_raises(
        self, store: JobStore,
    ) -> None:
        with pytest.raises(TypeError, match="callable"):
            JobRunner(
                job_store=store,
                orchestrator_factory="not callable",  # type: ignore[arg-type]
            )

    def test_invalid_report_renderer_raises(
        self, store: JobStore,
    ) -> None:
        with pytest.raises(TypeError, match="callable"):
            JobRunner(
                job_store=store,
                orchestrator_factory=_factory_with_stub(),
                report_renderer="not callable",  # type: ignore[arg-type]
            )


# ──────────────────────────────────────────────────────────────────────
# 4. Wait sur job inconnu
# ──────────────────────────────────────────────────────────────────────


class TestWaitEdgeCases:
    def test_wait_unknown_job_returns_true(
        self, store: JobStore, tmp_path: Path,
    ) -> None:
        runner = JobRunner(
            job_store=store,
            orchestrator_factory=_factory_with_stub(),
        )
        # job inconnu = considéré déjà fini
        assert runner.wait("ghost_job", timeout=1.0) is True

    def test_wait_timeout_returns_false(
        self, store: JobStore, tmp_path: Path,
    ) -> None:
        runner = JobRunner(
            job_store=store,
            orchestrator_factory=_factory_with_stub(delay=2.0),
        )
        job_id = runner.submit(run_spec={}, output_dir=tmp_path / "out")
        # Timeout court — le job n'aura pas fini
        assert runner.wait(job_id, timeout=0.1) is False
        # Cleanup : attendre que le thread se termine
        runner.wait(job_id, timeout=5.0)
