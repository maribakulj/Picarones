"""Sprint S4.1 — couverture des opérations SQL de ``JobStore``.

Avant S4 : ``job_store.py`` à 64% de couverture.  Lignes non
couvertes : ``create``, ``get``, ``list``, ``update_progress``,
``mark_*``, ``mark_orphaned_jobs_interrupted``, ``_set_status``,
``_row_to_record`` (gestion payload corrompu).

Cible : 90%+ de couverture.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from picarones.adapters.storage.job_store import JobRecord, JobStore, JobStoreError


@pytest.fixture
def store(tmp_path: Path) -> JobStore:
    """JobStore fraîchement créé sur un tmp_path."""
    return JobStore(db_path=tmp_path / "jobs.sqlite")


# ──────────────────────────────────────────────────────────────────────
# create
# ──────────────────────────────────────────────────────────────────────


class TestCreate:
    def test_create_returns_record(self, store: JobStore) -> None:
        rec = store.create("job_001", payload={"corpus": "test"}, total_docs=10)
        assert isinstance(rec, JobRecord)
        assert rec.job_id == "job_001"
        assert rec.status == "pending"
        assert rec.total_docs == 10
        assert rec.progress == 0.0

    def test_create_with_no_payload_uses_empty_dict(
        self, store: JobStore,
    ) -> None:
        rec = store.create("job_002")
        assert rec is not None
        assert rec.status == "pending"

    def test_create_empty_job_id_raises(self, store: JobStore) -> None:
        with pytest.raises(JobStoreError, match="vide"):
            store.create("")

    def test_create_duplicate_job_id_raises(self, store: JobStore) -> None:
        store.create("dup")
        with pytest.raises(JobStoreError, match="déjà existant"):
            store.create("dup")

    def test_create_persists_payload_json(self, store: JobStore) -> None:
        complex_payload = {
            "corpus": "manuscrits",
            "engines": ["tesseract", "pero"],
            "options": {"lang": "fra"},
        }
        store.create("payload_test", payload=complex_payload)
        rec = store.get("payload_test")
        assert rec is not None
        # Le payload est exposé via JobRecord.payload (dict).
        assert rec.payload == complex_payload


# ──────────────────────────────────────────────────────────────────────
# get + list
# ──────────────────────────────────────────────────────────────────────


class TestGetAndList:
    def test_get_unknown_returns_none(self, store: JobStore) -> None:
        assert store.get("does_not_exist") is None

    def test_get_returns_existing_record(self, store: JobStore) -> None:
        store.create("a")
        rec = store.get("a")
        assert rec is not None
        assert rec.job_id == "a"

    def test_list_empty_store_returns_empty_tuple(
        self, store: JobStore,
    ) -> None:
        assert store.list() == ()

    def test_list_orders_by_created_desc(self, store: JobStore) -> None:
        # Crée 3 jobs avec un délai pour garantir l'ordre temporel
        import time
        for i in range(3):
            store.create(f"job_{i:02d}")
            time.sleep(0.01)
        records = store.list()
        assert len(records) == 3
        # Le plus récent en premier
        assert records[0].job_id == "job_02"
        assert records[2].job_id == "job_00"

    def test_list_respects_limit(self, store: JobStore) -> None:
        for i in range(5):
            store.create(f"j{i}")
        results = store.list(limit=2)
        assert len(results) == 2

    def test_list_limit_zero_returns_empty(self, store: JobStore) -> None:
        store.create("j")
        assert store.list(limit=0) == ()


# ──────────────────────────────────────────────────────────────────────
# update_progress
# ──────────────────────────────────────────────────────────────────────


class TestUpdateProgress:
    def test_update_progress_sets_value(self, store: JobStore) -> None:
        store.create("p", total_docs=10)
        store.update_progress("p", progress=0.5, processed_docs=5,
                              current_engine="tesseract")
        rec = store.get("p")
        assert rec is not None
        assert rec.progress == 0.5
        assert rec.processed_docs == 5
        assert rec.current_engine == "tesseract"

    def test_update_progress_clamps_above_one(self, store: JobStore) -> None:
        store.create("p")
        store.update_progress("p", progress=2.5)
        rec = store.get("p")
        assert rec is not None
        assert rec.progress == 1.0

    def test_update_progress_clamps_below_zero(self, store: JobStore) -> None:
        store.create("p")
        store.update_progress("p", progress=-0.5)
        rec = store.get("p")
        assert rec is not None
        assert rec.progress == 0.0

    def test_update_progress_unknown_job_is_silent(
        self, store: JobStore,
    ) -> None:
        # UPDATE WHERE job_id matches nothing — ne lève pas, mutation 0 ligne.
        store.update_progress("ghost", progress=0.3)
        # Aucun side-effect : le job n'apparaît pas après l'opération.
        assert store.get("ghost") is None


# ──────────────────────────────────────────────────────────────────────
# mark_* (transitions de statut)
# ──────────────────────────────────────────────────────────────────────


class TestStatusTransitions:
    def test_mark_running(self, store: JobStore) -> None:
        store.create("r")
        store.mark_running("r")
        rec = store.get("r")
        assert rec is not None
        assert rec.status == "running"
        assert rec.finished_at is None

    def test_mark_complete_sets_output(self, store: JobStore) -> None:
        store.create("c")
        store.mark_complete("c", output_path="/tmp/report.html")
        rec = store.get("c")
        assert rec is not None
        assert rec.status == "complete"
        assert rec.output_path == "/tmp/report.html"
        assert rec.finished_at is not None

    def test_mark_error_sets_message(self, store: JobStore) -> None:
        store.create("e")
        store.mark_error("e", error_message="OCR engine failed")
        rec = store.get("e")
        assert rec is not None
        assert rec.status == "error"
        assert rec.error == "OCR engine failed"
        assert rec.finished_at is not None

    def test_mark_cancelled(self, store: JobStore) -> None:
        store.create("x")
        store.mark_cancelled("x")
        rec = store.get("x")
        assert rec is not None
        assert rec.status == "cancelled"
        assert rec.finished_at is not None

    def test_is_terminal_helper(self, store: JobStore) -> None:
        store.create("t")
        store.mark_complete("t")
        rec = store.get("t")
        assert rec is not None
        assert rec.is_terminal is True
        assert rec.is_live is False


# ──────────────────────────────────────────────────────────────────────
# mark_orphaned_jobs_interrupted (boot cleanup)
# ──────────────────────────────────────────────────────────────────────


class TestOrphanedJobsCleanup:
    def test_pending_and_running_become_interrupted(
        self, store: JobStore,
    ) -> None:
        store.create("p")  # pending
        store.create("r")
        store.mark_running("r")  # running
        store.create("c")
        store.mark_complete("c")  # complete (terminal)

        n = store.mark_orphaned_jobs_interrupted()
        assert n == 2  # p + r

        assert store.get("p").status == "interrupted"  # type: ignore[union-attr]
        assert store.get("r").status == "interrupted"  # type: ignore[union-attr]
        # Le job complete n'est pas affecté.
        assert store.get("c").status == "complete"  # type: ignore[union-attr]

    def test_no_orphans_returns_zero(self, store: JobStore) -> None:
        # Aucun job ou tous terminaux.
        assert store.mark_orphaned_jobs_interrupted() == 0

    def test_orphan_records_carry_explanation(
        self, store: JobStore,
    ) -> None:
        store.create("p")
        store.mark_orphaned_jobs_interrupted()
        rec = store.get("p")
        assert rec is not None
        assert rec.error == "process restart"


# ──────────────────────────────────────────────────────────────────────
# _row_to_record — payload corrompu
# ──────────────────────────────────────────────────────────────────────


class TestPayloadCorruptionTolerance:
    """Le store doit tolérer un payload_json corrompu (downgrade
    de version, écriture concurrente cassée, etc.) sans crasher."""

    def test_corrupted_payload_yields_empty_dict_with_warning(
        self,
        store: JobStore,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        store.create("corrupt")
        # Réécriture brutale du payload_json en JSON invalide.
        with sqlite3.connect(str(store.db_path)) as conn:
            conn.execute(
                "UPDATE jobs SET payload_json = ? WHERE job_id = ?",
                ("{not valid json", "corrupt"),
            )
            conn.commit()

        import logging
        with caplog.at_level(logging.WARNING):
            rec = store.get("corrupt")

        assert rec is not None
        assert rec.payload == {}
        # Un warning doit avoir été émis.
        assert any(
            "corrompu" in r.message.lower() or "corrupt" in r.message
            for r in caplog.records
        )


# ──────────────────────────────────────────────────────────────────────
# Persistence cross-instance — db_path
# ──────────────────────────────────────────────────────────────────────


class TestPersistence:
    def test_jobs_persist_across_store_instances(
        self, tmp_path: Path,
    ) -> None:
        db = tmp_path / "shared.sqlite"
        s1 = JobStore(db_path=db)
        s1.create("persisted", total_docs=42)

        s2 = JobStore(db_path=db)
        rec = s2.get("persisted")
        assert rec is not None
        assert rec.total_docs == 42

    def test_db_path_property_returns_path(self, store: JobStore) -> None:
        assert isinstance(store.db_path, Path)
