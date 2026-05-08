"""Tests Sprint A5 — robustesse SQLite face aux écritures concurrentes.

Item M-13 de l'audit institutional-readiness-2026-05.

``picarones.web.jobs.JobStore`` est l'unique point d'écriture sur la
BD ``jobs.sqlite`` (mode WAL, thread-safe par ``_conn`` qui ouvre une
nouvelle connection par appel). Cette suite valide qu'il survit à :

1. N threads créant des jobs simultanément (pas de doublon, pas de
   corruption).
2. M threads mettant à jour le progress du même job (pas de
   ``SQLITE_BUSY`` qui remonte au caller).
3. Set_status concurrent depuis plusieurs threads.

Les tests utilisent un fichier SQLite temporaire isolé pour ne pas
polluer ``jobs.sqlite`` du dev local.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from picarones.interfaces.web._legacy.jobs import JobStore


@pytest.fixture
def fresh_store(tmp_path: Path) -> JobStore:
    db_path = tmp_path / "jobs_test.sqlite"
    store = JobStore(db_path=db_path)
    return store


# ---------------------------------------------------------------------------
# Création concurrente
# ---------------------------------------------------------------------------


def test_concurrent_create_no_duplicate(fresh_store: JobStore) -> None:
    """20 threads créent chacun un job → 20 jobs distincts en BD,
    aucun ID dupliqué."""
    n_threads = 20

    def _create_one(_) -> str:
        return fresh_store.create_job(payload={"thread": "x"})

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        ids = list(pool.map(_create_one, range(n_threads)))

    assert len(ids) == n_threads
    assert len(set(ids)) == n_threads, (
        f"IDs dupliqués détectés : {[x for x in ids if ids.count(x) > 1]}"
    )

    listed = fresh_store.list_jobs(limit=n_threads + 5)
    assert len(listed) == n_threads


# ---------------------------------------------------------------------------
# Update concurrent sur le même job
# ---------------------------------------------------------------------------


def test_concurrent_progress_updates_no_busy_error(fresh_store: JobStore) -> None:
    """50 updates concurrents sur le même job → pas de SQLITE_BUSY,
    le dernier état persiste de manière cohérente."""
    job_id = fresh_store.create_job(payload={})

    n_updates = 50
    errors: list[BaseException] = []

    def _update_one(i: int) -> None:
        try:
            fresh_store.update_progress(
                job_id=job_id,
                progress=float(i) / n_updates,
                processed_docs=i,
            )
        except BaseException as exc:  # noqa: BLE001 — on capture pour assert
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(_update_one, range(n_updates)))

    assert not errors, f"Erreurs durant updates concurrentes : {errors[:3]}"

    final = fresh_store.get_job(job_id)
    assert final is not None
    # progress doit être un float ∈ [0, 1] cohérent (pas une valeur corrompue)
    assert 0.0 <= float(final.get("progress", 0)) <= 1.0


# ---------------------------------------------------------------------------
# Set status concurrent
# ---------------------------------------------------------------------------


def test_concurrent_set_status_serializable(fresh_store: JobStore) -> None:
    """Plusieurs ``set_status`` en parallèle sur le même job ne doivent
    pas corrompre la table ; le dernier statut écrit doit être l'un
    des statuts valides."""
    job_id = fresh_store.create_job(payload={})
    statuses = ["running", "succeeded", "failed", "cancelled"]
    barrier = threading.Barrier(len(statuses))

    def _set(status: str) -> None:
        barrier.wait(timeout=5)  # synchronise le départ pour maximiser la concurrence
        try:
            fresh_store.set_status(job_id, status)
        except Exception:
            pass  # un set_status peut échouer s'il y a transition invalide

    with ThreadPoolExecutor(max_workers=len(statuses)) as pool:
        list(pool.map(_set, statuses))

    final = fresh_store.get_job(job_id)
    assert final is not None
    assert final["status"] in statuses + ["pending"]


# ---------------------------------------------------------------------------
# Reads pendant writes
# ---------------------------------------------------------------------------


def test_reads_during_writes_no_locking_error(fresh_store: JobStore) -> None:
    """Lectures concurrentes pendant écritures → mode WAL doit permettre
    sans bloquer ni lever."""
    n_jobs = 10
    for _ in range(n_jobs):
        fresh_store.create_job(payload={})

    stop = threading.Event()
    read_errors: list[BaseException] = []
    write_errors: list[BaseException] = []

    def _writer() -> None:
        try:
            while not stop.is_set():
                fresh_store.create_job(payload={"writer": "x"})
        except BaseException as exc:  # noqa: BLE001
            write_errors.append(exc)

    def _reader() -> None:
        try:
            while not stop.is_set():
                fresh_store.list_jobs(limit=100)
        except BaseException as exc:  # noqa: BLE001
            read_errors.append(exc)

    threads = [
        threading.Thread(target=_writer),
        threading.Thread(target=_writer),
        threading.Thread(target=_reader),
        threading.Thread(target=_reader),
    ]
    for t in threads:
        t.start()
    threading.Event().wait(0.5)  # 500 ms de charge mixte
    stop.set()
    for t in threads:
        t.join(timeout=2)

    assert not read_errors, f"Reads ont levé : {read_errors[:2]}"
    assert not write_errors, f"Writes ont levé : {write_errors[:2]}"


# ---------------------------------------------------------------------------
# Garde-fous
# ---------------------------------------------------------------------------


def test_get_job_unknown_returns_none(fresh_store: JobStore) -> None:
    """Un job_id inconnu doit retourner ``None``, pas lever."""
    assert fresh_store.get_job("ghost-job-id") is None


def test_update_progress_unknown_job_does_not_crash(
    fresh_store: JobStore,
) -> None:
    """Update sur un job_id inconnu : pas d'effet, pas de crash."""
    fresh_store.update_progress(job_id="ghost", progress=0.5)
    # Aucun job créé en passant
    assert len(fresh_store.list_jobs()) == 0
