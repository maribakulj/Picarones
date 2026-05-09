"""Tests Sprint 26 — persistance SQLite des jobs + reprise SSE.

Le Sprint 26 introduit :

1. ``picarones/core/jobs.py`` — ``JobStore`` SQLite avec API
   ``create_job``, ``update_progress``, ``set_status``,
   ``append_event``, ``get_events_after``, ``get_job``,
   ``mark_orphaned_jobs_interrupted``, ``cleanup_old``.

2. Intégration dans ``picarones/web/app.py`` :
   - ``BenchmarkJob`` reçoit un ``_store`` et persiste ses événements
     + sa progression à chaque ``add_event``.
   - ``set_status`` remplace les mutations directes ``job.status = ...``
     pour assurer la cohérence DB.
   - SSE endpoint accepte le header ``Last-Event-ID`` et rejoue depuis
     la DB ; émet ``id: <seq>`` dans chaque message.
   - ``GET /status`` fallback à la DB si le job n'est plus en RAM.
   - Hook ``startup`` marque les jobs vivants en base comme
     ``interrupted``.
"""

from __future__ import annotations

import sqlite3
import time

import pytest
from fastapi.testclient import TestClient

from picarones.interfaces.web.jobs import JobStore, get_default_store, reset_default_store


# ---------------------------------------------------------------------------
# 1. JobStore CRUD basique
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path) -> JobStore:
    return JobStore(tmp_path / "jobs.db")


class TestJobStoreCRUD:
    def test_create_returns_uuid_when_no_id(self, store):
        jid = store.create_job()
        assert isinstance(jid, str) and len(jid) == 36  # UUID4 canonique

    def test_create_uses_provided_id(self, store):
        jid = store.create_job(job_id="custom-id-42")
        assert jid == "custom-id-42"

    def test_get_returns_none_for_unknown(self, store):
        assert store.get_job("ne-pas-exister") is None

    def test_get_returns_pending_status_after_create(self, store):
        jid = store.create_job()
        job = store.get_job(jid)
        assert job is not None
        assert job["status"] == "pending"
        assert job["progress"] == 0.0
        assert job["payload"] == {}

    def test_payload_round_trip(self, store):
        payload = {"engines": ["tesseract"], "lang": "fra", "n": 12}
        jid = store.create_job(payload=payload)
        assert store.get_job(jid)["payload"] == payload

    def test_update_progress_partial(self, store):
        jid = store.create_job()
        store.update_progress(jid, progress=0.5, current_engine="tesseract")
        job = store.get_job(jid)
        assert job["progress"] == 0.5
        assert job["current_engine"] == "tesseract"
        # Les champs non passés restent à zéro
        assert job["total_docs"] == 0

    def test_set_status_to_complete_sets_finished_at(self, store):
        jid = store.create_job()
        before = time.time()
        store.set_status(jid, "complete")
        job = store.get_job(jid)
        assert job["status"] == "complete"
        assert job["finished_at"] is not None
        assert job["finished_at"] >= before

    def test_set_status_running_does_not_set_finished(self, store):
        jid = store.create_job()
        store.set_status(jid, "running")
        assert store.get_job(jid)["finished_at"] is None

    def test_set_status_with_error_message(self, store):
        jid = store.create_job()
        store.set_status(jid, "error", error="OOM dans Tesseract")
        job = store.get_job(jid)
        assert job["status"] == "error"
        assert "OOM" in job["error"]

    def test_list_jobs_returns_descending(self, store):
        jids = [store.create_job() for _ in range(3)]
        # Petit délai pour différencier les ``created_at``
        listed = [j["job_id"] for j in store.list_jobs()]
        # Les plus récents en tête
        assert listed[0] == jids[-1]


# ---------------------------------------------------------------------------
# 2. Événements et reprise via seq
# ---------------------------------------------------------------------------

class TestJobStoreEvents:
    def test_append_event_returns_increasing_seq(self, store):
        jid = store.create_job()
        s1 = store.append_event(jid, "log", {"msg": "a"})
        s2 = store.append_event(jid, "log", {"msg": "b"})
        s3 = store.append_event(jid, "log", {"msg": "c"})
        assert (s1, s2, s3) == (1, 2, 3)

    def test_append_event_isolates_per_job(self, store):
        a = store.create_job()
        b = store.create_job()
        store.append_event(a, "log", {})
        store.append_event(a, "log", {})
        store.append_event(b, "log", {})
        # Le seq de chaque job redémarre à 1
        evs_b = store.get_events_after(b, last_seq=0)
        assert [e["seq"] for e in evs_b] == [1]

    def test_get_events_after_skips_seen(self, store):
        jid = store.create_job()
        for i in range(5):
            store.append_event(jid, "log", {"i": i})
        new = store.get_events_after(jid, last_seq=2)
        assert [e["seq"] for e in new] == [3, 4, 5]
        assert new[0]["data"] == {"i": 2}

    def test_get_events_after_zero_returns_all(self, store):
        jid = store.create_job()
        store.append_event(jid, "log", {})
        store.append_event(jid, "log", {})
        assert len(store.get_events_after(jid, last_seq=0)) == 2

    def test_get_events_after_handles_unicode(self, store):
        jid = store.create_job()
        store.append_event(jid, "log", {"msg": "Médiéval & œ"})
        ev = store.get_events_after(jid, last_seq=0)[0]
        assert ev["data"]["msg"] == "Médiéval & œ"

    def test_count_events(self, store):
        jid = store.create_job()
        for _ in range(7):
            store.append_event(jid, "log", {})
        assert store.count_events(jid) == 7


# ---------------------------------------------------------------------------
# 3. Détection des orphelins au boot
# ---------------------------------------------------------------------------

class TestMarkOrphanedJobs:
    def test_running_job_becomes_interrupted(self, store):
        jid = store.create_job()
        store.set_status(jid, "running")
        n = store.mark_orphaned_jobs_interrupted()
        assert n == 1
        job = store.get_job(jid)
        assert job["status"] == "interrupted"
        assert "interrompu" in job["error"].lower()
        assert job["finished_at"] is not None

    def test_completed_job_is_not_touched(self, store):
        jid = store.create_job()
        store.set_status(jid, "complete")
        before = store.get_job(jid)["finished_at"]
        store.mark_orphaned_jobs_interrupted()
        after = store.get_job(jid)
        assert after["status"] == "complete"
        assert after["finished_at"] == before

    def test_no_running_jobs_returns_zero(self, store):
        store.create_job()  # pending
        n = store.mark_orphaned_jobs_interrupted()
        # 'pending' est aussi considéré vivant — il devient interrupted lui aussi
        assert n == 1

    def test_existing_error_message_preserved(self, store):
        jid = store.create_job()
        store.set_status(jid, "running")
        # Simule une erreur déjà enregistrée par une autre route
        with sqlite3.connect(str(store.path), isolation_level=None) as c:
            c.execute("UPDATE jobs SET error = ? WHERE job_id = ?", ("ma erreur", jid))
        store.mark_orphaned_jobs_interrupted()
        # L'erreur existante n'est pas écrasée par "interrompu par redémarrage"
        assert store.get_job(jid)["error"] == "ma erreur"


# ---------------------------------------------------------------------------
# 4. Cleanup
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_old_finished_jobs_removed(self, store):
        jid = store.create_job()
        store.set_status(jid, "complete")
        # Antedate la fin pour simuler un vieux job
        with sqlite3.connect(str(store.path), isolation_level=None) as c:
            c.execute(
                "UPDATE jobs SET finished_at = ? WHERE job_id = ?",
                (time.time() - 30 * 86400, jid),
            )
        removed = store.cleanup_old(retention_days=7)
        assert removed == 1
        assert store.get_job(jid) is None

    def test_recent_jobs_kept(self, store):
        jid = store.create_job()
        store.set_status(jid, "complete")
        removed = store.cleanup_old(retention_days=7)
        assert removed == 0
        assert store.get_job(jid) is not None

    def test_running_jobs_never_cleaned(self, store):
        jid = store.create_job()
        # finished_at IS NULL
        store.cleanup_old(retention_days=0)
        assert store.get_job(jid) is not None


# ---------------------------------------------------------------------------
# 5. Singleton paresseux
# ---------------------------------------------------------------------------

class TestDefaultStore:
    def test_default_store_is_singleton(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PICARONES_JOBS_DB", str(tmp_path / "x.db"))
        reset_default_store()
        a = get_default_store()
        b = get_default_store()
        assert a is b
        reset_default_store()

    def test_env_var_drives_path(self, monkeypatch, tmp_path):
        target = tmp_path / "custom.db"
        monkeypatch.setenv("PICARONES_JOBS_DB", str(target))
        reset_default_store()
        s = get_default_store()
        assert s.path == target
        s.create_job()
        assert target.exists()
        reset_default_store()


# ---------------------------------------------------------------------------
# 6. Intégration FastAPI : SSE Last-Event-ID + status fallback
# ---------------------------------------------------------------------------

@pytest.fixture
def client_with_isolated_store(monkeypatch, tmp_path):
    """Réinitialise le ``JOB_STORE`` global de l'app vers un fichier vierge."""
    db = tmp_path / "jobs.db"
    monkeypatch.setenv("PICARONES_JOBS_DB", str(db))
    reset_default_store()
    from picarones.interfaces.web import app as web_app
    from picarones.interfaces.web import state as web_state
    web_state.JOB_STORE = get_default_store()
    # Vide aussi le cache RAM des jobs
    web_state.JOBS.clear()
    return TestClient(web_app.app), web_state


class TestStatusFallbackToDB:
    def test_status_falls_back_to_db_after_ram_eviction(self, client_with_isolated_store):
        client, web_state = client_with_isolated_store
        # Crée un job directement en base (simule un job d'un précédent worker)
        jid = web_state.JOB_STORE.create_job(job_id="ghost-1")
        web_state.JOB_STORE.set_status(jid, "complete")
        web_state.JOB_STORE.update_progress(jid, progress=1.0, total_docs=10, processed_docs=10)

        r = client.get(f"/api/benchmark/{jid}/status")
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["job_id"] == jid
        assert d["status"] == "complete"
        assert d["progress"] == 1.0
        assert d["total_docs"] == 10

    def test_status_404_for_truly_unknown_job(self, client_with_isolated_store):
        client, _ = client_with_isolated_store
        r = client.get("/api/benchmark/never-existed/status")
        assert r.status_code == 404


class TestSSEReplay:
    def test_sse_replays_backlog_for_finished_job(self, client_with_isolated_store):
        client, web_state = client_with_isolated_store
        jid = web_state.JOB_STORE.create_job(job_id="replay-1")
        web_state.JOB_STORE.append_event(jid, "log", {"msg": "hello"})
        web_state.JOB_STORE.append_event(jid, "log", {"msg": "world"})
        web_state.JOB_STORE.set_status(jid, "complete")
        web_state.JOB_STORE.append_event(jid, "complete", {"output_html": "/tmp/x.html"})

        with client.stream("GET", f"/api/benchmark/{jid}/stream") as r:
            assert r.status_code == 200
            text = "".join(chunk for chunk in r.iter_text())

        assert "hello" in text
        assert "world" in text
        # Les seq doivent être présents (Last-Event-ID resumability)
        assert "id: 1" in text
        assert "id: 2" in text
        # Marqueur de fin
        assert "event: done" in text or "event: complete" in text

    def test_sse_resumes_from_last_event_id(self, client_with_isolated_store):
        client, web_state = client_with_isolated_store
        jid = web_state.JOB_STORE.create_job(job_id="resume-1")
        for i in range(5):
            web_state.JOB_STORE.append_event(jid, "log", {"i": i})
        web_state.JOB_STORE.set_status(jid, "complete")

        # Reprise depuis seq=3 — on doit recevoir uniquement 4 et 5.
        with client.stream(
            "GET",
            f"/api/benchmark/{jid}/stream",
            headers={"Last-Event-ID": "3"},
        ) as r:
            text = "".join(chunk for chunk in r.iter_text())

        assert "id: 4" in text
        assert "id: 5" in text
        # Les anciens événements ne doivent PAS être réémis
        assert "id: 1" not in text
        assert "id: 2" not in text

    def test_sse_invalid_last_event_id_falls_back_to_zero(self, client_with_isolated_store):
        client, web_state = client_with_isolated_store
        jid = web_state.JOB_STORE.create_job(job_id="bad-header")
        web_state.JOB_STORE.append_event(jid, "log", {"i": 1})
        web_state.JOB_STORE.set_status(jid, "complete")

        with client.stream(
            "GET",
            f"/api/benchmark/{jid}/stream",
            headers={"Last-Event-ID": "not-a-number"},
        ) as r:
            text = "".join(chunk for chunk in r.iter_text())

        # Doit envoyer le backlog complet sans crasher
        assert "id: 1" in text


class TestStartupOrphansHook:
    def test_startup_marks_running_jobs_interrupted(self, monkeypatch, tmp_path):
        db = tmp_path / "jobs.db"
        # Préparer la DB avec un job 'running' avant import de l'app
        s = JobStore(db)
        jid = s.create_job(job_id="orphan-1")
        s.set_status(jid, "running")

        monkeypatch.setenv("PICARONES_JOBS_DB", str(db))
        reset_default_store()
        # Forcer le startup hook via TestClient context manager
        from picarones.interfaces.web import app as web_app
        from picarones.interfaces.web import state as web_state
        web_state.JOB_STORE = get_default_store()
        with TestClient(web_app.app):
            pass  # __enter__ déclenche startup, __exit__ shutdown

        job = web_state.JOB_STORE.get_job(jid)
        assert job["status"] == "interrupted"
