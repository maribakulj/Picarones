"""Persistance SQLite des jobs de benchmark (Sprint 26).

Avant le Sprint 26, l'état des benchmarks vivait uniquement en mémoire dans
``picarones.web.app._JOBS``. Trois conséquences :

1. Un redémarrage du worker uvicorn (OOM, déploiement, ``kill -HUP``)
   perdait l'état de tous les benchmarks en cours et un client SSE qui se
   reconnectait recevait un ``404`` cohérent.
2. Le ``asyncio.Queue(maxsize=200)`` des SSE perdait silencieusement des
   événements (``put_nowait`` swallow ``QueueFull``).
3. Aucune trace pour debug si un benchmark se figeait — pas d'historique
   au-delà de ce que ``BenchmarkJob.events`` portait en RAM.

Le Sprint 26 adresse les trois en persistant les jobs et leurs événements
dans une base SQLite locale (cohérent avec ``picarones.core.history``,
qui utilise déjà SQLite). La base joue trois rôles :

- **Source de vérité** pour le statut/progression d'un job — ``BenchmarkJob``
  reste comme cache RAM mais n'est plus le ground truth.
- **Backlog d'événements** pour les SSE — un client qui reprend la
  connexion envoie ``Last-Event-ID`` et reçoit tous les événements de
  ``seq > last_seq`` puis bascule en streaming live.
- **Détection des jobs orphelins** au boot — tout job ``running`` à
  l'initialisation de l'app est marqué ``interrupted`` (le processus
  précédent est mort sans le finir).

Conventions
-----------
- Une seule base par instance (``./jobs.db`` par défaut, configurable via
  l'env var ``PICARONES_JOBS_DB``).
- Mode WAL activé pour permettre les lectures concurrentes pendant qu'un
  thread écrit la progression.
- Chaque appel ouvre une nouvelle connexion : SQLite gère lui-même le
  pool ; ouvrir/fermer est sub-milliseconde.
- Les ``data`` d'événement et le ``payload`` du job sont stockés en JSON
  texte (``ensure_ascii=False`` pour la lisibilité dans ``sqlite3 .dump``).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


# Statuts terminaux : un job dans cet état ne peut plus changer.
_TERMINAL_STATUSES: frozenset[str] = frozenset({
    "complete", "error", "cancelled", "interrupted"
})

# Statuts vivants : un job dans cet état est marqué orphelin au boot.
_LIVE_STATUSES: frozenset[str] = frozenset({"pending", "running"})


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id          TEXT PRIMARY KEY,
    status          TEXT NOT NULL DEFAULT 'pending',
    progress        REAL NOT NULL DEFAULT 0.0,
    current_engine  TEXT NOT NULL DEFAULT '',
    total_docs      INTEGER NOT NULL DEFAULT 0,
    processed_docs  INTEGER NOT NULL DEFAULT 0,
    output_path     TEXT NOT NULL DEFAULT '',
    error           TEXT NOT NULL DEFAULT '',
    payload_json    TEXT NOT NULL DEFAULT '{}',
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    finished_at     REAL
);

CREATE TABLE IF NOT EXISTS job_events (
    job_id   TEXT NOT NULL,
    seq      INTEGER NOT NULL,
    kind     TEXT NOT NULL,
    data_json TEXT NOT NULL,
    ts       REAL NOT NULL,
    PRIMARY KEY (job_id, seq)
);

CREATE INDEX IF NOT EXISTS job_events_seq_idx ON job_events(job_id, seq);
CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs(status);
CREATE INDEX IF NOT EXISTS jobs_created_idx ON jobs(created_at);
"""


def _default_db_path() -> Path:
    """Chemin par défaut, surchargeable via ``PICARONES_JOBS_DB``."""
    env = os.environ.get("PICARONES_JOBS_DB")
    if env:
        return Path(env)
    return Path("./jobs.db")


class JobStore:
    """Backend SQLite thread-safe pour la persistance des jobs."""

    def __init__(self, db_path: str | Path | None = None):
        self._path = Path(db_path) if db_path is not None else _default_db_path()
        self._init_lock = threading.Lock()
        self._init_schema()

    @property
    def path(self) -> Path:
        return self._path

    # ---- Connexion -------------------------------------------------------

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        # ``check_same_thread=False`` parce qu'on ouvre une connexion par
        # appel — la sécurité vient de ne pas partager la connexion entre
        # threads. ``isolation_level=None`` pour gérer nous-mêmes les
        # transactions via ``BEGIN``/``COMMIT``.
        c = sqlite3.connect(
            str(self._path),
            isolation_level=None,
            check_same_thread=False,
        )
        c.row_factory = sqlite3.Row
        try:
            c.execute("PRAGMA journal_mode = WAL")
            c.execute("PRAGMA synchronous = NORMAL")
            c.execute("PRAGMA foreign_keys = ON")
            yield c
        finally:
            c.close()

    def _init_schema(self) -> None:
        # Crée le parent si l'utilisateur a passé un chemin imbriqué.
        with self._init_lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._conn() as c:
                c.executescript(_SCHEMA_SQL)

    # ---- Opérations sur les jobs -----------------------------------------

    def create_job(
        self,
        job_id: Optional[str] = None,
        payload: Optional[dict] = None,
    ) -> str:
        """Crée un job ``pending`` ; retourne son ``job_id``."""
        jid = job_id or str(uuid.uuid4())
        now = time.time()
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO jobs
                  (job_id, status, payload_json, created_at, updated_at)
                VALUES (?, 'pending', ?, ?, ?)
                """,
                (jid, json.dumps(payload or {}, ensure_ascii=False), now, now),
            )
        return jid

    def get_job(self, job_id: str) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if row is None:
            return None
        d = dict(row)
        try:
            d["payload"] = json.loads(d.pop("payload_json", "{}"))
        except json.JSONDecodeError:
            d["payload"] = {}
        return d

    def list_jobs(self, limit: int = 100, status: Optional[str] = None) -> list[dict]:
        with self._conn() as c:
            if status:
                rows = c.execute(
                    "SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["payload"] = json.loads(d.pop("payload_json", "{}"))
            except json.JSONDecodeError:
                d["payload"] = {}
            out.append(d)
        return out

    def update_progress(
        self,
        job_id: str,
        *,
        progress: Optional[float] = None,
        current_engine: Optional[str] = None,
        total_docs: Optional[int] = None,
        processed_docs: Optional[int] = None,
        output_path: Optional[str] = None,
    ) -> None:
        """Met à jour les champs de progression d'un job (les ``None`` sont ignorés)."""
        fields: list[str] = []
        values: list[Any] = []
        if progress is not None:
            fields.append("progress = ?")
            values.append(float(progress))
        if current_engine is not None:
            fields.append("current_engine = ?")
            values.append(current_engine)
        if total_docs is not None:
            fields.append("total_docs = ?")
            values.append(int(total_docs))
        if processed_docs is not None:
            fields.append("processed_docs = ?")
            values.append(int(processed_docs))
        if output_path is not None:
            fields.append("output_path = ?")
            values.append(output_path)
        if not fields:
            return
        fields.append("updated_at = ?")
        values.append(time.time())
        values.append(job_id)
        with self._conn() as c:
            c.execute(
                f"UPDATE jobs SET {', '.join(fields)} WHERE job_id = ?",
                values,
            )

    def set_status(
        self,
        job_id: str,
        status: str,
        *,
        error: Optional[str] = None,
    ) -> None:
        now = time.time()
        finished_at = now if status in _TERMINAL_STATUSES else None
        with self._conn() as c:
            c.execute(
                """
                UPDATE jobs
                SET status = ?, error = COALESCE(?, error),
                    updated_at = ?,
                    finished_at = COALESCE(?, finished_at)
                WHERE job_id = ?
                """,
                (status, error, now, finished_at, job_id),
            )

    def mark_orphaned_jobs_interrupted(self) -> int:
        """À appeler au boot : passe en ``interrupted`` les jobs en vie.

        Sous l'hypothèse qu'au boot d'un nouveau processus, *aucun* job
        ne peut être réellement vivant — le précédent worker a forcément
        été tué entre-temps. Retourne le nombre de jobs marqués.
        """
        now = time.time()
        with self._conn() as c:
            cur = c.execute(
                """
                UPDATE jobs
                SET status = 'interrupted',
                    updated_at = ?,
                    finished_at = ?,
                    error = CASE
                      WHEN error = '' THEN 'Job interrompu par redémarrage du serveur.'
                      ELSE error
                    END
                WHERE status IN ('pending', 'running')
                """,
                (now, now),
            )
            count = cur.rowcount
        if count > 0:
            logger.warning(
                "[jobs] %d job(s) orphelin(s) marqué(s) 'interrupted' au boot.", count
            )
        return count

    def cleanup_old(self, retention_days: int = 7) -> int:
        """Supprime les jobs terminés depuis plus de ``retention_days`` jours."""
        cutoff = time.time() - retention_days * 86400.0
        with self._conn() as c:
            cur = c.execute(
                """
                DELETE FROM jobs
                WHERE finished_at IS NOT NULL AND finished_at < ?
                """,
                (cutoff,),
            )
            removed = cur.rowcount
            # Cascade manuelle (pas de FK ON DELETE — schéma léger).
            c.execute(
                """
                DELETE FROM job_events
                WHERE job_id NOT IN (SELECT job_id FROM jobs)
                """
            )
        return removed

    # ---- Événements ------------------------------------------------------

    def append_event(self, job_id: str, kind: str, data: Any) -> int:
        """Ajoute un événement et retourne son numéro de séquence (>= 1)."""
        with self._conn() as c:
            row = c.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM job_events WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            next_seq = int(row[0]) + 1 if row else 1
            c.execute(
                """
                INSERT INTO job_events (job_id, seq, kind, data_json, ts)
                VALUES (?, ?, ?, ?, ?)
                """,
                (job_id, next_seq, kind, json.dumps(data, ensure_ascii=False), time.time()),
            )
        return next_seq

    def get_events_after(self, job_id: str, last_seq: int = 0) -> list[dict]:
        """Retourne les événements ``seq > last_seq``, triés croissant."""
        with self._conn() as c:
            rows = c.execute(
                """
                SELECT seq, kind, data_json, ts
                FROM job_events
                WHERE job_id = ? AND seq > ?
                ORDER BY seq ASC
                """,
                (job_id, int(last_seq)),
            ).fetchall()
        out: list[dict] = []
        for r in rows:
            try:
                data = json.loads(r["data_json"])
            except json.JSONDecodeError:
                data = {}
            out.append({
                "seq": int(r["seq"]),
                "kind": r["kind"],
                "data": data,
                "ts": float(r["ts"]),
            })
        return out

    def count_events(self, job_id: str) -> int:
        with self._conn() as c:
            row = c.execute(
                "SELECT COUNT(*) FROM job_events WHERE job_id = ?", (job_id,)
            ).fetchone()
        return int(row[0]) if row else 0


# ---------------------------------------------------------------------------
# Singleton paresseux — facilite l'import depuis web/app.py.
# ---------------------------------------------------------------------------

_default_store: Optional[JobStore] = None
_default_lock = threading.Lock()


def get_default_store() -> JobStore:
    """Retourne (ou crée) le ``JobStore`` par défaut.

    Le chemin est lu depuis ``PICARONES_JOBS_DB`` à la première création.
    """
    global _default_store
    with _default_lock:
        if _default_store is None:
            _default_store = JobStore()
        return _default_store


def reset_default_store() -> None:
    """Réinitialise le store par défaut (utilisé par les tests)."""
    global _default_store
    with _default_lock:
        _default_store = None
