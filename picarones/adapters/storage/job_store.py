"""``JobStore`` — Sprint A14-S37.

Persistance SQLite des jobs de benchmark.  Adapté du legacy
``picarones.web.jobs`` mais réécrit nativement pour le nouveau monde :
API plus simple, dataclass immuable, sans dépendance au ``state``
global.

Le legacy reste exposé jusqu'au S46.

Pourquoi SQLite
---------------
- Survie au redémarrage : un crash ou ``kill -HUP`` ne perd pas
  l'état des jobs en cours.
- Détection des jobs orphelins au boot : tout job ``running`` à
  l'initialisation est forcément un zombie du process précédent
  → marqué ``interrupted``.
- Indexation simple par ``job_id`` (TEXT PK).
- Mode WAL pour les lectures concurrentes pendant qu'un thread
  écrit la progression.

Statuts
-------
- ``pending``      : créé, en attente d'exécution.
- ``running``      : worker actif.
- ``complete``     : succès.
- ``error``        : échec applicatif (avec message).
- ``cancelled``    : interrompu par le caller.
- ``interrupted``  : zombie du process précédent (détecté au boot).

Les 4 derniers sont **terminaux** — un job dans cet état ne change
plus de statut.

API publique
------------
- ``JobStore(db_path)`` : connexion SQLite, init schema si absent.
- ``create(job_id, payload, total_docs=0)`` → JobRecord.
- ``get(job_id)`` → JobRecord | None.
- ``list(limit=None)`` → tuple[JobRecord, ...] triés par
  ``created_at`` décroissant.
- ``update_progress(job_id, progress, processed_docs, current_engine)``.
- ``mark_running(job_id)``.
- ``mark_complete(job_id, output_path="")``.
- ``mark_error(job_id, error_message)``.
- ``mark_cancelled(job_id)``.
- ``mark_orphaned_jobs_interrupted()`` → int (nombre marqué).
- ``close()`` (no-op : chaque appel ouvre/ferme sa propre connexion).

Anti-sur-ingénierie
-------------------
- Pas de notification SSE (les SSE legacy sont reportés à un sprint
  dédié si un caller en a besoin).
- Pas de queue d'événements — le legacy avait ``job_events`` ; on
  attend qu'un caller en ait besoin ; pour l'instant le statut +
  progress suffit pour le polling.
- Une connexion par appel — SQLite gère ça en sub-ms.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_TERMINAL_STATUSES: frozenset[str] = frozenset({
    "complete", "error", "cancelled", "interrupted",
})

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

CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs(status);
CREATE INDEX IF NOT EXISTS jobs_created_idx ON jobs(created_at);
"""


@dataclass(frozen=True)
class JobRecord:
    """Snapshot immuable d'un job persisté.

    Les setters mutants (``update_progress``, ``mark_*``) reconstruisent
    un nouveau ``JobRecord`` au prochain ``get``.
    """

    job_id: str
    status: str
    progress: float
    current_engine: str
    total_docs: int
    processed_docs: int
    output_path: str
    error: str
    payload: dict[str, Any]
    created_at: float
    updated_at: float
    finished_at: float | None

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL_STATUSES

    @property
    def is_live(self) -> bool:
        return self.status in _LIVE_STATUSES


class JobStoreError(Exception):
    """Erreur de persistance SQLite côté JobStore."""


class JobStore:
    """Store SQLite des jobs de benchmark.

    Parameters
    ----------
    db_path:
        Chemin du fichier SQLite.  Créé s'il n'existe pas.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Initialisation du schéma + WAL.
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)
            try:
                conn.execute("PRAGMA journal_mode = WAL;")
            except sqlite3.Error:  # pragma: no cover
                # Si WAL pas supporté (FAT32, etc.), on continue en
                # mode rollback journal — déjà fonctionnel.
                pass

    @property
    def db_path(self) -> Path:
        return self._path

    def _connect(self) -> sqlite3.Connection:
        """Ouvre une nouvelle connexion.  Le caller est responsable
        du commit + close (on utilise le context manager Python qui
        gère ça automatiquement)."""
        conn = sqlite3.connect(
            str(self._path),
            isolation_level=None,  # autocommit pour simplicité
            timeout=10.0,
        )
        conn.row_factory = sqlite3.Row
        return conn

    # ──────────────────────────────────────────────────────────────
    # Création / lecture
    # ──────────────────────────────────────────────────────────────

    def create(
        self,
        job_id: str,
        payload: dict[str, Any] | None = None,
        total_docs: int = 0,
    ) -> JobRecord:
        """Crée un nouveau job en statut ``pending``.

        Raises
        ------
        JobStoreError
            Si ``job_id`` existe déjà ou si la ligne ne s'insère
            pas correctement.
        """
        if not job_id:
            raise JobStoreError("create : job_id vide non autorisé.")
        now = time.time()
        payload_json = json.dumps(payload or {}, ensure_ascii=False)
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO jobs (
                        job_id, status, progress, current_engine,
                        total_docs, processed_docs, output_path, error,
                        payload_json, created_at, updated_at, finished_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id, "pending", 0.0, "",
                        total_docs, 0, "", "",
                        payload_json, now, now, None,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise JobStoreError(
                f"job_id {job_id!r} déjà existant.",
            ) from exc
        return self.get(job_id)  # type: ignore[return-value]

    def get(self, job_id: str) -> JobRecord | None:
        """Retourne le snapshot du job, ou ``None`` si inconnu."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def list(self, limit: int | None = None) -> tuple[JobRecord, ...]:
        """Liste les jobs triés par date de création décroissante."""
        sql = "SELECT * FROM jobs ORDER BY created_at DESC"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        with self._connect() as conn:
            rows = conn.execute(sql).fetchall()
        return tuple(self._row_to_record(r) for r in rows)

    # ──────────────────────────────────────────────────────────────
    # Mutations
    # ──────────────────────────────────────────────────────────────

    def update_progress(
        self,
        job_id: str,
        progress: float,
        processed_docs: int = 0,
        current_engine: str = "",
    ) -> None:
        """Met à jour la progression d'un job en ``running``.

        ``progress`` est tronqué à [0.0, 1.0].
        """
        progress = max(0.0, min(1.0, progress))
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET progress = ?, processed_docs = ?,
                    current_engine = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (progress, processed_docs, current_engine, now, job_id),
            )

    def mark_running(self, job_id: str) -> None:
        """Bascule le statut en ``running``."""
        self._set_status(job_id, "running", finished=False)

    def mark_complete(self, job_id: str, output_path: str = "") -> None:
        self._set_status(
            job_id, "complete", finished=True, output_path=output_path,
        )

    def mark_error(self, job_id: str, error_message: str) -> None:
        self._set_status(
            job_id, "error", finished=True, error=error_message,
        )

    def mark_cancelled(self, job_id: str) -> None:
        self._set_status(job_id, "cancelled", finished=True)

    def mark_orphaned_jobs_interrupted(self) -> int:
        """Marque tous les jobs ``pending``/``running`` comme
        ``interrupted``.  Appelé au boot de l'app pour nettoyer les
        zombies du process précédent.

        Returns
        -------
        int
            Nombre de jobs marqués.
        """
        now = time.time()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE jobs
                SET status = 'interrupted',
                    error = 'process restart',
                    updated_at = ?,
                    finished_at = ?
                WHERE status IN ('pending', 'running')
                """,
                (now, now),
            )
            return cur.rowcount

    # ──────────────────────────────────────────────────────────────
    # Helpers privés
    # ──────────────────────────────────────────────────────────────

    def _set_status(
        self,
        job_id: str,
        status: str,
        *,
        finished: bool,
        output_path: str = "",
        error: str = "",
    ) -> None:
        now = time.time()
        finished_at = now if finished else None
        with self._connect() as conn:
            if finished:
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = ?, output_path = ?, error = ?,
                        updated_at = ?, finished_at = ?
                    WHERE job_id = ?
                    """,
                    (status, output_path, error, now, finished_at, job_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = ?, updated_at = ?, finished_at = ?
                    WHERE job_id = ?
                    """,
                    (status, now, finished_at, job_id),
                )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> JobRecord:
        try:
            payload = json.loads(row["payload_json"] or "{}")
        except json.JSONDecodeError:
            logger.warning(
                "[job_store] payload corrompu pour job %s — ignoré.",
                row["job_id"],
            )
            payload = {}
        return JobRecord(
            job_id=row["job_id"],
            status=row["status"],
            progress=row["progress"],
            current_engine=row["current_engine"],
            total_docs=row["total_docs"],
            processed_docs=row["processed_docs"],
            output_path=row["output_path"],
            error=row["error"],
            payload=payload,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            finished_at=row["finished_at"],
        )


__all__ = [
    "JobRecord",
    "JobStore",
    "JobStoreError",
]
