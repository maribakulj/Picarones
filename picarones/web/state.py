"""État partagé du serveur web FastAPI — singletons et helpers transverses.

Ce module centralise tout ce qui est partagé entre routeurs : la
classe ``BenchmarkJob`` qui modélise un job en cours, le store SQLite
qui le persiste, le rate limiter, le sémaphore qui borne le nombre
de jobs concurrents, ainsi que les constantes et utilitaires
datetime/HTTP utilisés à plusieurs endroits.

Discipline : aucun routeur ne doit définir ses propres ``iso_now`` /
``enforce_rate_limit`` — tous passent par ce module pour garantir
la cohérence.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException, Request

from picarones.web.jobs import JobStore, get_default_store
from picarones.web.security import (
    RateLimiter,
    get_max_concurrent_jobs,
    get_rate_limit_per_hour,
)

_logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Constantes partagées
# ──────────────────────────────────────────────────────────────────────────

IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"})
"""Extensions d'image acceptées à l'upload et lors de la validation corpus."""

UPLOADS_DIR = Path("./uploads")
"""Dossier où sont stockés les corpus uploadés via l'interface web."""

SUPPORTED_LANGS = ("fr", "en")
"""Langues supportées par l'interface."""

LANG_COOKIE = "picarones_lang"
"""Nom du cookie qui mémorise la langue choisie par l'utilisateur."""


# ──────────────────────────────────────────────────────────────────────────
# Helpers transverses
# ──────────────────────────────────────────────────────────────────────────

def iso_now() -> str:
    """Timestamp ISO 8601 UTC (précision seconde)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _client_ip(request: Request) -> str:
    """IP client en respectant ``X-Forwarded-For`` derrière un proxy.

    Helper interne au module — utilisé uniquement par
    ``enforce_rate_limit``. Pas exposé dans ``__all__`` car aucun
    consommateur externe n'en a besoin (un router qui veut l'IP doit
    appeler ``enforce_rate_limit`` directement).
    """
    fwd = request.headers.get("x-forwarded-for") or ""
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def enforce_rate_limit(request: Request) -> None:
    """Applique le rate limit ; lève ``HTTPException 429`` si dépassé."""
    try:
        RATE_LIMITER.check(_client_ip(request))
    except PermissionError as exc:
        raise HTTPException(status_code=429, detail=str(exc))


# ──────────────────────────────────────────────────────────────────────────
# Singletons : rate limiter, sémaphore, job store
# ──────────────────────────────────────────────────────────────────────────

RATE_LIMITER = RateLimiter(max_per_hour=get_rate_limit_per_hour())
"""Rate limiter global (no-op si non public ou quota = 0). Sprint 24."""

JOBS_SEMAPHORE = threading.Semaphore(get_max_concurrent_jobs())
"""Sémaphore qui borne le nombre de benchmarks concurrents. Sprint 24."""

JOB_STORE: JobStore = get_default_store()
"""Store SQLite singleton injecté dans chaque ``BenchmarkJob``. Sprint 26."""


# ──────────────────────────────────────────────────────────────────────────
# Modèle de job (avec persistance Sprint 26)
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkJob:
    """Job de benchmark en cours d'exécution.

    Chaque job a un ``job_id`` unique, un statut, une progression et
    un flux d'événements consommé via SSE. La persistance est gérée
    par un ``JobStore`` SQLite optionnel — si présent, chaque
    événement est sérialisé en base avant d'être diffusé aux abonnés
    SSE, ce qui permet la reprise via ``Last-Event-ID`` (Sprint 26).
    """

    job_id: str
    status: str = "pending"
    """Un des : ``pending``, ``running``, ``complete``, ``error``,
    ``cancelled``, ``interrupted``."""
    progress: float = 0.0  # 0.0 – 1.0
    current_engine: str = ""
    total_docs: int = 0
    processed_docs: int = 0
    output_path: str = ""
    error: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    events: list[dict] = field(default_factory=list)
    _subscribers: list[asyncio.Queue] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _cancel_event: threading.Event = field(default_factory=threading.Event)
    _store: Optional[JobStore] = None
    """Store SQLite optionnel injecté à la création. Si ``None``,
    le job vit uniquement en mémoire."""

    def add_event(self, kind: str, data: Any) -> None:
        """Persiste l'événement dans le store puis le diffuse aux abonnés SSE.

        L'ordre persistance → diffusion garantit qu'à chaque ``seq``
        rendu visible au client, le snapshot du job en base est
        cohérent avec ce que vit le client (reprise possible via
        ``Last-Event-ID``).
        """
        seq: Optional[int] = None
        if self._store is not None:
            try:
                seq = self._store.append_event(self.job_id, kind, data)
                self._store.update_progress(
                    self.job_id,
                    progress=self.progress,
                    current_engine=self.current_engine,
                    total_docs=self.total_docs,
                    processed_docs=self.processed_docs,
                    output_path=self.output_path,
                )
            except Exception as exc:  # pragma: no cover — défense en profondeur
                _logger.warning(
                    "[jobs] persistance d'événement échouée pour %s : %s",
                    self.job_id, exc,
                )
        event = {"kind": kind, "data": data, "ts": iso_now(), "seq": seq}
        with self._lock:
            self.events.append(event)
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                _logger.warning(
                    "[jobs] queue SSE pleine pour job %s — événement déjà persisté seq=%s",
                    self.job_id, seq,
                )

    def set_status(self, status: str, error: str = "") -> None:
        """Met à jour le statut + persiste vers le store (Sprint 26)."""
        self.status = status
        if error:
            self.error = error
        if status in ("complete", "error", "cancelled", "interrupted"):
            self.finished_at = iso_now()
        if self._store is not None:
            try:
                self._store.set_status(
                    self.job_id, status, error=error or None,
                )
            except Exception as exc:  # pragma: no cover
                _logger.warning(
                    "[jobs] set_status persisté en échec pour %s : %s",
                    self.job_id, exc,
                )

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def as_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": self.progress,
            "current_engine": self.current_engine,
            "total_docs": self.total_docs,
            "processed_docs": self.processed_docs,
            "output_path": self.output_path,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


# ──────────────────────────────────────────────────────────────────────────
# Registre en mémoire des jobs actifs
# ──────────────────────────────────────────────────────────────────────────

JOBS: dict[str, BenchmarkJob] = {}
"""Registre en mémoire des jobs (par ``job_id``)."""

JOBS_MAX = 100
"""Nombre maximum de jobs conservés en mémoire avant nettoyage."""

JOBS_LOCK = threading.Lock()


def cleanup_old_jobs() -> None:
    """Supprime les jobs terminés les plus anciens si on dépasse ``JOBS_MAX``."""
    with JOBS_LOCK:
        if len(JOBS) <= JOBS_MAX:
            return
        finished = [
            (jid, j) for jid, j in JOBS.items()
            if j.status in ("complete", "error", "cancelled")
        ]
        finished.sort(key=lambda x: x[1].finished_at or "")
        to_remove = len(JOBS) - JOBS_MAX
        for jid, _ in finished[:to_remove]:
            del JOBS[jid]


__all__ = [
    "IMAGE_EXTS",
    "UPLOADS_DIR",
    "SUPPORTED_LANGS",
    "LANG_COOKIE",
    "iso_now",
    "enforce_rate_limit",
    "RATE_LIMITER",
    "JOBS_SEMAPHORE",
    "JOB_STORE",
    "BenchmarkJob",
    "JOBS",
    "JOBS_MAX",
    "JOBS_LOCK",
    "cleanup_old_jobs",
]
