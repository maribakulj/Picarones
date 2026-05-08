"""Tâches de maintenance asynchrones — purge RGPD des uploads anciens.

Sprint A11 (item M-8 du plan de remédiation institutionnelle).

Démarrée par le ``lifespan`` de ``picarones.web.app``, cette tâche
asyncio scanne le dossier ``uploads/`` toutes les ``interval_seconds``
et supprime tout sous-dossier dont :

1. le ``mtime`` est plus ancien que ``retention_days`` jours, **ET**
2. aucun job actif (``status ∈ {running, queued}``) ne référence
   le ``corpus_id`` correspondant.

Comportement :

- Aucune blockage de l'event loop (utilise ``asyncio.to_thread`` pour
  l'I/O disque).
- Erreur d'I/O sur un sous-dossier individuel : log warning + skip,
  ne tue pas la tâche.
- Configurable via variables d'env (cf. ``data-retention-rgpd.md``).

Désactivation : si ``PICARONES_UPLOAD_RETENTION_DAYS=0``, la tâche
ne fait rien (mode rétention illimitée).
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


def _get_retention_days() -> int:
    """Lit ``PICARONES_UPLOAD_RETENTION_DAYS`` (défaut : 7).

    ``0`` = pas de rétention (mode développement uniquement).
    """
    raw = os.environ.get("PICARONES_UPLOAD_RETENTION_DAYS", "7")
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(
            "[maintenance] PICARONES_UPLOAD_RETENTION_DAYS invalide (%r) — "
            "défaut 7 jours.", raw,
        )
        return 7


def _get_check_interval_seconds() -> int:
    """Lit ``PICARONES_PURGE_INTERVAL_SECONDS`` (défaut : 6 h)."""
    raw = os.environ.get("PICARONES_PURGE_INTERVAL_SECONDS", str(6 * 3600))
    try:
        return max(60, int(raw))  # plancher 1 minute pour éviter la boucle folle
    except ValueError:
        return 6 * 3600


def _list_corpus_dirs(uploads_root: Path) -> list[Path]:
    """Retourne les sous-dossiers de premier niveau de ``uploads_root``."""
    if not uploads_root.exists():
        return []
    return [p for p in uploads_root.iterdir() if p.is_dir()]


def _should_purge(
    corpus_dir: Path,
    retention_days: int,
    active_corpus_ids: set[str],
    now: float | None = None,
) -> bool:
    """Décide si ``corpus_dir`` doit être supprimé.

    Critères (ET logique) :
    - ``mtime`` > ``retention_days`` jours,
    - le ``corpus_id`` (= nom du dossier) n'est pas référencé par un
      job actif.
    """
    if retention_days == 0:
        return False  # rétention illimitée
    if corpus_dir.name in active_corpus_ids:
        return False
    now = now if now is not None else time.time()
    age_seconds = now - corpus_dir.stat().st_mtime
    age_days = age_seconds / 86400
    return age_days > retention_days


def _active_corpus_ids() -> set[str]:
    """Lit la BD jobs et retourne les ``corpus_id`` des jobs en cours.

    Robuste à l'absence de BD (renvoie set vide) : la purge devient
    plus agressive mais cohérente avec l'absence de référence.
    """
    try:
        from picarones.interfaces.web._legacy import state
        store = state.JOB_STORE
        jobs = store.list_jobs(limit=1000, status=None)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[maintenance] BD jobs inaccessible (%s) — purge sans "
            "filtrage par job actif.", exc,
        )
        return set()
    out: set[str] = set()
    for job in jobs:
        if job.get("status") not in ("running", "queued", "pending"):
            continue
        payload = job.get("payload") or {}
        if isinstance(payload, dict):
            corpus = payload.get("corpus") or payload.get("corpus_id")
            if isinstance(corpus, str):
                # Le payload peut contenir un chemin ou un id ; on
                # extrait le dernier composant.
                out.add(Path(corpus).name)
    return out


def purge_old_uploads(
    uploads_root: Path,
    retention_days: int | None = None,
    *,
    active_corpus_ids: Iterable[str] | None = None,
    now: float | None = None,
) -> list[Path]:
    """Effectue une passe de purge synchrone et retourne la liste des
    dossiers supprimés.

    Conçu pour être appelé soit depuis la tâche async (via
    ``asyncio.to_thread``) soit directement depuis un test.
    """
    if retention_days is None:
        retention_days = _get_retention_days()
    active = (
        set(active_corpus_ids)
        if active_corpus_ids is not None
        else _active_corpus_ids()
    )
    purged: list[Path] = []
    for corpus_dir in _list_corpus_dirs(uploads_root):
        try:
            if not _should_purge(corpus_dir, retention_days, active, now=now):
                continue
            shutil.rmtree(corpus_dir)
            logger.info(
                "[maintenance] purged upload %s (retention=%dj, mtime=%s)",
                corpus_dir.name,
                retention_days,
                time.strftime(
                    "%Y-%m-%d", time.gmtime(corpus_dir.stat().st_mtime)
                )
                if corpus_dir.exists()
                else "deleted",
            )
            purged.append(corpus_dir)
        except OSError as exc:
            # Permission denied, fichier verrouillé Windows, etc.
            # Log + continue (la prochaine passe retentera).
            logger.warning(
                "[maintenance] échec purge %s : %s — sera retenté à la "
                "prochaine passe.", corpus_dir, exc,
            )
    return purged


async def upload_purge_task(uploads_root: Path) -> None:
    """Tâche asyncio à démarrer dans le ``lifespan`` de l'application.

    Boucle infinie : purge → sleep(interval) → purge.  Capture les
    erreurs pour ne jamais tomber.  Annulable via ``task.cancel()``.
    """
    interval = _get_check_interval_seconds()
    retention = _get_retention_days()
    if retention == 0:
        logger.info(
            "[maintenance] PICARONES_UPLOAD_RETENTION_DAYS=0 — tâche "
            "de purge désactivée."
        )
        return
    logger.info(
        "[maintenance] purge auto activée : %d jours de rétention, "
        "passage toutes les %d secondes (uploads_root=%s)",
        retention, interval, uploads_root,
    )
    try:
        while True:
            try:
                await asyncio.to_thread(purge_old_uploads, uploads_root)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[maintenance] passe de purge a échoué (%s) — "
                    "retentée dans %d s.", exc, interval,
                )
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        logger.info("[maintenance] tâche de purge annulée proprement.")
        raise


__all__ = [
    "purge_old_uploads",
    "upload_purge_task",
]
