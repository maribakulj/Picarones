"""Router jobs — Sprint A14-S37.

Endpoints de listing/lecture/cancellation des jobs de benchmark
persistés via ``JobStore`` (S37, ``picarones.adapters.storage``).

Endpoints
---------
- ``GET    /api/jobs``            : liste des jobs (récents en tête).
- ``GET    /api/jobs/{job_id}``   : détail + progression.
- ``DELETE /api/jobs/{job_id}``   : annulation explicite.

L'endpoint **POST /api/jobs** (création + lancement asynchrone) est
volontairement reporté à un sprint dédié de l'intégration runtime
— il nécessite un thread d'exécution branché sur ``RunOrchestrator``
(au-delà du périmètre S37 qui livre la persistance + les endpoints
de lecture).

Anti-sur-ingénierie
-------------------
- Pas de SSE / event stream (legacy ``job_events``) : reporté quand
  un caller en a besoin ; le polling sur ``progress`` suffit pour
  l'UI minimaliste S38.
- Pas de filtre par status/corpus : facile à ajouter quand un caller
  le demande.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/jobs", tags=["jobs"])


# ──────────────────────────────────────────────────────────────────────
# Schémas
# ──────────────────────────────────────────────────────────────────────


class JobSummary(BaseModel):
    """Résumé d'un job pour la liste."""

    job_id: str
    status: str
    progress: float
    current_engine: str
    total_docs: int
    processed_docs: int
    created_at: float
    updated_at: float
    finished_at: float | None = None


class JobListResponse(BaseModel):
    jobs: list[JobSummary]


class JobDetailResponse(BaseModel):
    """Détail complet d'un job — incluant payload + erreur."""

    job_id: str
    status: str
    progress: float
    current_engine: str
    total_docs: int
    processed_docs: int
    output_path: str
    error: str
    payload: dict = Field(default_factory=dict)
    created_at: float
    updated_at: float
    finished_at: float | None = None


class JobCancelResponse(BaseModel):
    job_id: str
    status: str


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _require_job_store(state) -> "object":
    if state.job_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Job store non configuré dans WebAppState — la persistance "
                "des jobs n'est pas activée."
            ),
        )
    return state.job_store


def _to_summary(rec) -> JobSummary:
    return JobSummary(
        job_id=rec.job_id,
        status=rec.status,
        progress=rec.progress,
        current_engine=rec.current_engine,
        total_docs=rec.total_docs,
        processed_docs=rec.processed_docs,
        created_at=rec.created_at,
        updated_at=rec.updated_at,
        finished_at=rec.finished_at,
    )


def _to_detail(rec) -> JobDetailResponse:
    return JobDetailResponse(
        job_id=rec.job_id,
        status=rec.status,
        progress=rec.progress,
        current_engine=rec.current_engine,
        total_docs=rec.total_docs,
        processed_docs=rec.processed_docs,
        output_path=rec.output_path,
        error=rec.error,
        payload=rec.payload,
        created_at=rec.created_at,
        updated_at=rec.updated_at,
        finished_at=rec.finished_at,
    )


# ──────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────


@router.get("", response_model=JobListResponse)
async def list_jobs(request: Request) -> JobListResponse:
    """Liste les jobs (récents en tête)."""
    state = request.app.state.picarones
    store = _require_job_store(state)
    return JobListResponse(
        jobs=[_to_summary(r) for r in store.list()],
    )


@router.get("/{job_id}", response_model=JobDetailResponse)
async def get_job(request: Request, job_id: str) -> JobDetailResponse:
    """Détail d'un job avec payload + progression."""
    state = request.app.state.picarones
    store = _require_job_store(state)
    rec = store.get(job_id)
    if rec is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id!r} introuvable.",
        )
    return _to_detail(rec)


@router.delete("/{job_id}", response_model=JobCancelResponse)
async def cancel_job(request: Request, job_id: str) -> JobCancelResponse:
    """Annule un job (uniquement s'il est encore vivant).

    Idempotent : annuler un job déjà terminal retourne le statut
    actuel sans erreur.
    """
    state = request.app.state.picarones
    store = _require_job_store(state)
    rec = store.get(job_id)
    if rec is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id!r} introuvable.",
        )
    if rec.is_terminal:
        # Idempotent : on retourne le statut actuel sans changer.
        return JobCancelResponse(job_id=rec.job_id, status=rec.status)

    store.mark_cancelled(job_id)
    updated = store.get(job_id)
    return JobCancelResponse(
        job_id=updated.job_id, status=updated.status,
    )


__all__ = ["router"]
