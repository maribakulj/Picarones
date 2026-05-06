"""Router jobs — Sprints A14-S37 + S48.

Endpoints de gestion des jobs de benchmark, adossés à
``JobStore`` (S37) + ``JobRunner`` (S48).

Endpoints
---------
- ``GET    /api/jobs``            : liste des jobs (récents en tête).
- ``GET    /api/jobs/{job_id}``   : détail + progression.
- ``POST   /api/jobs``            : création + lancement asynchrone.
- ``DELETE /api/jobs/{job_id}``   : annulation explicite.

S37 (initial) livrait les 3 premiers (lecture + cancellation).
S48 ajoute ``POST`` qui était identifié comme **manque critique**
dans l'audit du rewrite (l'audit #2).

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

from fastapi import APIRouter, Body, HTTPException, Request, status
from pydantic import BaseModel, Field

from picarones.app.schemas.run_spec import (
    RunSpecLoadError,
    load_run_spec_from_yaml,
)

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


class JobSubmitResponse(BaseModel):
    """Réponse JSON pour ``POST /api/jobs`` (202 Accepted)."""

    job_id: str
    status: str = Field(
        default="pending",
        description=(
            "Statut au moment de la soumission.  Le client poll "
            "``GET /api/jobs/{job_id}`` pour suivre la progression."
        ),
    )


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


def _require_job_runner(state) -> "object":
    if state.job_runner is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Job runner non configuré dans WebAppState — "
                "l'exécution asynchrone des jobs n'est pas activée. "
                "Voir picarones.app.services.JobRunner pour le câblage."
            ),
        )
    return state.job_runner


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


@router.post(
    "",
    response_model=JobSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_job(
    request: Request,
    run_spec_yaml: str = Body(
        ...,
        media_type="text/plain",
        description=(
            "Contenu YAML d'un ``RunSpec`` (cf. picarones.app.schemas."
            "run_spec).  Le corps de la requête est le YAML brut."
        ),
    ),
) -> JobSubmitResponse:
    """Crée un job + lance son exécution en arrière-plan (S48).

    Le corps de la requête est le YAML brut d'un ``RunSpec`` (mêmes
    champs que ce que la CLI ``picarones-rewrite run`` accepte).

    Comportement :

    1. Le YAML est parsé et validé (``load_run_spec_from_yaml``).
       Erreur de format → 400 avec message du loader.
    2. Un ``JobRecord`` est créé en statut ``pending`` avec un
       ``job_id`` UUID4.
    3. Un thread daemon est lancé pour exécuter le ``RunOrchestrator``
       avec le ``RunSpec``.
    4. Réponse immédiate ``202 Accepted`` avec ``job_id`` — le
       client poll ``GET /api/jobs/{job_id}`` pour suivre.

    Concurrence
    -----------
    Un thread par job ; pas de queue/backpressure.  Pour 100+ jobs
    simultanés, ajouter un ``ThreadPoolExecutor`` au niveau de
    ``JobRunner`` (post-livraison).
    """
    state = request.app.state.picarones
    runner = _require_job_runner(state)

    if not run_spec_yaml or not run_spec_yaml.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Corps de la requête vide — YAML RunSpec attendu.",
        )

    try:
        run_spec = load_run_spec_from_yaml(run_spec_yaml)
    except RunSpecLoadError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"RunSpec invalide : {exc}",
        ) from exc

    # Output dir : sous-dossier dédié au job dans le workspace.  Le
    # JobRunner s'en sert pour construire un RunOrchestrator isolé.
    import uuid
    job_id_candidate = uuid.uuid4().hex
    output_dir = (
        state.workspace.root / "runs" / job_id_candidate
    )

    try:
        job_id = runner.submit(
            run_spec=run_spec,
            output_dir=output_dir,
            job_id=job_id_candidate,
            payload={"corpus_name": run_spec.corpus_name or ""},
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "[jobs] échec de submit pour run_spec : %s", exc, exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Échec de soumission du job : {type(exc).__name__}",
        ) from exc

    return JobSubmitResponse(job_id=job_id, status="pending")


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
