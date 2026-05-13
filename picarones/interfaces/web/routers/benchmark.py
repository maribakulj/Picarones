"""Router des endpoints de benchmark : start, status, cancel, stream, run.

Le ``stream`` SSE supporte la reprise via ``Last-Event-ID`` (Sprint 26).
``start`` lance un benchmark à liste de moteurs ; ``run`` accepte des
``PipelineConfig`` composés (OCR + LLM, pipelines mutualisés) —
deux endpoints distincts pour deux UX historiquement séparées.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from typing import AsyncIterator, Callable, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from picarones.interfaces.web import state
from picarones.interfaces.web.benchmark_utils import (
    run_benchmark_thread,
    run_benchmark_thread_v2,
    sse_format,
)
from picarones.interfaces.web.models import BenchmarkRequest, BenchmarkRunRequest
from picarones.interfaces.web.security import (
    PathValidationError,
    assert_engines_allowed,
    assert_llm_provider_allowed,
    compute_workspace_roots,
    get_max_concurrent_jobs,
    validated_path,
    validated_prompt_filename,
)
from picarones.interfaces.web.state import UPLOADS_DIR

router = APIRouter()


def _start_job_thread(
    job: state.BenchmarkJob,
    worker: Callable[..., None],
    req,
) -> None:
    """Démarre ``worker`` dans un thread daemon en libérant le sémaphore à la fin.

    Helper partagé par les deux endpoints qui lancent un benchmark
    (``/api/benchmark/start`` et ``/api/benchmark/run``). Garantit
    que ``JOBS_SEMAPHORE`` est libéré, succès ou échec, sans avoir
    à dupliquer le ``try/finally`` au site d'appel.
    """
    def _release_after_worker():
        try:
            worker(job, req)
        finally:
            state.JOBS_SEMAPHORE.release()

    threading.Thread(target=_release_after_worker, daemon=True).start()


# ──────────────────────────────────────────────────────────────────────────
# Lancement legacy : liste de moteurs (BenchmarkRequest)
# ──────────────────────────────────────────────────────────────────────────

@router.post("/api/benchmark/start")
async def api_benchmark_start(req: BenchmarkRequest, request: Request) -> dict:
    """Lance un benchmark sur une liste de moteurs OCR (mode legacy)."""
    # Sprint 24 — mode public : refuse les moteurs OCR cloud mutualisés.
    # Vérifié AVANT la validation des chemins pour que la réponse
    # 403 mode public reste prioritaire (cf. tests sprint24).
    try:
        assert_engines_allowed(req.engines)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))

    # Sprint A14-S1 — A.I.0 P0 : validation des chemins utilisateur
    # contre les racines workspace autorisées.  Bloque les chemins
    # absolus arbitraires, la traversée (``..``), les liens symboliques
    # vers l'extérieur, etc.
    workspace_roots = compute_workspace_roots(UPLOADS_DIR)
    try:
        validated_path(
            req.corpus_path,
            allowed_roots=workspace_roots,
            must_be_dir=True,
        )
        # ``output_dir`` peut ne pas encore exister, on valide juste
        # qu'il sera créé dans une racine autorisée.
        validated_path(
            req.output_dir,
            allowed_roots=workspace_roots,
            must_exist=False,
        )
    except PathValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Sprint 24 — rate limit + sémaphore concurrents.
    state.enforce_rate_limit(request)
    if not state.JOBS_SEMAPHORE.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail=(
                f"Trop de benchmarks concurrents (max "
                f"{get_max_concurrent_jobs()}). Réessayer plus tard."
            ),
        )

    job_id = str(uuid.uuid4())
    job = state.BenchmarkJob(job_id=job_id, _store=state.JOB_STORE)
    # Phase 4 du chantier post-rewrite : le payload du job contient
    # désormais le ``corpus_path`` actif, pour que la tâche de purge
    # ``upload_purge_task`` puisse identifier les corpus référencés
    # par des jobs en cours et ne pas les supprimer.  Avant ce
    # branchement, la purge supprimait potentiellement des corpus
    # actifs dont les uploads étaient plus anciens que la rétention.
    state.JOB_STORE.create_job(job_id, payload={"corpus": req.corpus_path})
    state.register_job(job)
    state.cleanup_old_jobs()

    _start_job_thread(job, run_benchmark_thread, req)
    return {"job_id": job_id, "status": "pending"}


# ──────────────────────────────────────────────────────────────────────────
# Lancement composé : liste de PipelineConfig (BenchmarkRunRequest)
# ──────────────────────────────────────────────────────────────────────────

@router.post("/api/benchmark/run")
async def api_benchmark_run(req: BenchmarkRunRequest, request: Request) -> dict:
    """Lance un benchmark à concurrents composés (OCR + LLM, pipelines).

    Chaque ``PipelineConfig`` peut combiner un moteur OCR et un
    provider LLM (mode post-correction, zero-shot, ou OCR seul).
    """
    # ``competitors`` non vide est garanti par Pydantic ``min_length=1``.

    # Mode public : refuse les pipelines LLM mutualisés et les moteurs
    # OCR cloud sollicités par n'importe quel concurrent.
    # Vérifié AVANT la validation des chemins (cf. /api/benchmark/start
    # pour le rationale).
    try:
        for comp in req.competitors:
            assert_engines_allowed([comp.engine_name] if comp.engine_name else [])
            assert_llm_provider_allowed(comp.llm_provider)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))

    # Sprint A14-S1 — A.I.0 P0 : validation des chemins utilisateur
    # (cf. /api/benchmark/start).  Idempotent : refuse un corpus_path
    # absolu hors workspaces, et refuse un output_dir qui s'évaderait
    # via ``..`` ou symlinks.
    workspace_roots = compute_workspace_roots(UPLOADS_DIR)
    try:
        validated_path(
            req.corpus_path,
            allowed_roots=workspace_roots,
            must_be_dir=True,
        )
        validated_path(
            req.output_dir,
            allowed_roots=workspace_roots,
            must_exist=False,
        )
        # Sprint A14-S1 — restriction des prompts à la bibliothèque
        # intégrée (``picarones/prompts/``).  Cf. validated_prompt_filename
        # pour le rationale (vecteur d'exfiltration via LLM).
        for comp in req.competitors:
            if comp.prompt_file:
                validated_prompt_filename(comp.prompt_file)
    except PathValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Sprint 24 — rate limit + sémaphore concurrents.
    state.enforce_rate_limit(request)
    if not state.JOBS_SEMAPHORE.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail=(
                f"Trop de benchmarks concurrents (max "
                f"{get_max_concurrent_jobs()}). Réessayer plus tard."
            ),
        )

    job_id = str(uuid.uuid4())
    job = state.BenchmarkJob(job_id=job_id, _store=state.JOB_STORE)
    # Phase 4 — payload incluant le corpus actif pour la purge auto.
    state.JOB_STORE.create_job(job_id, payload={"corpus": req.corpus_path})
    state.register_job(job)

    _start_job_thread(job, run_benchmark_thread_v2, req)
    return {"job_id": job_id, "status": "pending"}


# ──────────────────────────────────────────────────────────────────────────
# Statut + annulation
# ──────────────────────────────────────────────────────────────────────────

@router.get("/api/benchmark/{job_id}/status")
async def api_benchmark_status(job_id: str) -> dict:
    """Statut courant d'un job (RAM si disponible, sinon DB)."""
    job = state.get_job_in_memory(job_id)
    if job is not None:
        return job.as_dict()
    # Sprint 26 — fallback DB : le job n'est pas (plus) en RAM dans ce
    # worker mais peut exister en base (autre worker, ou redémarrage).
    db_job = state.JOB_STORE.get_job(job_id)
    if db_job is None:
        raise HTTPException(status_code=404, detail=f"Job non trouvé : {job_id}")
    return {
        "job_id": db_job["job_id"],
        "status": db_job["status"],
        "progress": db_job["progress"],
        "current_engine": db_job["current_engine"],
        "total_docs": db_job["total_docs"],
        "processed_docs": db_job["processed_docs"],
        "output_path": db_job["output_path"],
        "error": db_job["error"],
        "started_at": None,
        "finished_at": db_job["finished_at"],
    }


@router.post("/api/benchmark/{job_id}/cancel")
async def api_benchmark_cancel(job_id: str) -> dict:
    """Annule un job en cours (no-op si déjà terminé)."""
    job = state.get_job_in_memory(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job non trouvé : {job_id}")
    if job.status in ("complete", "error"):
        return {
            "job_id": job_id,
            "status": job.status,
            "message": "Job déjà terminé.",
        }
    job.set_status("cancelled")
    job._cancel_event.set()  # Signal d'annulation pour run_benchmark
    job.add_event("cancelled", {"message": "Benchmark annulé par l'utilisateur."})
    return {"job_id": job_id, "status": "cancelled"}


# ──────────────────────────────────────────────────────────────────────────
# SSE de progression (avec reprise via Last-Event-ID)
# ──────────────────────────────────────────────────────────────────────────

@router.get("/api/benchmark/{job_id}/stream")
async def api_benchmark_stream(job_id: str, request: Request) -> StreamingResponse:
    """SSE de progression d'un benchmark.

    Sprint 26 — supporte la reprise via le header standard
    ``Last-Event-ID`` (clamped à un ``int``) : le client envoie le
    dernier ``seq`` reçu, le serveur rejoue tous les événements
    ``> seq`` puis bascule sur le live. Si le job est terminé (ou
    orphelin/interrompu), on envoie le backlog puis ``done`` et on
    ferme la connexion.

    Trois cas :

    1. Job en RAM ET vivant ⇒ subscribe + backlog DB depuis ``last_seq``.
    2. Job en RAM mais terminé ⇒ backlog DB + ``done``.
    3. Job uniquement en DB (orphelin, autre worker) ⇒ backlog DB + ``done``.
    """
    last_event_id = request.headers.get("last-event-id", "0").strip()
    try:
        last_seq = max(0, int(last_event_id))
    except ValueError:
        last_seq = 0

    job = state.get_job_in_memory(job_id)
    db_job = state.JOB_STORE.get_job(job_id)
    if job is None and db_job is None:
        raise HTTPException(status_code=404, detail=f"Job non trouvé : {job_id}")

    async def event_generator() -> AsyncIterator[str]:
        queue: Optional[asyncio.Queue] = None
        if job is not None:
            queue = job.subscribe()
        try:
            # 1) Backlog depuis la base — l'autorité de vérité (Sprint 26).
            backlog = state.JOB_STORE.get_events_after(job_id, last_seq=last_seq)
            seen_seqs: set[int] = set()
            for ev in backlog:
                seen_seqs.add(ev["seq"])
                yield sse_format(ev["kind"], ev["data"], seq=ev["seq"])

            current_status = (
                job.status if job is not None
                else (db_job or {}).get("status", "")
            )
            if current_status in ("complete", "error", "cancelled", "interrupted"):
                yield sse_format("done", {"status": current_status})
                return

            if queue is None or job is None:
                # Pas de live possible (job pas en RAM dans ce worker) — on
                # ne peut pas suivre la progression future. Au pire le
                # client se reconnecte avec le nouveau ``Last-Event-ID``.
                yield sse_format("done", {"status": current_status or "unknown"})
                return

            # Sprint S3.1 — narrowing mypy explicite : à ce point,
            # ``queue`` et ``job`` sont garantis non-``None`` (par
            # construction lignes 268-271).  Les ``assert`` ne sont
            # pas pour la défense (impossible) mais pour le type
            # checker — sans eux, ``job.status`` ligne ~315 échoue
            # ``mypy --strict``.
            assert job is not None
            assert queue is not None

            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    seq = event.get("seq")
                    if seq is not None and seq in seen_seqs:
                        # Déjà délivré dans le backlog — éviter le doublon.
                        continue
                    yield sse_format(event["kind"], event["data"], seq=seq)
                    if event["kind"] in ("complete", "error", "cancelled", "done"):
                        break
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    if job.status in ("complete", "error", "cancelled", "interrupted"):
                        yield sse_format("done", {"status": job.status})
                        break
        finally:
            if queue is not None and job is not None:
                job.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
