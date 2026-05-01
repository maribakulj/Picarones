"""Interface web locale Picarones — application FastAPI.

Lance avec :
    picarones serve [--port 8000] [--host 127.0.0.1]
ou directement :
    uvicorn picarones.web.app:app --reload --port 8000

Routes
------
GET  /                          Page principale (SPA)
GET  /api/status                Version et état de l'application
GET  /api/engines               Statut des moteurs OCR et LLMs disponibles
GET  /api/corpus/browse         Parcourir les dossiers du serveur
GET  /api/reports               Liste des rapports générés
GET  /api/normalization/profiles Profils de normalisation disponibles
POST /api/benchmark/start       Lancer un benchmark (retourne job_id)
GET  /api/benchmark/{job_id}/stream  Stream SSE de progression
GET  /api/benchmark/{job_id}/status  Statut courant d'un job
POST /api/benchmark/{job_id}/cancel  Annuler un job
GET  /api/htr-united/catalogue  Catalogue HTR-United
POST /api/htr-united/import     Importer un corpus HTR-United
GET  /api/huggingface/search    Rechercher des datasets HuggingFace
POST /api/huggingface/import    Importer un dataset HuggingFace
GET  /reports/{filename}        Accéder à un rapport HTML généré
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from pathlib import Path
from typing import AsyncIterator, Optional

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from picarones import __version__
from picarones.web.benchmark_utils import (
    run_benchmark_thread as _run_benchmark_thread,
    run_benchmark_thread_v2 as _run_benchmark_thread_v2,
    sse_format as _sse_format,
)
from picarones.web.models import (
    BenchmarkRequest,
    BenchmarkRunRequest,
)
from picarones.web.security import (
    assert_engines_allowed,
    assert_llm_provider_allowed,
    csp_middleware,
    get_max_concurrent_jobs,
)
from picarones.web.routers import (
    config as _config_router,
    corpus as _corpus_router,
    engines as _engines_router,
    history as _history_router,
    home as _home_router,
    importers as _importers_router,
    normalization as _normalization_router,
    reports as _reports_router,
    synthesis as _synthesis_router,
    system as _system_router,
)
from picarones.web import state as _state
from picarones.web.state import (
    BenchmarkJob,
    cleanup_old_jobs as _cleanup_old_jobs,
    enforce_rate_limit as _enforce_rate_limit,
)

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Lifespan FastAPI — Sprint 26 : marque les jobs orphelins au boot.

    Au démarrage d'un nouveau processus, tous les jobs encore en statut
    ``pending`` ou ``running`` en base sont forcément orphelins (le
    processus précédent est mort sans les finir). On les bascule en
    ``interrupted`` une bonne fois pour toutes pour ne pas laisser
    d'état mensonger sur le tableau de bord.
    """
    try:
        _state.JOB_STORE.mark_orphaned_jobs_interrupted()
    except Exception as exc:  # pragma: no cover — défense en profondeur
        _logger.warning("[jobs] mark_orphaned_jobs_interrupted échoué : %s", exc)
    yield


app = FastAPI(
    title="Picarones",
    description="Plateforme de comparaison de moteurs OCR/HTR pour documents patrimoniaux",
    version=__version__,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=_lifespan,
)

# Sprint 24 — middleware CSP + en-têtes durcis (X-Frame-Options, etc.)
app.middleware("http")(csp_middleware)

# Fichiers statiques (CSS, icônes…)
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# Routers thématiques (extraits dans picarones.web.routers)
# ---------------------------------------------------------------------------

app.include_router(_system_router.router)
app.include_router(_engines_router.router)
app.include_router(_corpus_router.router)
app.include_router(_normalization_router.router)
app.include_router(_config_router.router)
app.include_router(_synthesis_router.router)
app.include_router(_history_router.router)
app.include_router(_reports_router.router)
app.include_router(_importers_router.router)
app.include_router(_home_router.router)


# ---------------------------------------------------------------------------
# API — benchmark
# ---------------------------------------------------------------------------

@app.post("/api/benchmark/start")
async def api_benchmark_start(req: BenchmarkRequest, request: Request) -> dict:
    corpus_path = Path(req.corpus_path)
    if not corpus_path.exists() or not corpus_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Corpus non trouvé : {req.corpus_path}")

    # Sprint 24 — mode public : refuse les moteurs OCR cloud mutualisés.
    try:
        assert_engines_allowed(req.engines)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))

    # Sprint 24 — rate limit + sémaphore concurrents.
    _enforce_rate_limit(request)
    if not _state.JOBS_SEMAPHORE.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail=(
                f"Trop de benchmarks concurrents (max "
                f"{get_max_concurrent_jobs()}). Réessayer plus tard."
            ),
        )

    job_id = str(uuid.uuid4())
    job = BenchmarkJob(job_id=job_id, _store=_state.JOB_STORE)
    _state.JOB_STORE.create_job(job_id)
    _state.JOBS[job_id] = job
    _cleanup_old_jobs()

    def _release_after(job_, fn, *args):
        try:
            fn(job_, *args)
        finally:
            _state.JOBS_SEMAPHORE.release()

    # Démarrer le benchmark dans un thread séparé
    thread = threading.Thread(
        target=_release_after,
        args=(job, _run_benchmark_thread, req),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "status": "pending"}


@app.get("/api/benchmark/{job_id}/status")
async def api_benchmark_status(job_id: str) -> dict:
    job = _state.JOBS.get(job_id)
    if job is not None:
        return job.as_dict()
    # Sprint 26 — fallback DB : le job n'est pas (plus) en RAM dans ce
    # worker mais peut exister en base (autre worker, ou redémarrage).
    db_job = _state.JOB_STORE.get_job(job_id)
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


@app.post("/api/benchmark/{job_id}/cancel")
async def api_benchmark_cancel(job_id: str) -> dict:
    job = _state.JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job non trouvé : {job_id}")
    if job.status in ("complete", "error"):
        return {"job_id": job_id, "status": job.status, "message": "Job déjà terminé."}
    job.set_status("cancelled")
    job._cancel_event.set()  # Signal d'annulation pour run_benchmark
    job.add_event("cancelled", {"message": "Benchmark annulé par l'utilisateur."})
    return {"job_id": job_id, "status": "cancelled"}


@app.get("/api/benchmark/{job_id}/stream")
async def api_benchmark_stream(job_id: str, request: Request) -> StreamingResponse:
    """SSE de progression d'un benchmark.

    Sprint 26 — supporte la reprise via le header standard
    ``Last-Event-ID`` (clamped à un ``int``) : le client envoie le dernier
    ``seq`` reçu, le serveur rejoue tous les événements ``> seq`` puis
    bascule sur le live. Si le job est terminé (ou orphelin/interrompu),
    on envoie le backlog puis ``done`` et on ferme la connexion.

    Trois cas :
      1. Job en RAM ET vivant ⇒ subscribe + backlog DB depuis last_seq.
      2. Job en RAM mais terminé ⇒ backlog DB + done.
      3. Job uniquement en DB (orphelin, autre worker) ⇒ backlog DB + done.
    """
    last_event_id = request.headers.get("last-event-id", "0").strip()
    try:
        last_seq = max(0, int(last_event_id))
    except ValueError:
        last_seq = 0

    job = _state.JOBS.get(job_id)
    db_job = _state.JOB_STORE.get_job(job_id)
    if job is None and db_job is None:
        raise HTTPException(status_code=404, detail=f"Job non trouvé : {job_id}")

    async def event_generator() -> AsyncIterator[str]:
        queue: Optional[asyncio.Queue] = None
        if job is not None:
            queue = job.subscribe()
        try:
            # 1) Backlog depuis la base — l'autorité de vérité (Sprint 26).
            backlog = _state.JOB_STORE.get_events_after(job_id, last_seq=last_seq)
            seen_seqs: set[int] = set()
            for ev in backlog:
                seen_seqs.add(ev["seq"])
                yield _sse_format(ev["kind"], ev["data"], seq=ev["seq"])

            # Statut courant : RAM si dispo, sinon DB.
            current_status = job.status if job is not None else (db_job or {}).get("status", "")
            if current_status in ("complete", "error", "cancelled", "interrupted"):
                yield _sse_format("done", {"status": current_status})
                return

            if queue is None:
                # Pas de live possible (job pas en RAM dans ce worker) — on
                # ne peut pas suivre la progression future. Au pire le
                # client se reconnecte avec le nouveau ``Last-Event-ID``.
                yield _sse_format("done", {"status": current_status or "unknown"})
                return

            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    seq = event.get("seq")
                    if seq is not None and seq in seen_seqs:
                        # Déjà délivré dans le backlog — éviter le doublon.
                        continue
                    yield _sse_format(event["kind"], event["data"], seq=seq)
                    if event["kind"] in ("complete", "error", "cancelled", "done"):
                        break
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    if job.status in ("complete", "error", "cancelled", "interrupted"):
                        yield _sse_format("done", {"status": job.status})
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


# ---------------------------------------------------------------------------
# API — benchmark/run (concurrents composés)
# ---------------------------------------------------------------------------

@app.post("/api/benchmark/run")
async def api_benchmark_run(req: BenchmarkRunRequest, request: Request) -> dict:
    corpus_path = Path(req.corpus_path)
    if not corpus_path.exists() or not corpus_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Corpus non trouvé : {req.corpus_path}")
    if not req.competitors:
        raise HTTPException(status_code=400, detail="Aucun concurrent défini.")

    # Sprint 24 — mode public : refuse les pipelines LLM mutualisés et
    # les moteurs OCR cloud sollicités par n'importe quel concurrent.
    try:
        for comp in req.competitors:
            assert_engines_allowed([comp.ocr_engine] if comp.ocr_engine else [])
            assert_llm_provider_allowed(comp.llm_provider)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))

    # Sprint 24 — rate limit + sémaphore concurrents.
    _enforce_rate_limit(request)
    if not _state.JOBS_SEMAPHORE.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail=(
                f"Trop de benchmarks concurrents (max "
                f"{get_max_concurrent_jobs()}). Réessayer plus tard."
            ),
        )

    job_id = str(uuid.uuid4())
    job = BenchmarkJob(job_id=job_id, _store=_state.JOB_STORE)
    _state.JOB_STORE.create_job(job_id)
    _state.JOBS[job_id] = job

    def _release_after(job_, fn, *args):
        try:
            fn(job_, *args)
        finally:
            _state.JOBS_SEMAPHORE.release()

    thread = threading.Thread(
        target=_release_after,
        args=(job, _run_benchmark_thread_v2, req),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "pending"}

