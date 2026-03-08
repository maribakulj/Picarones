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
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from fastapi import Cookie, FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from picarones import __version__

# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Picarones",
    description="Plateforme de comparaison de moteurs OCR/HTR pour documents patrimoniaux",
    version=__version__,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# ---------------------------------------------------------------------------
# Job management
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkJob:
    job_id: str
    status: str = "pending"   # pending | running | complete | error | cancelled
    progress: float = 0.0     # 0.0 – 1.0
    current_engine: str = ""
    total_docs: int = 0
    processed_docs: int = 0
    output_path: str = ""
    error: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    events: list[dict] = field(default_factory=list)
    _subscribers: list[asyncio.Queue] = field(default_factory=list)

    def add_event(self, kind: str, data: Any) -> None:
        event = {"kind": kind, "data": data, "ts": _iso_now()}
        self.events.append(event)
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
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


_JOBS: dict[str, BenchmarkJob] = {}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class BenchmarkRequest(BaseModel):
    corpus_path: str
    engines: list[str] = ["tesseract"]
    normalization_profile: str = "nfc"
    output_dir: str = "./rapports/"
    report_name: str = ""
    lang: str = "fra"
    report_lang: str = "fr"   # langue du rapport HTML : "fr" ou "en"

class HTRUnitedImportRequest(BaseModel):
    entry_id: str
    output_dir: str = "./corpus/"
    max_samples: int = 100

class HuggingFaceImportRequest(BaseModel):
    dataset_id: str
    output_dir: str = "./corpus/"
    split: str = "train"
    max_samples: int = 100


# ---------------------------------------------------------------------------
# API — status
# ---------------------------------------------------------------------------

@app.get("/api/status")
async def api_status() -> dict:
    return {
        "app": "Picarones",
        "version": __version__,
        "status": "ok",
        "timestamp": _iso_now(),
    }


# ---------------------------------------------------------------------------
# API — langue / i18n
# ---------------------------------------------------------------------------

_SUPPORTED_LANGS = ("fr", "en")
_LANG_COOKIE = "picarones_lang"


@app.get("/api/lang")
async def api_get_lang(
    picarones_lang: str = Cookie(default="fr"),
) -> dict:
    """Retourne la langue courante de l'interface (lue depuis le cookie de session)."""
    lang = picarones_lang if picarones_lang in _SUPPORTED_LANGS else "fr"
    return {"lang": lang, "supported": list(_SUPPORTED_LANGS)}


@app.post("/api/lang/{lang_code}")
async def api_set_lang(lang_code: str, response: Response) -> dict:
    """Définit la langue de l'interface et la persiste dans un cookie de session.

    Langues supportées : ``fr`` (français), ``en`` (anglais patrimonial).
    """
    if lang_code not in _SUPPORTED_LANGS:
        raise HTTPException(
            status_code=400,
            detail=f"Langue non supportée : '{lang_code}'. Disponibles : {', '.join(_SUPPORTED_LANGS)}",
        )
    response.set_cookie(
        key=_LANG_COOKIE,
        value=lang_code,
        max_age=60 * 60 * 24 * 365,  # 1 an
        httponly=False,
        samesite="lax",
    )
    return {"lang": lang_code, "message": f"Langue définie : {lang_code}"}


# ---------------------------------------------------------------------------
# API — engines
# ---------------------------------------------------------------------------

@app.get("/api/engines")
async def api_engines() -> dict:
    engines = []

    # Tesseract
    tess = _check_engine("tesseract", "pytesseract")
    engines.append(tess)

    # Pero OCR
    pero = _check_engine("pero_ocr", "pero_ocr", label="Pero OCR")
    engines.append(pero)

    # Kraken
    kraken = _check_engine("kraken", "kraken", label="Kraken")
    engines.append(kraken)

    # Calamari
    calamari = _check_engine("calamari", "calamari_ocr", label="Calamari")
    engines.append(calamari)

    llms = []

    # OpenAI
    llms.append({
        "id": "openai",
        "label": "OpenAI (GPT-4o, GPT-4o mini)",
        "type": "llm",
        "available": bool(os.environ.get("OPENAI_API_KEY")),
        "key_env": "OPENAI_API_KEY",
        "status": "configured" if os.environ.get("OPENAI_API_KEY") else "missing_key",
    })

    # Anthropic
    llms.append({
        "id": "anthropic",
        "label": "Anthropic (Claude Sonnet, Haiku)",
        "type": "llm",
        "available": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "key_env": "ANTHROPIC_API_KEY",
        "status": "configured" if os.environ.get("ANTHROPIC_API_KEY") else "missing_key",
    })

    # Mistral
    llms.append({
        "id": "mistral",
        "label": "Mistral (Mistral OCR, Pixtral, Large)",
        "type": "llm",
        "available": bool(os.environ.get("MISTRAL_API_KEY")),
        "key_env": "MISTRAL_API_KEY",
        "status": "configured" if os.environ.get("MISTRAL_API_KEY") else "missing_key",
    })

    # Ollama
    ollama_available = _check_ollama()
    ollama_models = _list_ollama_models() if ollama_available else []
    llms.append({
        "id": "ollama",
        "label": "Ollama (Llama 3, Gemma, Phi — local)",
        "type": "llm_local",
        "available": ollama_available,
        "status": "running" if ollama_available else "not_running",
        "models": ollama_models,
        "base_url": "http://localhost:11434",
    })

    return {"engines": engines, "llms": llms}


def _check_engine(engine_id: str, module_name: str, label: str = "") -> dict:
    label = label or engine_id.replace("_", " ").title()
    try:
        __import__(module_name)
        installed = True
    except ImportError:
        installed = False

    version = ""
    if installed and engine_id == "tesseract":
        try:
            import pytesseract
            version = pytesseract.get_tesseract_version()
            version = str(version)
        except Exception:
            version = "installé"
    elif installed:
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "installé")
        except Exception:
            version = "installé"

    return {
        "id": engine_id,
        "label": label,
        "type": "ocr",
        "available": installed,
        "version": version,
        "status": "available" if installed else "not_installed",
    }


def _check_ollama() -> bool:
    import urllib.error, urllib.request
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _list_ollama_models() -> list[str]:
    import urllib.error, urllib.request
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as r:
            data = json.loads(r.read().decode())
        return [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# API — corpus browse
# ---------------------------------------------------------------------------

@app.get("/api/corpus/browse")
async def api_corpus_browse(path: str = Query(default=".", description="Chemin à explorer")) -> dict:
    target = Path(path).resolve()
    if not target.exists() or not target.is_dir():
        raise HTTPException(status_code=404, detail=f"Dossier non trouvé : {path}")

    items = []
    try:
        for entry in sorted(target.iterdir()):
            item: dict[str, Any] = {
                "name": entry.name,
                "path": str(entry),
                "is_dir": entry.is_dir(),
            }
            if entry.is_dir():
                # Compter les paires image/gt
                gt_count = sum(1 for f in entry.iterdir() if f.suffix == ".txt" and f.stem.endswith(".gt"))
                item["gt_count"] = gt_count
                item["has_corpus"] = gt_count > 0
            items.append(item)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))

    return {
        "current_path": str(target),
        "parent_path": str(target.parent) if target.parent != target else None,
        "items": items,
    }


# ---------------------------------------------------------------------------
# API — normalization profiles
# ---------------------------------------------------------------------------

@app.get("/api/normalization/profiles")
async def api_normalization_profiles() -> dict:
    from picarones.core.normalization import get_builtin_profile

    profile_ids = [
        "nfc",
        "caseless",
        "minimal",
        "medieval_french",
        "early_modern_french",
        "medieval_latin",
    ]

    profiles = []
    for pid in profile_ids:
        try:
            p = get_builtin_profile(pid)
            profiles.append({
                "id": pid,
                "name": p.name,
                "description": p.description or p.name,
                "caseless": p.caseless,
                "diplomatic_rules": len(p.diplomatic_table),
            })
        except Exception:
            pass

    return {"profiles": profiles}


# ---------------------------------------------------------------------------
# API — reports
# ---------------------------------------------------------------------------

@app.get("/api/reports")
async def api_reports(reports_dir: str = Query(default=".", description="Dossier rapports")) -> dict:
    target = Path(reports_dir).resolve()
    reports = []

    search_dirs = [target, Path(".").resolve(), Path("./rapports").resolve()]
    seen: set[str] = set()

    for d in search_dirs:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.html"), key=lambda x: x.stat().st_mtime, reverse=True):
            if str(f) not in seen:
                seen.add(str(f))
                stat = f.stat()
                reports.append({
                    "filename": f.name,
                    "path": str(f),
                    "size_kb": round(stat.st_size / 1024, 1),
                    "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                    "url": f"/reports/{f.name}",
                })

    return {"reports": reports}


@app.get("/reports/{filename}")
async def serve_report(filename: str) -> FileResponse:
    # Cherche dans le répertoire courant et ./rapports/
    for d in [Path("."), Path("./rapports")]:
        f = d / filename
        if f.exists() and f.suffix == ".html":
            return FileResponse(str(f.resolve()), media_type="text/html")
    raise HTTPException(status_code=404, detail=f"Rapport non trouvé : {filename}")


# ---------------------------------------------------------------------------
# API — HTR-United
# ---------------------------------------------------------------------------

@app.get("/api/htr-united/catalogue")
async def api_htr_united_catalogue(
    query: str = Query(default="", description="Recherche textuelle"),
    language: str = Query(default="", description="Filtre langue"),
    script: str = Query(default="", description="Filtre type d'écriture"),
) -> dict:
    from picarones.importers.htr_united import HTRUnitedCatalogue

    cat = HTRUnitedCatalogue.from_demo()
    results = cat.search(
        query=query,
        language=language or None,
        script=script or None,
    )
    return {
        "source": cat.source,
        "total": len(results),
        "entries": [e.as_dict() for e in results],
        "available_languages": cat.available_languages(),
        "available_scripts": cat.available_scripts(),
    }


@app.post("/api/htr-united/import")
async def api_htr_united_import(req: HTRUnitedImportRequest) -> dict:
    from picarones.importers.htr_united import HTRUnitedCatalogue, import_htr_united_corpus

    cat = HTRUnitedCatalogue.from_demo()
    entry = cat.get_by_id(req.entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Entrée non trouvée : {req.entry_id}")

    result = import_htr_united_corpus(
        entry=entry,
        output_dir=req.output_dir,
        max_samples=req.max_samples,
    )
    return result


# ---------------------------------------------------------------------------
# API — HuggingFace
# ---------------------------------------------------------------------------

@app.get("/api/huggingface/search")
async def api_huggingface_search(
    query: str = Query(default="", description="Requête de recherche"),
    language: str = Query(default="", description="Filtre langue"),
    tags: str = Query(default="", description="Tags séparés par des virgules"),
    limit: int = Query(default=20, ge=1, le=50),
) -> dict:
    from picarones.importers.huggingface import HuggingFaceImporter

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    importer = HuggingFaceImporter()
    results = importer.search(
        query=query,
        tags=tag_list,
        language=language or None,
        limit=limit,
    )
    return {
        "total": len(results),
        "datasets": [ds.as_dict() for ds in results],
    }


@app.post("/api/huggingface/import")
async def api_huggingface_import(req: HuggingFaceImportRequest) -> dict:
    from picarones.importers.huggingface import HuggingFaceImporter

    importer = HuggingFaceImporter()
    result = importer.import_dataset(
        dataset_id=req.dataset_id,
        output_dir=req.output_dir,
        split=req.split,
        max_samples=req.max_samples,
    )
    return result


# ---------------------------------------------------------------------------
# API — benchmark
# ---------------------------------------------------------------------------

@app.post("/api/benchmark/start")
async def api_benchmark_start(req: BenchmarkRequest) -> dict:
    corpus_path = Path(req.corpus_path)
    if not corpus_path.exists() or not corpus_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Corpus non trouvé : {req.corpus_path}")

    job_id = str(uuid.uuid4())
    job = BenchmarkJob(job_id=job_id)
    _JOBS[job_id] = job

    # Démarrer le benchmark dans un thread séparé
    thread = threading.Thread(
        target=_run_benchmark_thread,
        args=(job, req),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "status": "pending"}


@app.get("/api/benchmark/{job_id}/status")
async def api_benchmark_status(job_id: str) -> dict:
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job non trouvé : {job_id}")
    return job.as_dict()


@app.post("/api/benchmark/{job_id}/cancel")
async def api_benchmark_cancel(job_id: str) -> dict:
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job non trouvé : {job_id}")
    if job.status in ("complete", "error"):
        return {"job_id": job_id, "status": job.status, "message": "Job déjà terminé."}
    job.status = "cancelled"
    job.add_event("cancelled", {"message": "Benchmark annulé par l'utilisateur."})
    return {"job_id": job_id, "status": "cancelled"}


@app.get("/api/benchmark/{job_id}/stream")
async def api_benchmark_stream(job_id: str) -> StreamingResponse:
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job non trouvé : {job_id}")

    async def event_generator() -> AsyncIterator[str]:
        # Envoie d'abord les événements déjà produits
        for event in list(job.events):
            yield _sse_format(event["kind"], event["data"])

        if job.status in ("complete", "error", "cancelled"):
            yield _sse_format("done", {"status": job.status})
            return

        queue = job.subscribe()
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield _sse_format(event["kind"], event["data"])
                    if event["kind"] in ("complete", "error", "cancelled", "done"):
                        break
                except asyncio.TimeoutError:
                    # Keepalive
                    yield ": keepalive\n\n"
                    if job.status in ("complete", "error", "cancelled"):
                        yield _sse_format("done", {"status": job.status})
                        break
        finally:
            job.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _sse_format(event_type: str, data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event_type}\ndata: {payload}\n\n"


def _run_benchmark_thread(job: BenchmarkJob, req: BenchmarkRequest) -> None:
    """Exécute le benchmark dans un thread et envoie des événements SSE."""
    import time

    job.status = "running"
    job.started_at = _iso_now()
    job.add_event("start", {"message": "Démarrage du benchmark…", "corpus": req.corpus_path})

    try:
        from picarones.core.corpus import load_corpus_from_directory
        from picarones.core.runner import run_benchmark

        # Charger le corpus
        job.add_event("log", {"message": f"Chargement du corpus : {req.corpus_path}"})
        corpus = load_corpus_from_directory(req.corpus_path)
        job.total_docs = len(corpus)
        job.add_event("log", {"message": f"{job.total_docs} documents chargés."})

        if job.status == "cancelled":
            return

        # Instancier les moteurs
        from picarones.cli import _engine_from_name
        import click

        ocr_engines = []
        for engine_name in req.engines:
            try:
                eng = _engine_from_name(engine_name, lang=req.lang, psm=6)
                ocr_engines.append(eng)
                job.add_event("log", {"message": f"Moteur chargé : {engine_name}"})
            except (click.BadParameter, Exception) as exc:
                job.add_event("warning", {"message": f"Moteur ignoré '{engine_name}' : {exc}"})

        if not ocr_engines:
            raise ValueError("Aucun moteur valide disponible.")

        # Répertoire de sortie
        output_dir = Path(req.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_name = req.report_name or f"rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_json = str(output_dir / f"{report_name}.json")
        output_html = str(output_dir / f"{report_name}.html")

        # Callback de progression (injecté dans un wrapper)
        n_engines = len(ocr_engines)
        total_steps = job.total_docs * n_engines

        step_counter = [0]

        original_engine_names = [e.name for e in ocr_engines]

        def _progress_callback(engine_name: str, doc_idx: int, doc_id: str) -> None:
            if job.status == "cancelled":
                return
            step_counter[0] += 1
            job.current_engine = engine_name
            job.processed_docs = doc_idx
            job.progress = step_counter[0] / max(total_steps, 1)
            job.add_event("progress", {
                "engine": engine_name,
                "doc_idx": doc_idx,
                "doc_id": doc_id,
                "progress": job.progress,
                "processed": step_counter[0],
                "total": total_steps,
            })

        # Lancer le benchmark
        result = run_benchmark(
            corpus=corpus,
            engines=ocr_engines,
            output_json=output_json,
            show_progress=False,
            progress_callback=_progress_callback,
        )

        if job.status == "cancelled":
            return

        # Générer le rapport HTML
        job.add_event("log", {"message": "Génération du rapport HTML…"})
        from picarones.report.generator import ReportGenerator
        report_lang = getattr(req, "report_lang", "fr")
        gen = ReportGenerator(result, lang=report_lang)
        gen.generate(output_html)

        job.output_path = output_html
        job.progress = 1.0
        job.status = "complete"
        job.finished_at = _iso_now()

        # Classement final
        ranking = result.ranking()
        job.add_event("complete", {
            "message": "Benchmark terminé.",
            "output_html": output_html,
            "output_json": output_json,
            "ranking": ranking,
        })

    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
        job.finished_at = _iso_now()
        job.add_event("error", {"message": f"Erreur : {exc}"})


# ---------------------------------------------------------------------------
# Page principale HTML (SPA)
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(picarones_lang: str = Cookie(default="fr")) -> HTMLResponse:
    lang = picarones_lang if picarones_lang in _SUPPORTED_LANGS else "fr"
    # Injecte le code langue dans la SPA via une balise meta
    page = _HTML_TEMPLATE.replace(
        "<head>",
        f'<head>\n<meta name="picarones-lang" content="{lang}">',
        1,
    )
    return HTMLResponse(content=page)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# HTML Template (SPA, French/English, Vanilla JS)
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Picarones — OCR Benchmark</title>
<style>
:root {
  --bg: #f8f7f4;
  --bg2: #ffffff;
  --border: #d8d5ce;
  --accent: #2d5a9e;
  --accent-hover: #1e4080;
  --success: #2a7a3b;
  --warning: #c17b00;
  --danger: #c0392b;
  --text: #2c2c2c;
  --text-muted: #6b6b6b;
  --radius: 6px;
  --shadow: 0 1px 4px rgba(0,0,0,0.1);
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); font-size: 14px; line-height: 1.5; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

/* Layout */
#header { background: var(--accent); color: #fff; padding: 0 24px; display: flex; align-items: center; height: 52px; gap: 24px; position: sticky; top: 0; z-index: 100; }
#header h1 { font-size: 18px; font-weight: 600; letter-spacing: -0.3px; }
#header span.version { font-size: 11px; opacity: 0.7; margin-left: 4px; }
#nav { display: flex; gap: 4px; margin-left: auto; }
.nav-btn { background: transparent; border: 1px solid rgba(255,255,255,0.3); color: #fff; padding: 5px 12px; border-radius: var(--radius); cursor: pointer; font-size: 13px; transition: background 0.15s; }
.nav-btn:hover, .nav-btn.active { background: rgba(255,255,255,0.18); }
#lang-btn { margin-left: 12px; font-size: 12px; background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.3); color: #fff; padding: 4px 10px; border-radius: var(--radius); cursor: pointer; }

#main { max-width: 1100px; margin: 0 auto; padding: 24px 16px; }
.view { display: none; }
.view.active { display: block; }

/* Cards */
.card { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; margin-bottom: 16px; box-shadow: var(--shadow); }
.card h2 { font-size: 15px; font-weight: 600; margin-bottom: 14px; padding-bottom: 8px; border-bottom: 1px solid var(--border); color: var(--accent); }
.card h3 { font-size: 13px; font-weight: 600; margin-bottom: 10px; color: var(--text); }

/* Forms */
.form-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; align-items: flex-start; }
.form-group { display: flex; flex-direction: column; gap: 4px; flex: 1; min-width: 160px; }
label { font-size: 12px; font-weight: 500; color: var(--text-muted); }
input[type=text], input[type=number], select { padding: 7px 10px; border: 1px solid var(--border); border-radius: var(--radius); font-size: 13px; color: var(--text); background: #fff; width: 100%; }
input:focus, select:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
.path-input-row { display: flex; gap: 8px; }
.path-input-row input { flex: 1; }
.btn { padding: 7px 16px; border: none; border-radius: var(--radius); cursor: pointer; font-size: 13px; font-weight: 500; transition: background 0.15s; display: inline-flex; align-items: center; gap: 6px; }
.btn-primary { background: var(--accent); color: #fff; }
.btn-primary:hover { background: var(--accent-hover); }
.btn-secondary { background: #e8e5de; color: var(--text); }
.btn-secondary:hover { background: #d8d5ce; }
.btn-danger { background: var(--danger); color: #fff; }
.btn-sm { padding: 4px 10px; font-size: 12px; }

/* Checkboxes list */
.checkbox-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 8px; }
.checkbox-item { display: flex; align-items: center; gap: 8px; padding: 8px 10px; border: 1px solid var(--border); border-radius: var(--radius); cursor: pointer; transition: border-color 0.1s; }
.checkbox-item:hover { border-color: var(--accent); }
.checkbox-item input { cursor: pointer; }
.checkbox-item.checked { border-color: var(--accent); background: #eef2fc; }
.engine-status { width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
.status-ok { background: var(--success); }
.status-warn { background: var(--warning); }
.status-err { background: var(--danger); }

/* Progress */
.progress-bar-outer { height: 10px; background: #e0ddd5; border-radius: 5px; overflow: hidden; margin: 4px 0; }
.progress-bar-inner { height: 100%; background: var(--accent); border-radius: 5px; transition: width 0.3s; }
.log-box { background: #1a1a2e; color: #c8d8f8; font-family: monospace; font-size: 12px; padding: 12px; border-radius: var(--radius); max-height: 260px; overflow-y: auto; white-space: pre-wrap; line-height: 1.6; }
.log-box .log-warn { color: #f0c060; }
.log-box .log-error { color: #ff6b6b; }
.log-box .log-success { color: #6bf08a; }

/* Tables */
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; padding: 8px 10px; border-bottom: 2px solid var(--border); color: var(--text-muted); font-weight: 600; font-size: 12px; }
td { padding: 8px 10px; border-bottom: 1px solid var(--border); }
tr:last-child td { border-bottom: none; }
tr:hover td { background: #f0ede6; }
.badge { padding: 2px 7px; border-radius: 10px; font-size: 11px; font-weight: 500; }
.badge-ok { background: #d4edda; color: var(--success); }
.badge-warn { background: #fff3cd; color: var(--warning); }
.badge-err { background: #fde8e8; color: var(--danger); }

/* File browser */
#file-browser { border: 1px solid var(--border); border-radius: var(--radius); max-height: 300px; overflow-y: auto; }
.fb-item { display: flex; align-items: center; gap: 8px; padding: 8px 12px; cursor: pointer; border-bottom: 1px solid var(--border); }
.fb-item:last-child { border-bottom: none; }
.fb-item:hover { background: #f0ede6; }
.fb-icon { font-size: 16px; flex-shrink: 0; }
.fb-name { flex: 1; font-size: 13px; }
.fb-badge { font-size: 11px; color: var(--text-muted); }
.fb-path { font-size: 12px; color: var(--text-muted); padding: 6px 12px; background: #f4f2ed; border-bottom: 1px solid var(--border); font-family: monospace; }

/* Notifications */
.alert { padding: 10px 14px; border-radius: var(--radius); margin-bottom: 12px; font-size: 13px; }
.alert-success { background: #d4edda; color: var(--success); border: 1px solid #b8dfc4; }
.alert-error { background: #fde8e8; color: var(--danger); border: 1px solid #f5c6cb; }
.alert-info { background: #d0e4f7; color: #1a568c; border: 1px solid #b8d4ef; }

/* Dataset cards */
.ds-grid { display: grid; gap: 10px; }
.ds-card { border: 1px solid var(--border); border-radius: var(--radius); padding: 12px; background: #fff; }
.ds-card h4 { font-size: 13px; font-weight: 600; margin-bottom: 4px; }
.ds-card p { font-size: 12px; color: var(--text-muted); margin-bottom: 6px; }
.ds-meta { display: flex; gap: 8px; flex-wrap: wrap; }
.ds-tag { font-size: 11px; background: #eef2fc; color: var(--accent); padding: 2px 7px; border-radius: 10px; }

/* Spinner */
.spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid #ccc; border-top-color: var(--accent); border-radius: 50%; animation: spin 0.7s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>

<div id="header">
  <h1 data-i18n="app_title">Picarones <span class="version" id="app-version"></span></h1>
  <nav id="nav">
    <button class="nav-btn active" onclick="showView('benchmark')" data-i18n="nav_benchmark">Benchmark</button>
    <button class="nav-btn" onclick="showView('reports')" data-i18n="nav_reports">Rapports</button>
    <button class="nav-btn" onclick="showView('engines')" data-i18n="nav_engines">Moteurs</button>
    <button class="nav-btn" onclick="showView('import')" data-i18n="nav_import">Import</button>
  </nav>
  <button id="lang-btn" onclick="toggleLang()">EN</button>
</div>

<div id="main">

  <!-- ===== VUE BENCHMARK ===== -->
  <div id="view-benchmark" class="view active">

    <div class="card">
      <h2 data-i18n="bench_corpus_title">1. Corpus</h2>
      <div class="form-group">
        <label data-i18n="bench_corpus_label">Chemin vers le dossier corpus (paires image/.gt.txt)</label>
        <div class="path-input-row">
          <input type="text" id="corpus-path" placeholder="./corpus/" value="" />
          <button class="btn btn-secondary btn-sm" onclick="openFileBrowser()" data-i18n="bench_browse">Parcourir</button>
        </div>
      </div>
      <div id="file-browser-container" style="display:none; margin-top:10px;">
        <div class="fb-path" id="fb-current-path">.</div>
        <div id="file-browser"></div>
      </div>
      <div id="corpus-info" style="margin-top:8px; font-size:12px; color: var(--text-muted);"></div>
    </div>

    <div class="card">
      <h2 data-i18n="bench_engines_title">2. Moteurs et pipelines</h2>
      <div id="engine-checkboxes" class="checkbox-grid">
        <div style="color: var(--text-muted); font-size: 12px;" data-i18n="loading">Chargement…</div>
      </div>
    </div>

    <div class="card">
      <h2 data-i18n="bench_options_title">3. Options</h2>
      <div class="form-row">
        <div class="form-group">
          <label data-i18n="bench_norm_label">Profil de normalisation</label>
          <select id="norm-profile">
            <option value="nfc">NFC (standard)</option>
          </select>
        </div>
        <div class="form-group">
          <label data-i18n="bench_lang_label">Langue (Tesseract)</label>
          <input type="text" id="bench-lang" value="fra" placeholder="fra" />
        </div>
        <div class="form-group">
          <label data-i18n="bench_output_label">Dossier de sortie</label>
          <input type="text" id="output-dir" value="./rapports/" />
        </div>
        <div class="form-group">
          <label data-i18n="bench_name_label">Nom du rapport (optionnel)</label>
          <input type="text" id="report-name" placeholder="rapport_2024_01_15" />
        </div>
      </div>
    </div>

    <div style="display:flex; gap:10px; align-items:center; margin-bottom:16px;">
      <button class="btn btn-primary" id="start-btn" onclick="startBenchmark()" data-i18n="bench_start">▶ Lancer le benchmark</button>
      <button class="btn btn-secondary" id="cancel-btn" style="display:none;" onclick="cancelBenchmark()" data-i18n="bench_cancel">✕ Annuler</button>
      <span id="bench-status-text" style="font-size:12px; color: var(--text-muted);"></span>
    </div>

    <div id="bench-progress-section" style="display:none;">
      <div class="card">
        <h2 data-i18n="bench_progress_title">Progression</h2>
        <div id="engine-progress-list"></div>
        <div style="margin-top: 12px;">
          <label style="font-size:12px; color: var(--text-muted); display:block; margin-bottom:4px;" data-i18n="bench_log">Journal</label>
          <div class="log-box" id="bench-log"></div>
        </div>
      </div>
    </div>

    <div id="bench-result-section" style="display:none;">
      <div class="card">
        <h2 data-i18n="bench_result_title">Résultats</h2>
        <div id="bench-ranking-table"></div>
        <div style="margin-top:12px;">
          <a id="bench-report-link" href="#" class="btn btn-primary" target="_blank" data-i18n="bench_open_report">Ouvrir le rapport</a>
        </div>
      </div>
    </div>
  </div>

  <!-- ===== VUE RAPPORTS ===== -->
  <div id="view-reports" class="view">
    <div class="card">
      <h2 data-i18n="reports_title">Rapports générés</h2>
      <div class="form-row" style="margin-bottom:12px;">
        <div class="form-group" style="max-width:320px;">
          <label data-i18n="reports_dir_label">Dossier de rapports</label>
          <div class="path-input-row">
            <input type="text" id="reports-dir" value="." />
            <button class="btn btn-secondary btn-sm" onclick="loadReports()" data-i18n="reports_refresh">Rafraîchir</button>
          </div>
        </div>
      </div>
      <div id="reports-list">
        <div style="color: var(--text-muted); font-size: 12px;" data-i18n="loading">Chargement…</div>
      </div>
    </div>
  </div>

  <!-- ===== VUE MOTEURS ===== -->
  <div id="view-engines" class="view">
    <div class="card">
      <h2 data-i18n="engines_ocr_title">Moteurs OCR</h2>
      <div id="engines-ocr-list">
        <div style="color: var(--text-muted); font-size: 12px;" data-i18n="loading">Chargement…</div>
      </div>
    </div>
    <div class="card">
      <h2 data-i18n="engines_llm_title">LLMs disponibles</h2>
      <div id="engines-llm-list">
        <div style="color: var(--text-muted); font-size: 12px;" data-i18n="loading">Chargement…</div>
      </div>
    </div>
  </div>

  <!-- ===== VUE IMPORT ===== -->
  <div id="view-import" class="view">

    <!-- HTR-United -->
    <div class="card">
      <h2 data-i18n="import_htr_title">Import HTR-United</h2>
      <p style="font-size:12px; color:var(--text-muted); margin-bottom:12px;" data-i18n="import_htr_desc">
        Catalogue communautaire de corpus HTR/OCR pour documents patrimoniaux.
      </p>
      <div class="form-row">
        <div class="form-group" style="flex:2;">
          <label data-i18n="import_search_label">Recherche</label>
          <input type="text" id="htr-search" placeholder="médiéval, latin, manuscrits…" />
        </div>
        <div class="form-group">
          <label data-i18n="import_lang_filter">Langue</label>
          <select id="htr-lang-filter">
            <option value="" data-i18n="all">Toutes</option>
          </select>
        </div>
        <div class="form-group">
          <label data-i18n="import_script_filter">Type d'écriture</label>
          <select id="htr-script-filter">
            <option value="" data-i18n="all">Tous</option>
          </select>
        </div>
        <div class="form-group" style="justify-content: flex-end; padding-top: 18px;">
          <button class="btn btn-primary btn-sm" onclick="searchHTRUnited()" data-i18n="search">Rechercher</button>
        </div>
      </div>
      <div id="htr-results" class="ds-grid"></div>
    </div>

    <!-- HuggingFace -->
    <div class="card">
      <h2 data-i18n="import_hf_title">Import HuggingFace Datasets</h2>
      <p style="font-size:12px; color:var(--text-muted); margin-bottom:12px;" data-i18n="import_hf_desc">
        Datasets OCR/HTR publics depuis HuggingFace Hub (IAM, RIMES, CATMuS, Gallica…).
      </p>
      <div class="form-row">
        <div class="form-group" style="flex:2;">
          <label data-i18n="import_search_label">Recherche</label>
          <input type="text" id="hf-search" placeholder="medieval OCR, IAM, RIMES…" />
        </div>
        <div class="form-group">
          <label data-i18n="import_lang_filter">Langue</label>
          <input type="text" id="hf-lang-filter" placeholder="French, Latin…" />
        </div>
        <div class="form-group">
          <label data-i18n="import_tag_filter">Tags</label>
          <input type="text" id="hf-tags" placeholder="ocr, htr, historical…" />
        </div>
        <div class="form-group" style="justify-content: flex-end; padding-top: 18px;">
          <button class="btn btn-primary btn-sm" onclick="searchHuggingFace()" data-i18n="search">Rechercher</button>
        </div>
      </div>
      <div id="hf-results" class="ds-grid"></div>
    </div>

  </div><!-- end view-import -->

</div><!-- end #main -->

<!-- Import modal -->
<div id="import-modal" style="display:none; position:fixed; inset:0; background:rgba(0,0,0,0.4); z-index:200; align-items:center; justify-content:center;">
  <div class="card" style="width: 420px; max-width: 95vw;">
    <h2 id="import-modal-title" data-i18n="import_modal_title">Importer le corpus</h2>
    <input type="hidden" id="import-modal-type" />
    <input type="hidden" id="import-modal-id" />
    <div class="form-group" style="margin-bottom:12px;">
      <label data-i18n="import_output_dir">Dossier de destination</label>
      <input type="text" id="import-modal-output" value="./corpus/" />
    </div>
    <div class="form-group" style="margin-bottom:16px;">
      <label data-i18n="import_max_samples">Nombre max de documents</label>
      <input type="number" id="import-modal-max" value="100" min="1" max="10000" />
    </div>
    <div id="import-modal-status" style="margin-bottom:12px;"></div>
    <div style="display:flex; gap:8px;">
      <button class="btn btn-primary" onclick="confirmImport()" data-i18n="import_confirm">Importer</button>
      <button class="btn btn-secondary" onclick="closeImportModal()" data-i18n="cancel">Annuler</button>
    </div>
  </div>
</div>

<script>
// ─── i18n ────────────────────────────────────────────────────────────────────
const T = {
  fr: {
    app_title: "Picarones",
    nav_benchmark: "Benchmark",
    nav_reports: "Rapports",
    nav_engines: "Moteurs",
    nav_import: "Import",
    loading: "Chargement…",
    search: "Rechercher",
    all: "Tous",
    cancel: "Annuler",
    bench_corpus_title: "1. Corpus",
    bench_corpus_label: "Chemin vers le dossier corpus (paires image / .gt.txt)",
    bench_browse: "Parcourir",
    bench_engines_title: "2. Moteurs et pipelines",
    bench_options_title: "3. Options",
    bench_norm_label: "Profil de normalisation",
    bench_lang_label: "Langue (Tesseract)",
    bench_output_label: "Dossier de sortie",
    bench_name_label: "Nom du rapport (optionnel)",
    bench_start: "▶ Lancer le benchmark",
    bench_cancel: "✕ Annuler",
    bench_progress_title: "Progression",
    bench_log: "Journal",
    bench_result_title: "Résultats",
    bench_open_report: "Ouvrir le rapport",
    reports_title: "Rapports générés",
    reports_dir_label: "Dossier de rapports",
    reports_refresh: "Rafraîchir",
    engines_ocr_title: "Moteurs OCR",
    engines_llm_title: "LLMs disponibles",
    import_htr_title: "Import HTR-United",
    import_htr_desc: "Catalogue communautaire de corpus HTR/OCR pour documents patrimoniaux.",
    import_hf_title: "Import HuggingFace Datasets",
    import_hf_desc: "Datasets OCR/HTR publics depuis HuggingFace Hub (IAM, RIMES, CATMuS, Gallica…).",
    import_search_label: "Recherche",
    import_lang_filter: "Langue",
    import_script_filter: "Type d'écriture",
    import_tag_filter: "Tags",
    import_modal_title: "Importer le corpus",
    import_output_dir: "Dossier de destination",
    import_max_samples: "Nombre max de documents",
    import_confirm: "Importer",
    available: "disponible",
    not_installed: "non installé",
    configured: "configuré",
    missing_key: "clé manquante",
    running: "actif",
    not_running: "inactif",
    no_reports: "Aucun rapport trouvé.",
    lines: "lignes",
    centuries: "siècles",
  },
  en: {
    app_title: "Picarones",
    nav_benchmark: "Benchmark",
    nav_reports: "Reports",
    nav_engines: "Engines",
    nav_import: "Import",
    loading: "Loading…",
    search: "Search",
    all: "All",
    cancel: "Cancel",
    bench_corpus_title: "1. Corpus",
    bench_corpus_label: "Path to corpus directory (image / .gt.txt pairs)",
    bench_browse: "Browse",
    bench_engines_title: "2. Engines & pipelines",
    bench_options_title: "3. Options",
    bench_norm_label: "Normalization profile",
    bench_lang_label: "Language (Tesseract)",
    bench_output_label: "Output directory",
    bench_name_label: "Report name (optional)",
    bench_start: "▶ Start benchmark",
    bench_cancel: "✕ Cancel",
    bench_progress_title: "Progress",
    bench_log: "Log",
    bench_result_title: "Results",
    bench_open_report: "Open report",
    reports_title: "Generated reports",
    reports_dir_label: "Reports directory",
    reports_refresh: "Refresh",
    engines_ocr_title: "OCR Engines",
    engines_llm_title: "Available LLMs",
    import_htr_title: "Import from HTR-United",
    import_htr_desc: "Community catalogue of HTR/OCR datasets for heritage documents.",
    import_hf_title: "Import from HuggingFace Datasets",
    import_hf_desc: "Public OCR/HTR datasets from HuggingFace Hub (IAM, RIMES, CATMuS, Gallica…).",
    import_search_label: "Search",
    import_lang_filter: "Language",
    import_script_filter: "Script type",
    import_tag_filter: "Tags",
    import_modal_title: "Import corpus",
    import_output_dir: "Output directory",
    import_max_samples: "Max documents",
    import_confirm: "Import",
    available: "available",
    not_installed: "not installed",
    configured: "configured",
    missing_key: "key missing",
    running: "running",
    not_running: "not running",
    no_reports: "No reports found.",
    lines: "lines",
    centuries: "centuries",
  },
};
let lang = "fr";
function t(key) { return (T[lang][key]) || key; }
function toggleLang() {
  lang = lang === "fr" ? "en" : "fr";
  document.getElementById("lang-btn").textContent = lang === "fr" ? "EN" : "FR";
  document.querySelectorAll("[data-i18n]").forEach(el => {
    const k = el.getAttribute("data-i18n");
    if (T[lang][k]) el.textContent = T[lang][k];
  });
}

// ─── Navigation ──────────────────────────────────────────────────────────────
function showView(name) {
  document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
  document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
  const view = document.getElementById("view-" + name);
  if (view) view.classList.add("active");
  const btns = document.querySelectorAll(".nav-btn");
  const idx = ["benchmark","reports","engines","import"].indexOf(name);
  if (btns[idx]) btns[idx].classList.add("active");

  if (name === "reports") loadReports();
  if (name === "engines") loadEngines();
  if (name === "import") { searchHTRUnited(); searchHuggingFace(); }
}

// ─── Status / version ────────────────────────────────────────────────────────
async function loadStatus() {
  try {
    const r = await fetch("/api/status");
    const d = await r.json();
    document.getElementById("app-version").textContent = "v" + d.version;
  } catch(e) {}
}

// ─── Engine checkboxes ───────────────────────────────────────────────────────
async function loadEngineCheckboxes() {
  try {
    const r = await fetch("/api/engines");
    const d = await r.json();
    const container = document.getElementById("engine-checkboxes");
    container.innerHTML = "";

    [...d.engines, ...d.llms].forEach(eng => {
      const item = document.createElement("label");
      item.className = "checkbox-item" + (eng.available ? " checked" : "");
      const dot = `<span class="engine-status ${eng.available ? "status-ok" : "status-err"}"></span>`;
      const chk = `<input type="checkbox" name="engine" value="${eng.id}" ${eng.available ? "checked" : ""} ${eng.available ? "" : ""}>`;
      item.innerHTML = `${chk}${dot}<span>${eng.label}</span>`;
      item.querySelector("input").addEventListener("change", e => {
        item.classList.toggle("checked", e.target.checked);
      });
      container.appendChild(item);
    });

    // Store all engine data for later
    window._enginesData = d;
  } catch(e) {
    document.getElementById("engine-checkboxes").innerHTML =
      '<span style="color: var(--danger); font-size:12px;">Erreur chargement moteurs</span>';
  }
}

// ─── Normalization profiles ──────────────────────────────────────────────────
async function loadNormProfiles() {
  try {
    const r = await fetch("/api/normalization/profiles");
    const d = await r.json();
    const sel = document.getElementById("norm-profile");
    sel.innerHTML = "";
    d.profiles.forEach(p => {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = `${p.name} — ${p.description}`;
      if (p.id === "nfc") opt.selected = true;
      sel.appendChild(opt);
    });
  } catch(e) {}
}

// ─── File browser ────────────────────────────────────────────────────────────
let _fbVisible = false;
function openFileBrowser() {
  _fbVisible = !_fbVisible;
  const c = document.getElementById("file-browser-container");
  c.style.display = _fbVisible ? "block" : "none";
  if (_fbVisible) browsePath(".");
}
async function browsePath(path) {
  try {
    const r = await fetch(`/api/corpus/browse?path=${encodeURIComponent(path)}`);
    const d = await r.json();
    document.getElementById("fb-current-path").textContent = d.current_path;
    const fb = document.getElementById("file-browser");
    fb.innerHTML = "";
    if (d.parent_path) {
      const up = document.createElement("div");
      up.className = "fb-item";
      up.innerHTML = `<span class="fb-icon">⬆</span><span class="fb-name">..</span>`;
      up.onclick = () => browsePath(d.parent_path);
      fb.appendChild(up);
    }
    d.items.filter(i => i.is_dir).forEach(item => {
      const el = document.createElement("div");
      el.className = "fb-item";
      const hasCorpus = item.has_corpus ? `<span class="fb-badge" style="color:var(--success)">✓ ${item.gt_count} GT</span>` : "";
      el.innerHTML = `<span class="fb-icon">📁</span><span class="fb-name">${item.name}</span>${hasCorpus}`;
      el.onclick = () => {
        if (item.has_corpus) {
          document.getElementById("corpus-path").value = item.path;
          document.getElementById("corpus-info").textContent = `✓ ${item.gt_count} documents GT trouvés.`;
          _fbVisible = false;
          document.getElementById("file-browser-container").style.display = "none";
        } else {
          browsePath(item.path);
        }
      };
      fb.appendChild(el);
    });
    if (fb.children.length === 0) {
      fb.innerHTML = '<div style="padding:12px; color: var(--text-muted); font-size:12px;">Dossier vide</div>';
    }
  } catch(e) {
    document.getElementById("file-browser").innerHTML =
      `<div style="padding:12px; color: var(--danger); font-size:12px;">Erreur : ${e.message}</div>`;
  }
}

// ─── Benchmark ───────────────────────────────────────────────────────────────
let _currentJobId = null;
let _eventSource = null;

async function startBenchmark() {
  const corpusPath = document.getElementById("corpus-path").value.trim();
  if (!corpusPath) {
    alert(lang === "fr" ? "Veuillez sélectionner un dossier corpus." : "Please select a corpus directory.");
    return;
  }
  const engines = Array.from(document.querySelectorAll("input[name=engine]:checked")).map(e => e.value);
  if (engines.length === 0) {
    alert(lang === "fr" ? "Veuillez sélectionner au moins un moteur." : "Please select at least one engine.");
    return;
  }

  const payload = {
    corpus_path: corpusPath,
    engines: engines,
    normalization_profile: document.getElementById("norm-profile").value,
    output_dir: document.getElementById("output-dir").value,
    report_name: document.getElementById("report-name").value,
    lang: document.getElementById("bench-lang").value,
  };

  document.getElementById("start-btn").disabled = true;
  document.getElementById("cancel-btn").style.display = "inline-flex";
  document.getElementById("bench-progress-section").style.display = "block";
  document.getElementById("bench-result-section").style.display = "none";
  document.getElementById("bench-log").textContent = "";
  document.getElementById("engine-progress-list").innerHTML = "";
  document.getElementById("bench-status-text").textContent = lang === "fr" ? "Démarrage…" : "Starting…";

  try {
    const r = await fetch("/api/benchmark/start", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    if (!r.ok) {
      const err = await r.json();
      throw new Error(err.detail || "Erreur serveur");
    }
    const d = await r.json();
    _currentJobId = d.job_id;
    _startSSE(_currentJobId, engines);
  } catch(e) {
    appendLog(`Erreur : ${e.message}`, "error");
    document.getElementById("start-btn").disabled = false;
    document.getElementById("cancel-btn").style.display = "none";
    document.getElementById("bench-status-text").textContent = "";
  }
}

function _startSSE(jobId, engines) {
  if (_eventSource) _eventSource.close();
  // Init engine progress bars
  const pl = document.getElementById("engine-progress-list");
  pl.innerHTML = "";
  engines.forEach(eng => {
    const div = document.createElement("div");
    div.id = `eng-progress-${eng}`;
    div.style = "margin-bottom: 8px;";
    div.innerHTML = `<div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:3px;">
      <span>${eng}</span><span id="eng-pct-${eng}">0%</span></div>
      <div class="progress-bar-outer"><div class="progress-bar-inner" id="eng-bar-${eng}" style="width:0%"></div></div>`;
    pl.appendChild(div);
  });

  _eventSource = new EventSource(`/api/benchmark/${jobId}/stream`);

  _eventSource.addEventListener("start", e => {
    const d = JSON.parse(e.data);
    appendLog(d.message, "success");
    document.getElementById("bench-status-text").textContent = lang === "fr" ? "En cours…" : "Running…";
  });

  _eventSource.addEventListener("log", e => {
    const d = JSON.parse(e.data);
    appendLog(d.message);
  });

  _eventSource.addEventListener("warning", e => {
    const d = JSON.parse(e.data);
    appendLog(d.message, "warn");
  });

  _eventSource.addEventListener("progress", e => {
    const d = JSON.parse(e.data);
    const pct = Math.round(d.progress * 100);
    document.getElementById("bench-status-text").textContent =
      `${pct}% — ${d.engine} (${d.processed}/${d.total})`;
    engines.forEach(eng => {
      const bar = document.getElementById(`eng-bar-${eng}`);
      const pctEl = document.getElementById(`eng-pct-${eng}`);
      if (d.engine === eng && bar && pctEl) {
        bar.style.width = pct + "%";
        pctEl.textContent = pct + "%";
      }
    });
  });

  _eventSource.addEventListener("complete", e => {
    const d = JSON.parse(e.data);
    appendLog(d.message, "success");
    _showResults(d);
    _finishBenchmark();
  });

  _eventSource.addEventListener("error", e => {
    const d = JSON.parse(e.data);
    appendLog(d.message, "error");
    _finishBenchmark();
  });

  _eventSource.addEventListener("cancelled", e => {
    appendLog(lang === "fr" ? "Benchmark annulé." : "Benchmark cancelled.", "warn");
    _finishBenchmark();
  });

  _eventSource.addEventListener("done", e => {
    _finishBenchmark();
  });

  _eventSource.onerror = () => {
    if (_currentJobId) {
      _finishBenchmark();
    }
  };
}

function _showResults(data) {
  const section = document.getElementById("bench-result-section");
  section.style.display = "block";
  if (data.output_html) {
    const link = document.getElementById("bench-report-link");
    link.href = `/reports/${data.output_html.split("/").pop()}`;
  }
  if (data.ranking) {
    let html = `<table><thead><tr><th>#</th><th>${lang==="fr"?"Moteur":"Engine"}</th><th>CER</th><th>WER</th><th>${lang==="fr"?"Docs":"Docs"}</th></tr></thead><tbody>`;
    data.ranking.forEach((row, i) => {
      const cer = row.mean_cer != null ? (row.mean_cer*100).toFixed(2)+"%" : "N/A";
      const wer = row.mean_wer != null ? (row.mean_wer*100).toFixed(2)+"%" : "N/A";
      html += `<tr><td>${i+1}</td><td>${row.engine}</td><td>${cer}</td><td>${wer}</td><td>${row.total_docs || ""}</td></tr>`;
    });
    html += "</tbody></table>";
    document.getElementById("bench-ranking-table").innerHTML = html;
  }
}

function _finishBenchmark() {
  if (_eventSource) { _eventSource.close(); _eventSource = null; }
  document.getElementById("start-btn").disabled = false;
  document.getElementById("cancel-btn").style.display = "none";
  document.getElementById("bench-status-text").textContent = "";
}

async function cancelBenchmark() {
  if (!_currentJobId) return;
  await fetch(`/api/benchmark/${_currentJobId}/cancel`, {method: "POST"});
}

function appendLog(msg, cls) {
  const box = document.getElementById("bench-log");
  const line = document.createElement("div");
  if (cls === "error") line.className = "log-error";
  else if (cls === "warn") line.className = "log-warn";
  else if (cls === "success") line.className = "log-success";
  line.textContent = msg;
  box.appendChild(line);
  box.scrollTop = box.scrollHeight;
}

// ─── Reports ─────────────────────────────────────────────────────────────────
async function loadReports() {
  const dir = document.getElementById("reports-dir").value || ".";
  const container = document.getElementById("reports-list");
  container.innerHTML = `<div style="color: var(--text-muted); font-size:12px;">${t("loading")}</div>`;
  try {
    const r = await fetch(`/api/reports?reports_dir=${encodeURIComponent(dir)}`);
    const d = await r.json();
    if (d.reports.length === 0) {
      container.innerHTML = `<div style="color: var(--text-muted); font-size:12px;">${t("no_reports")}</div>`;
      return;
    }
    let html = `<table><thead><tr><th>${lang==="fr"?"Fichier":"File"}</th><th>${lang==="fr"?"Taille":"Size"}</th><th>${lang==="fr"?"Modifié":"Modified"}</th><th></th></tr></thead><tbody>`;
    d.reports.forEach(rep => {
      const date = new Date(rep.modified).toLocaleString(lang === "fr" ? "fr-FR" : "en-US");
      html += `<tr><td>${rep.filename}</td><td>${rep.size_kb} Ko</td><td>${date}</td>
        <td><a href="${rep.url}" target="_blank" class="btn btn-primary btn-sm">${lang==="fr"?"Ouvrir":"Open"}</a></td></tr>`;
    });
    html += "</tbody></table>";
    container.innerHTML = html;
  } catch(e) {
    container.innerHTML = `<div style="color: var(--danger); font-size:12px;">Erreur : ${e.message}</div>`;
  }
}

// ─── Engines status ──────────────────────────────────────────────────────────
async function loadEngines() {
  try {
    const r = await fetch("/api/engines");
    const d = await r.json();

    // OCR
    let html = `<table><thead><tr><th>ID</th><th>${lang==="fr"?"Nom":"Name"}</th><th>Version</th><th>Statut</th></tr></thead><tbody>`;
    d.engines.forEach(e => {
      const cls = e.available ? "badge-ok" : "badge-err";
      const lbl = e.available ? t("available") : t("not_installed");
      html += `<tr><td><code>${e.id}</code></td><td>${e.label}</td><td>${e.version||"—"}</td>
        <td><span class="badge ${cls}">${lbl}</span></td></tr>`;
    });
    html += "</tbody></table>";
    document.getElementById("engines-ocr-list").innerHTML = html;

    // LLMs
    let llmHtml = `<table><thead><tr><th>ID</th><th>${lang==="fr"?"Nom":"Name"}</th><th>Statut</th><th>${lang==="fr"?"Détail":"Detail"}</th></tr></thead><tbody>`;
    d.llms.forEach(e => {
      const cls = e.available ? "badge-ok" : "badge-warn";
      const statusKey = e.status === "configured" ? "configured"
        : e.status === "running" ? "running"
        : e.status === "not_running" ? "not_running"
        : "missing_key";
      const lbl = t(statusKey);
      let detail = "";
      if (e.key_env) detail = `<code style="font-size:11px;">${e.key_env}</code>`;
      if (e.models && e.models.length > 0) detail = e.models.slice(0, 3).join(", ");
      llmHtml += `<tr><td><code>${e.id}</code></td><td>${e.label}</td>
        <td><span class="badge ${cls}">${lbl}</span></td><td>${detail}</td></tr>`;
    });
    llmHtml += "</tbody></table>";
    document.getElementById("engines-llm-list").innerHTML = llmHtml;
  } catch(e) {
    document.getElementById("engines-ocr-list").innerHTML =
      `<div style="color: var(--danger); font-size:12px;">Erreur : ${e.message}</div>`;
  }
}

// ─── HTR-United ──────────────────────────────────────────────────────────────
async function initHTRFilters() {
  try {
    const r = await fetch("/api/htr-united/catalogue");
    const d = await r.json();
    const langSel = document.getElementById("htr-lang-filter");
    const scriptSel = document.getElementById("htr-script-filter");
    langSel.innerHTML = `<option value="">${t("all")}</option>`;
    d.available_languages.forEach(l => {
      langSel.innerHTML += `<option value="${l}">${l}</option>`;
    });
    scriptSel.innerHTML = `<option value="">${t("all")}</option>`;
    d.available_scripts.forEach(s => {
      scriptSel.innerHTML += `<option value="${s}">${s}</option>`;
    });
  } catch(e) {}
}

async function searchHTRUnited() {
  const q = document.getElementById("htr-search").value;
  const lang2 = document.getElementById("htr-lang-filter").value;
  const script = document.getElementById("htr-script-filter").value;
  const container = document.getElementById("htr-results");
  container.innerHTML = `<div style="color: var(--text-muted); font-size:12px;">${t("loading")}</div>`;
  try {
    const url = `/api/htr-united/catalogue?query=${encodeURIComponent(q)}&language=${encodeURIComponent(lang2)}&script=${encodeURIComponent(script)}`;
    const r = await fetch(url);
    const d = await r.json();
    if (d.entries.length === 0) {
      container.innerHTML = `<div style="color: var(--text-muted); font-size:12px;">${lang==="fr"?"Aucun résultat.":"No results."}</div>`;
      return;
    }
    container.innerHTML = d.entries.map(e => {
      const tags = [...e.language, ...e.script].map(s => `<span class="ds-tag">${s}</span>`).join("");
      return `<div class="ds-card">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
          <h4>${e.title}</h4>
          <button class="btn btn-primary btn-sm" onclick="openImportModal('htr', '${e.id}', '${e.title.replace(/'/g,"\\'")}')">
            ${lang==="fr"?"Importer":"Import"}
          </button>
        </div>
        <p>${e.description}</p>
        <p style="color: var(--text-muted);">${e.institution} — ${e.lines.toLocaleString()} ${t("lines")} — ${e.format}</p>
        <div class="ds-meta">${tags}</div>
      </div>`;
    }).join("");
  } catch(e) {
    container.innerHTML = `<div style="color: var(--danger); font-size:12px;">Erreur : ${e.message}</div>`;
  }
}

async function searchHuggingFace() {
  const q = document.getElementById("hf-search").value;
  const langFilter = document.getElementById("hf-lang-filter").value;
  const tags = document.getElementById("hf-tags").value;
  const container = document.getElementById("hf-results");
  container.innerHTML = `<div style="color: var(--text-muted); font-size:12px;">${t("loading")}</div>`;
  try {
    const url = `/api/huggingface/search?query=${encodeURIComponent(q)}&language=${encodeURIComponent(langFilter)}&tags=${encodeURIComponent(tags)}`;
    const r = await fetch(url);
    const d = await r.json();
    if (d.datasets.length === 0) {
      container.innerHTML = `<div style="color: var(--text-muted); font-size:12px;">${lang==="fr"?"Aucun résultat.":"No results."}</div>`;
      return;
    }
    container.innerHTML = d.datasets.map(ds => {
      const tags2 = ds.tags.slice(0,5).map(s => `<span class="ds-tag">${s}</span>`).join("");
      return `<div class="ds-card">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
          <h4>${ds.title}</h4>
          <button class="btn btn-primary btn-sm" onclick="openImportModal('hf', '${ds.dataset_id.replace(/'/g,"\\'")}', '${ds.title.replace(/'/g,"\\'")}')">
            ${lang==="fr"?"Importer":"Import"}
          </button>
        </div>
        <p>${ds.description}</p>
        <p style="color: var(--text-muted);">${ds.institution||ds.dataset_id} ${ds.downloads ? "— " + ds.downloads.toLocaleString() + " téléchargements" : ""}</p>
        <div class="ds-meta">${tags2}</div>
      </div>`;
    }).join("");
  } catch(e) {
    container.innerHTML = `<div style="color: var(--danger); font-size:12px;">Erreur : ${e.message}</div>`;
  }
}

// ─── Import modal ─────────────────────────────────────────────────────────────
function openImportModal(type, id, title) {
  document.getElementById("import-modal-type").value = type;
  document.getElementById("import-modal-id").value = id;
  document.getElementById("import-modal-title").textContent = `${t("import_modal_title")} : ${title}`;
  document.getElementById("import-modal-status").innerHTML = "";
  document.getElementById("import-modal").style.display = "flex";
}
function closeImportModal() {
  document.getElementById("import-modal").style.display = "none";
}
async function confirmImport() {
  const type = document.getElementById("import-modal-type").value;
  const id = document.getElementById("import-modal-id").value;
  const outputDir = document.getElementById("import-modal-output").value;
  const maxSamples = parseInt(document.getElementById("import-modal-max").value);
  const statusDiv = document.getElementById("import-modal-status");
  statusDiv.innerHTML = `<div class="alert alert-info"><span class="spinner"></span> ${lang==="fr"?"Import en cours…":"Importing…"}</div>`;

  try {
    let url, body;
    if (type === "htr") {
      url = "/api/htr-united/import";
      body = {entry_id: id, output_dir: outputDir, max_samples: maxSamples};
    } else {
      url = "/api/huggingface/import";
      body = {dataset_id: id, output_dir: outputDir, max_samples: maxSamples};
    }
    const r = await fetch(url, {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body)});
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || "Erreur");
    const msg = lang === "fr"
      ? `✓ Import terminé. ${d.files_imported || 0} fichiers dans <code>${d.output_dir}</code>`
      : `✓ Import done. ${d.files_imported || 0} files in <code>${d.output_dir}</code>`;
    statusDiv.innerHTML = `<div class="alert alert-success">${msg}</div>`;
    // Suggestion de corpus path
    document.getElementById("corpus-path").value = d.output_dir;
  } catch(e) {
    statusDiv.innerHTML = `<div class="alert alert-error">Erreur : ${e.message}</div>`;
  }
}

// ─── Init ────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  loadStatus();
  loadEngineCheckboxes();
  loadNormProfiles();
  initHTRFilters();
  // Close modal on backdrop click
  document.getElementById("import-modal").addEventListener("click", e => {
    if (e.target === document.getElementById("import-modal")) closeImportModal();
  });
});
</script>
</body>
</html>"""
