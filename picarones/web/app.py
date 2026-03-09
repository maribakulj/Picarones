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
import shutil
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from fastapi import Cookie, FastAPI, File, HTTPException, Query, Response, UploadFile
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

_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"})
_UPLOADS_DIR = Path("./uploads")

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


class CompetitorConfig(BaseModel):
    name: str = ""
    ocr_engine: str
    ocr_model: str = ""
    llm_provider: str = ""
    llm_model: str = ""
    pipeline_mode: str = ""
    prompt_file: str = ""


class BenchmarkRunRequest(BaseModel):
    corpus_path: str
    competitors: list[CompetitorConfig]
    normalization_profile: str = "nfc"
    output_dir: str = "./rapports/"
    report_name: str = ""
    report_lang: str = "fr"


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
    tess["langs"] = _get_tesseract_langs()
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

    # Mistral OCR (API cloud)
    mistral_key = os.environ.get("MISTRAL_API_KEY")
    engines.append({
        "id": "mistral_ocr",
        "label": "Mistral OCR (Pixtral / mistral-ocr-latest)",
        "type": "ocr_cloud",
        "available": bool(mistral_key),
        "key_env": "MISTRAL_API_KEY",
        "status": "configured" if mistral_key else "missing_key",
        "version": "",
    })

    # Google Vision (API cloud)
    gv_key = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get("GOOGLE_API_KEY")
    engines.append({
        "id": "google_vision",
        "label": "Google Vision API",
        "type": "ocr_cloud",
        "available": bool(gv_key),
        "key_env": "GOOGLE_APPLICATION_CREDENTIALS",
        "status": "configured" if gv_key else "missing_key",
        "version": "",
    })

    # Azure Document Intelligence (API cloud)
    az_key = os.environ.get("AZURE_DOC_INTEL_KEY")
    engines.append({
        "id": "azure_doc_intel",
        "label": "Azure Document Intelligence",
        "type": "ocr_cloud",
        "available": bool(az_key),
        "key_env": "AZURE_DOC_INTEL_KEY",
        "status": "configured" if az_key else "missing_key",
        "version": "",
    })

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

    # Mistral LLM
    llms.append({
        "id": "mistral",
        "label": "Mistral LLM (Mistral Large, Small…)",
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


def _get_tesseract_langs() -> list[str]:
    try:
        import pytesseract
        langs = pytesseract.get_languages(config="")
        return sorted(l for l in langs if l != "osd")
    except Exception:
        return ["fra", "lat", "eng", "deu", "ita", "spa"]


# ---------------------------------------------------------------------------
# API — models (dynamic per provider)
# ---------------------------------------------------------------------------

@app.get("/api/models/{provider}")
async def api_models(provider: str) -> dict:
    """Retourne la liste des modèles disponibles pour un provider, en temps réel."""
    import urllib.error
    import urllib.request as _urlreq

    def _fetch_json(url: str, headers: dict) -> dict:
        req = _urlreq.Request(url, headers=headers)
        with _urlreq.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())

    if provider == "tesseract":
        return {"provider": provider, "models": _get_tesseract_langs()}

    if provider == "mistral_ocr":
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            return {"provider": provider, "models": [], "error": "MISTRAL_API_KEY non définie"}
        try:
            data = _fetch_json(
                "https://api.mistral.ai/v1/models",
                {"Authorization": f"Bearer {api_key}"},
            )
            models = sorted(
                m["id"] for m in data.get("data", [])
                if "pixtral" in m["id"].lower() or "mistral-ocr" in m["id"].lower()
            )
            return {"provider": provider, "models": models}
        except Exception as exc:
            return {
                "provider": provider,
                "models": ["pixtral-12b-2409", "pixtral-large-latest", "mistral-ocr-latest"],
                "error": str(exc),
            }

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return {"provider": provider, "models": [], "error": "OPENAI_API_KEY non définie"}
        try:
            data = _fetch_json(
                "https://api.openai.com/v1/models",
                {"Authorization": f"Bearer {api_key}"},
            )
            models = sorted(
                (m["id"] for m in data.get("data", []) if "gpt-4" in m["id"].lower()),
                reverse=True,
            )
            return {"provider": provider, "models": models}
        except Exception as exc:
            return {
                "provider": provider,
                "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
                "error": str(exc),
            }

    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return {"provider": provider, "models": [], "error": "ANTHROPIC_API_KEY non définie"}
        try:
            data = _fetch_json(
                "https://api.anthropic.com/v1/models",
                {"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            )
            models = [m["id"] for m in data.get("data", [])]
            return {"provider": provider, "models": models}
        except Exception as exc:
            return {
                "provider": provider,
                "models": ["claude-sonnet-4-6", "claude-haiku-4-5-20251001", "claude-opus-4-6"],
                "error": str(exc),
            }

    if provider == "mistral":
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            return {"provider": provider, "models": [], "error": "MISTRAL_API_KEY non définie"}
        try:
            data = _fetch_json(
                "https://api.mistral.ai/v1/models",
                {"Authorization": f"Bearer {api_key}"},
            )
            models = sorted(
                m["id"] for m in data.get("data", [])
                if "pixtral" not in m["id"].lower() and "mistral-ocr" not in m["id"].lower()
            )
            return {"provider": provider, "models": models}
        except Exception as exc:
            return {
                "provider": provider,
                "models": ["mistral-large-latest", "mistral-small-latest"],
                "error": str(exc),
            }

    if provider == "ollama":
        return {"provider": provider, "models": _list_ollama_models()}

    if provider == "google_vision":
        return {"provider": provider, "models": ["document_text_detection", "text_detection"]}

    if provider == "azure_doc_intel":
        return {"provider": provider, "models": ["prebuilt-document", "prebuilt-read"]}

    if provider == "prompts":
        prompts_dir = Path(__file__).parent.parent / "prompts"
        if prompts_dir.exists():
            prompts = sorted(f.name for f in prompts_dir.glob("*.txt"))
        else:
            prompts = []
        return {"provider": provider, "models": prompts}

    raise HTTPException(status_code=404, detail=f"Provider inconnu : {provider}")


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
# API — corpus upload
# ---------------------------------------------------------------------------

def _analyze_corpus_dir(path: Path) -> dict:
    """Analyse un dossier et retourne un résumé des paires image/GT détectées."""
    images = sorted(f.name for f in path.iterdir() if f.suffix.lower() in _IMAGE_EXTS)
    pairs: list[dict] = []
    missing_gt: list[str] = []
    for img in images:
        stem = Path(img).stem
        gt = path / (stem + ".gt.txt")
        if gt.exists():
            pairs.append({"image": img, "gt": stem + ".gt.txt"})
        else:
            missing_gt.append(img)
    return {
        "doc_count": len(pairs),
        "pairs": pairs[:20],
        "total_pairs": len(pairs),
        "missing_gt": missing_gt[:10],
        "has_missing_gt": len(missing_gt) > 0,
        "warnings": [f"GT manquant : {img}" for img in missing_gt[:5]],
        "usable": len(pairs) > 0,
    }


def _flatten_zip_to_dir(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extrait un ZIP en aplatissant les paires image/.gt.txt dans dest."""
    dest.mkdir(parents=True, exist_ok=True)
    for member in zf.infolist():
        if member.is_dir():
            continue
        p = Path(member.filename)
        name = p.name
        # Accepter images et .gt.txt
        if p.suffix.lower() in _IMAGE_EXTS or name.endswith(".gt.txt"):
            data = zf.read(member.filename)
            (dest / name).write_bytes(data)


@app.post("/api/corpus/upload")
async def api_corpus_upload(files: list[UploadFile] = File(...)) -> dict:
    """Upload un corpus : soit un .zip, soit une sélection d'images + .gt.txt."""
    corpus_id = str(uuid.uuid4())
    corpus_dir = _UPLOADS_DIR / corpus_id
    corpus_dir.mkdir(parents=True, exist_ok=True)

    try:
        for uf in files:
            filename = uf.filename or "upload"
            data = await uf.read()
            suffix = Path(filename).suffix.lower()

            if suffix == ".zip":
                # Extraire le ZIP en aplatissant les paires
                import io
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    _flatten_zip_to_dir(zf, corpus_dir)
            elif suffix in _IMAGE_EXTS or filename.endswith(".gt.txt") or suffix == ".txt":
                (corpus_dir / filename).write_bytes(data)
            # Ignorer les autres types

        summary = _analyze_corpus_dir(corpus_dir)
        if not summary["usable"]:
            shutil.rmtree(corpus_dir, ignore_errors=True)
            raise HTTPException(
                status_code=422,
                detail="Aucune paire image/.gt.txt valide trouvée dans les fichiers uploadés.",
            )

        return {
            "corpus_id": corpus_id,
            "corpus_path": str(corpus_dir),
            **summary,
        }
    except HTTPException:
        raise
    except Exception as exc:
        shutil.rmtree(corpus_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/corpus/uploads")
async def api_corpus_uploads() -> dict:
    """Liste les corpus uploadés disponibles."""
    if not _UPLOADS_DIR.exists():
        return {"uploads": []}

    uploads = []
    for d in sorted(_UPLOADS_DIR.iterdir()):
        if not d.is_dir():
            continue
        try:
            summary = _analyze_corpus_dir(d)
            uploads.append({
                "corpus_id": d.name,
                "corpus_path": str(d),
                "doc_count": summary["doc_count"],
                "has_missing_gt": summary["has_missing_gt"],
            })
        except Exception:
            pass
    return {"uploads": uploads}


@app.delete("/api/corpus/uploads/{corpus_id}")
async def api_corpus_delete(corpus_id: str) -> dict:
    """Supprime un corpus uploadé."""
    # Sécurité : interdire les path traversal
    if "/" in corpus_id or "\\" in corpus_id or ".." in corpus_id:
        raise HTTPException(status_code=400, detail="corpus_id invalide")
    corpus_dir = _UPLOADS_DIR / corpus_id
    if not corpus_dir.exists() or not corpus_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Corpus non trouvé : {corpus_id}")
    shutil.rmtree(corpus_dir)
    return {"deleted": corpus_id}


# ---------------------------------------------------------------------------
# API — normalization profiles
# ---------------------------------------------------------------------------

@app.get("/api/normalization/profiles")
async def api_normalization_profiles() -> dict:
    from picarones.core.normalization import NORMALIZATION_PROFILES

    profiles = [
        {
            "id": pid,
            "name": p.name,
            "description": p.description or p.name,
            "caseless": p.caseless,
            "diplomatic_rules": len(p.diplomatic_table),
        }
        for pid, p in NORMALIZATION_PROFILES.items()
    ]
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


@app.get("/reports/{filename}", response_class=HTMLResponse)
async def serve_report(filename: str) -> HTMLResponse:
    # Cherche dans le répertoire courant et ./rapports/
    # Lecture directe + renvoi en text/html pour fonctionner depuis un Codespace
    # ou tout reverse-proxy distant (pas de redirect vers fichier statique).
    for d in [Path("."), Path("./rapports")]:
        f = d / filename
        if f.exists() and f.suffix == ".html":
            content = f.read_text(encoding="utf-8")
            return HTMLResponse(content=content)
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


# ---------------------------------------------------------------------------
# API — benchmark/run (concurrents composés)
# ---------------------------------------------------------------------------

@app.post("/api/benchmark/run")
async def api_benchmark_run(req: BenchmarkRunRequest) -> dict:
    corpus_path = Path(req.corpus_path)
    if not corpus_path.exists() or not corpus_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Corpus non trouvé : {req.corpus_path}")
    if not req.competitors:
        raise HTTPException(status_code=400, detail="Aucun concurrent défini.")

    job_id = str(uuid.uuid4())
    job = BenchmarkJob(job_id=job_id)
    _JOBS[job_id] = job

    thread = threading.Thread(
        target=_run_benchmark_thread_v2,
        args=(job, req),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "pending"}


def _engine_from_competitor(comp: CompetitorConfig) -> Any:
    """Instancie un moteur OCR (ou pipeline OCR+LLM) depuis une CompetitorConfig."""
    from picarones.engines.tesseract import TesseractEngine
    from picarones.engines.mistral_ocr import MistralOCREngine

    engine_id = comp.ocr_engine

    if engine_id == "tesseract":
        ocr = TesseractEngine(config={"lang": comp.ocr_model or "fra", "psm": 6})
    elif engine_id == "mistral_ocr":
        ocr = MistralOCREngine(config={"model": comp.ocr_model or "pixtral-12b-2409"})
    elif engine_id == "google_vision":
        try:
            from picarones.engines.google_vision import GoogleVisionEngine
            ocr = GoogleVisionEngine(config={"detection_type": comp.ocr_model or "document_text_detection"})
        except ImportError as exc:
            raise RuntimeError("Google Vision non disponible (google-cloud-vision non installé).") from exc
    elif engine_id == "azure_doc_intel":
        try:
            from picarones.engines.azure_doc_intel import AzureDocIntelEngine
            ocr = AzureDocIntelEngine(config={"model": comp.ocr_model or "prebuilt-document"})
        except ImportError as exc:
            raise RuntimeError("Azure Document Intelligence non disponible.") from exc
    else:
        raise ValueError(f"Moteur OCR inconnu : {engine_id}")

    if not comp.llm_provider:
        return ocr

    # Pipeline OCR+LLM
    _mode_map = {
        "text_only": "text_only",
        "post_correction_text": "text_only",
        "text_and_image": "text_and_image",
        "post_correction_image": "text_and_image",
        "zero_shot": "zero_shot",
    }
    mode = _mode_map.get(comp.pipeline_mode, "text_only")

    if comp.llm_provider == "openai":
        from picarones.llm.openai_adapter import OpenAIAdapter
        llm = OpenAIAdapter(model=comp.llm_model or None)
    elif comp.llm_provider == "anthropic":
        from picarones.llm.anthropic_adapter import AnthropicAdapter
        llm = AnthropicAdapter(model=comp.llm_model or None)
    elif comp.llm_provider == "mistral":
        from picarones.llm.mistral_adapter import MistralAdapter
        llm = MistralAdapter(model=comp.llm_model or None)
    elif comp.llm_provider == "ollama":
        from picarones.llm.ollama_adapter import OllamaAdapter
        llm = OllamaAdapter(model=comp.llm_model or None)
    else:
        raise ValueError(f"Provider LLM inconnu : {comp.llm_provider}")

    from picarones.pipelines.base import OCRLLMPipeline
    prompt = comp.prompt_file or "correction_medieval_french.txt"
    pipeline_name = comp.name or f"{engine_id}→{comp.llm_model or comp.llm_provider}"
    return OCRLLMPipeline(
        ocr_engine=ocr,
        llm_adapter=llm,
        mode=mode,
        prompt=prompt,
        pipeline_name=pipeline_name,
    )


def _run_benchmark_thread_v2(job: BenchmarkJob, req: BenchmarkRunRequest) -> None:
    """Exécute un benchmark à partir d'une liste de CompetitorConfig."""
    import time

    job.status = "running"
    job.started_at = _iso_now()
    job.add_event("start", {"message": "Démarrage du benchmark…", "corpus": req.corpus_path})

    try:
        from picarones.core.corpus import load_corpus_from_directory
        from picarones.core.runner import run_benchmark

        corpus = load_corpus_from_directory(req.corpus_path)
        job.total_docs = len(corpus)
        job.add_event("log", {"message": f"{job.total_docs} documents chargés."})

        if job.status == "cancelled":
            return

        engines = []
        for comp in req.competitors:
            try:
                eng = _engine_from_competitor(comp)
                engines.append(eng)
                job.add_event("log", {"message": f"Concurrent : {eng.name}"})
            except Exception as exc:
                job.add_event("warning", {
                    "message": f"Concurrent ignoré '{comp.name or comp.ocr_engine}' : {exc}"
                })

        if not engines:
            raise ValueError("Aucun concurrent valide disponible.")

        output_dir = Path(req.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_name = req.report_name or f"rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_json = str(output_dir / f"{report_name}.json")
        output_html = str(output_dir / f"{report_name}.html")

        n_engines = len(engines)
        total_steps = job.total_docs * n_engines
        step_counter = [0]

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

        result = run_benchmark(
            corpus=corpus,
            engines=engines,
            output_json=output_json,
            show_progress=False,
            progress_callback=_progress_callback,
        )

        if job.status == "cancelled":
            return

        job.add_event("log", {"message": "Génération du rapport HTML…"})
        from picarones.report.generator import ReportGenerator
        gen = ReportGenerator(result, lang=req.report_lang)
        gen.generate(output_html)

        job.output_path = output_html
        job.progress = 1.0
        job.status = "complete"
        job.finished_at = _iso_now()

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

/* Corpus upload tabs */
.corpus-tabs { display: flex; gap: 0; margin-bottom: 14px; border-bottom: 2px solid var(--border); }
.corpus-tab { background: transparent; border: none; border-bottom: 2px solid transparent; margin-bottom: -2px; padding: 6px 14px; font-size: 13px; font-weight: 500; cursor: pointer; color: var(--text-muted); transition: color 0.15s, border-color 0.15s; }
.corpus-tab:hover { color: var(--accent); }
.corpus-tab.active { color: var(--accent); border-bottom-color: var(--accent); }

/* Upload drop zone */
.upload-dropzone { border: 2px dashed var(--border); border-radius: var(--radius); padding: 28px 20px; text-align: center; cursor: pointer; transition: border-color 0.2s, background 0.2s; color: var(--text-muted); font-size: 13px; }
.upload-dropzone:hover, .upload-dropzone.dragover { border-color: var(--accent); background: #eef2fc; color: var(--accent); }
.upload-dropzone .upload-icon { font-size: 28px; display: block; margin-bottom: 6px; }

/* Upload mode toggle */
.upload-mode-row { display: flex; gap: 20px; padding: 8px 12px; background: #f4f2ed; border-radius: var(--radius); margin-bottom: 12px; }
.upload-mode-row label { display: flex; align-items: center; gap: 7px; cursor: pointer; font-size: 13px; font-weight: 500; }

/* Corpus preview */
.corpus-preview { border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; margin-top: 10px; }
.corpus-preview-header { padding: 8px 12px; background: #f4f2ed; border-bottom: 1px solid var(--border); font-size: 12px; font-weight: 600; display: flex; align-items: center; gap: 8px; }
.corpus-preview-pair { display: flex; align-items: center; gap: 8px; padding: 5px 12px; border-bottom: 1px solid var(--border); font-size: 12px; font-family: monospace; }
.corpus-preview-pair:last-child { border-bottom: none; }
.corpus-preview-more { padding: 6px 12px; font-size: 11px; color: var(--text-muted); background: #fafaf8; }

/* Uploaded corpus list */
.upload-corpus-item { display: flex; align-items: center; gap: 10px; padding: 8px 12px; border: 1px solid var(--border); border-radius: var(--radius); margin-bottom: 6px; background: #fff; cursor: pointer; transition: border-color 0.15s; }
.upload-corpus-item:hover { border-color: var(--accent); background: #f8f7ff; }
.upload-corpus-item.selected { border-color: var(--accent); background: #eef2fc; }
.upload-corpus-label { flex: 1; font-size: 13px; }

/* Provider rows (OCR/LLM status sections) */
.provider-row { display: flex; align-items: center; gap: 10px; padding: 7px 10px; border: 1px solid var(--border); border-radius: var(--radius); margin-bottom: 6px; background: #fff; }
.provider-label { min-width: 200px; display: flex; align-items: center; gap: 8px; font-size: 13px; font-weight: 500; }
.provider-status { font-size: 11px; color: var(--text-muted); min-width: 80px; }
.provider-model-select { flex: 1; font-size: 12px; color: var(--text-muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* Competitor composer */
.mode-toggle { display: flex; gap: 20px; padding: 10px 14px; background: #f4f2ed; border-radius: var(--radius); margin-bottom: 12px; }
.mode-toggle label { display: flex; align-items: center; gap: 7px; cursor: pointer; font-size: 13px; font-weight: 500; }
.composer-row { display: flex; gap: 10px; flex-wrap: wrap; align-items: flex-end; margin-bottom: 10px; }
.composer-row .form-group { min-width: 150px; }

/* Competitor cards */
.competitor-card { display: flex; align-items: center; justify-content: space-between; padding: 9px 14px; border: 1px solid var(--border); border-radius: var(--radius); margin-bottom: 7px; background: #fff; gap: 10px; }
.competitor-card:hover { border-color: var(--accent); background: #f8f7ff; }
.competitor-info { display: flex; align-items: center; gap: 10px; flex: 1; min-width: 0; }
.competitor-badge { font-size: 11px; background: #eef2fc; color: var(--accent); padding: 2px 8px; border-radius: 10px; white-space: nowrap; flex-shrink: 0; }
.competitor-name { font-size: 13px; font-weight: 500; }
.competitor-detail { font-size: 11px; color: var(--text-muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
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

      <!-- Tab bar -->
      <div class="corpus-tabs">
        <button class="corpus-tab active" id="ctab-browse" onclick="switchCorpusTab('browse')" data-i18n="corpus_tab_browse">📁 Parcourir</button>
        <button class="corpus-tab" id="ctab-upload" onclick="switchCorpusTab('upload')" data-i18n="corpus_tab_upload">⬆ Uploader</button>
      </div>

      <!-- Browse tab -->
      <div id="corpus-tab-browse">
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
      </div>

      <!-- Upload tab -->
      <div id="corpus-tab-upload" style="display:none;">
        <div class="upload-mode-row">
          <label><input type="radio" name="upload-mode" value="zip" checked onchange="onUploadModeChange()"> 🗜 <span data-i18n="upload_zip_mode">Archive ZIP</span></label>
          <label><input type="radio" name="upload-mode" value="files" onchange="onUploadModeChange()"> 🖼 <span data-i18n="upload_files_mode">Fichiers individuels</span></label>
        </div>
        <!-- Drop zone -->
        <div id="upload-dropzone" class="upload-dropzone"
             onclick="document.getElementById('upload-file-input').click()"
             ondragover="event.preventDefault(); this.classList.add('dragover')"
             ondragleave="this.classList.remove('dragover')"
             ondrop="onDropFiles(event)">
          <span class="upload-icon">⬆</span>
          <span id="upload-dropzone-text" data-i18n="upload_drop_zip">Glissez un .zip ici ou cliquez pour sélectionner</span>
          <input type="file" id="upload-file-input" style="display:none" accept=".zip" onchange="onFileInputChange(event)" />
        </div>
        <!-- Progress -->
        <div id="upload-progress-container" style="display:none; margin-top:10px;">
          <div class="progress-bar-outer">
            <div class="progress-bar-inner" id="upload-progress-bar" style="width:0%; transition:width 0.2s;"></div>
          </div>
          <div id="upload-progress-text" style="font-size:12px; color:var(--text-muted); margin-top:4px;"></div>
        </div>
        <!-- Preview after upload -->
        <div id="upload-preview" style="margin-top:10px;"></div>
        <!-- Previously uploaded corpora -->
        <div id="uploads-list" style="margin-top:14px;"></div>
      </div>

      <div id="corpus-info" style="margin-top:8px; font-size:12px; color: var(--text-muted);"></div>
    </div>

    <!-- ── Section 1 : Moteurs OCR ─────────────────────────────────── -->
    <div class="card">
      <h2 data-i18n="bench_ocr_title">2. Moteurs OCR</h2>
      <div id="ocr-engines-status-list">
        <div style="color: var(--text-muted); font-size: 12px;"><span class="spinner"></span> Chargement…</div>
      </div>
    </div>

    <!-- ── Section 2 : Modèles LLM ──────────────────────────────────── -->
    <div class="card">
      <h2 data-i18n="bench_llm_title">3. Modèles LLM</h2>
      <div id="llm-status-list">
        <div style="color: var(--text-muted); font-size: 12px;"><span class="spinner"></span> Chargement…</div>
      </div>
    </div>

    <!-- ── Section 3 : Composition des concurrents ──────────────────── -->
    <div class="card">
      <h2 data-i18n="bench_compose_title">4. Concurrents à benchmarker</h2>

      <div class="mode-toggle">
        <label><input type="radio" name="compose-mode" value="ocr" checked onchange="onComposeModeChange()"> 🔍 <span data-i18n="compose_ocr_only">OCR seul</span></label>
        <label><input type="radio" name="compose-mode" value="pipeline" onchange="onComposeModeChange()"> ⛓ <span data-i18n="compose_pipeline">Pipeline OCR+LLM</span></label>
      </div>

      <div class="composer-row">
        <div class="form-group">
          <label data-i18n="compose_ocr_engine">Moteur OCR</label>
          <select id="compose-ocr-engine" onchange="onComposeOCRChange()">
            <option value="tesseract">Tesseract</option>
            <option value="mistral_ocr">Mistral OCR</option>
            <option value="google_vision">Google Vision</option>
            <option value="azure_doc_intel">Azure Doc Intel</option>
          </select>
        </div>
        <div class="form-group" style="flex:1;">
          <label data-i18n="compose_ocr_model">Modèle / Langue <span class="spinner" id="sp-ocr-model" style="display:none"></span></label>
          <select id="compose-ocr-model"></select>
        </div>
      </div>

      <div id="compose-pipeline-section" style="display:none;">
        <div class="composer-row">
          <div class="form-group">
            <label data-i18n="compose_llm_provider">Provider LLM</label>
            <select id="compose-llm-provider" onchange="onComposeLLMChange()">
              <option value="openai">OpenAI</option>
              <option value="anthropic">Anthropic</option>
              <option value="mistral">Mistral LLM</option>
              <option value="ollama">Ollama</option>
            </select>
          </div>
          <div class="form-group" style="flex:1;">
            <label data-i18n="compose_llm_model">Modèle LLM <span class="spinner" id="sp-llm-model" style="display:none"></span></label>
            <select id="compose-llm-model"></select>
          </div>
        </div>
        <div class="composer-row">
          <div class="form-group">
            <label data-i18n="compose_mode">Mode pipeline</label>
            <select id="compose-pipeline-mode">
              <option value="text_only" data-i18n="mode_text_only">Post-correction texte</option>
              <option value="text_and_image" data-i18n="mode_text_image">Post-correction image+texte</option>
              <option value="zero_shot" data-i18n="mode_zero_shot">Zero-shot</option>
            </select>
          </div>
          <div class="form-group" style="flex:1;">
            <label data-i18n="compose_prompt">Prompt <span class="spinner" id="sp-prompt" style="display:none"></span></label>
            <select id="compose-prompt"></select>
          </div>
        </div>
      </div>

      <div style="display:flex; gap:10px; align-items:center; margin-top:10px;">
        <button class="btn btn-primary btn-sm" onclick="addCompetitor()" data-i18n="compose_add">+ Ajouter</button>
        <span id="compose-error" style="color: var(--danger); font-size:12px;"></span>
      </div>

      <div id="competitors-list" style="margin-top:14px;">
        <div style="color: var(--text-muted); font-size:12px;" data-i18n="compose_empty">Aucun concurrent ajouté.</div>
      </div>
    </div>

    <!-- ── 5. Options ─────────────────────────────────────────────────── -->
    <div class="card">
      <h2 data-i18n="bench_options_title">5. Options</h2>
      <div class="form-row">
        <div class="form-group">
          <label data-i18n="bench_norm_label">Profil de normalisation</label>
          <select id="norm-profile">
            <option value="nfc">NFC (standard)</option>
          </select>
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
    corpus_tab_browse: "📁 Parcourir",
    corpus_tab_upload: "⬆ Uploader",
    upload_zip_mode: "Archive ZIP",
    upload_files_mode: "Fichiers individuels",
    upload_drop_zip: "Glissez un .zip ici ou cliquez pour sélectionner",
    upload_drop_files: "Glissez des images + .gt.txt ou cliquez pour sélectionner",
    upload_uploading: "Upload en cours…",
    upload_success: "Corpus chargé avec succès",
    upload_no_corpus: "Aucun corpus uploadé.",
    upload_select: "Utiliser ce corpus",
    upload_delete: "Supprimer",
    upload_pairs: "paires",
    upload_missing_gt: "GT manquant(s)",
    bench_engines_title: "2. Moteurs et pipelines",
    bench_ocr_title: "2. Moteurs OCR",
    bench_llm_title: "3. Modèles LLM",
    bench_compose_title: "4. Concurrents à benchmarker",
    bench_options_title: "5. Options",
    compose_ocr_only: "OCR seul",
    compose_pipeline: "Pipeline OCR+LLM",
    compose_ocr_engine: "Moteur OCR",
    compose_ocr_model: "Modèle / Langue",
    compose_llm_provider: "Provider LLM",
    compose_llm_model: "Modèle LLM",
    compose_mode: "Mode pipeline",
    compose_prompt: "Prompt",
    compose_add: "+ Ajouter",
    compose_empty: "Aucun concurrent ajouté.",
    mode_text_only: "Post-correction texte",
    mode_text_image: "Post-correction image+texte",
    mode_zero_shot: "Zero-shot",
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
    corpus_tab_browse: "📁 Browse",
    corpus_tab_upload: "⬆ Upload",
    upload_zip_mode: "ZIP archive",
    upload_files_mode: "Individual files",
    upload_drop_zip: "Drop a .zip here or click to select",
    upload_drop_files: "Drop images + .gt.txt files or click to select",
    upload_uploading: "Uploading…",
    upload_success: "Corpus loaded successfully",
    upload_no_corpus: "No corpus uploaded.",
    upload_select: "Use this corpus",
    upload_delete: "Delete",
    upload_pairs: "pairs",
    upload_missing_gt: "missing GT",
    bench_engines_title: "2. Engines & pipelines",
    bench_ocr_title: "2. OCR Engines",
    bench_llm_title: "3. LLM Models",
    bench_compose_title: "4. Competitors",
    bench_options_title: "5. Options",
    compose_ocr_only: "OCR only",
    compose_pipeline: "OCR+LLM Pipeline",
    compose_ocr_engine: "OCR Engine",
    compose_ocr_model: "Model / Language",
    compose_llm_provider: "LLM Provider",
    compose_llm_model: "LLM Model",
    compose_mode: "Pipeline mode",
    compose_prompt: "Prompt",
    compose_add: "+ Add",
    compose_empty: "No competitors added.",
    mode_text_only: "Text post-correction",
    mode_text_image: "Image+text post-correction",
    mode_zero_shot: "Zero-shot",
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

// ─── Models cache & fetching ─────────────────────────────────────────────────
let _modelsCache = {};
let _enginesData = null;
let _competitors = [];
let _refreshIntervalId = null;

async function fetchModels(provider) {
  if (_modelsCache[provider]) return _modelsCache[provider];
  const r = await fetch(`/api/models/${provider}`);
  const d = await r.json();
  const models = d.models || [];
  _modelsCache[provider] = models;
  return models;
}

function populateSelect(selectId, models, spinnerId) {
  const sel = document.getElementById(selectId);
  if (spinnerId) { const sp = document.getElementById(spinnerId); if (sp) sp.style.display = "none"; }
  if (!sel) return;
  sel.innerHTML = models.length === 0
    ? '<option value="">— aucun modèle —</option>'
    : models.map(m => `<option value="${m}">${m}</option>`).join("");
}

// ─── Benchmark sections (OCR + LLM status + composer init) ───────────────────
async function loadBenchmarkSections() {
  try {
    const r = await fetch("/api/engines");
    const d = await r.json();
    _enginesData = d;
    renderOCREnginesSection(d.engines);
    renderLLMSection(d.llms);
  } catch(e) {
    document.getElementById("ocr-engines-status-list").innerHTML =
      `<div style="color:var(--danger);font-size:12px;">Erreur : ${e.message}</div>`;
  }
}

function _makeProviderRow(eng, msId) {
  const dotCls = eng.available ? "status-ok" : (eng.status === "not_running" ? "status-warn" : "status-err");
  let statusLabel;
  if (eng.available) statusLabel = eng.version ? eng.version : (lang === "fr" ? "disponible" : "available");
  else if (eng.status === "missing_key") statusLabel = eng.key_env ? `<code style="font-size:11px;color:var(--warning)">${eng.key_env}</code>` : (lang === "fr" ? "clé manquante" : "key missing");
  else if (eng.status === "not_running") statusLabel = lang === "fr" ? "inactif" : "not running";
  else statusLabel = lang === "fr" ? "non installé" : "not installed";

  const row = document.createElement("div");
  row.className = "provider-row";
  row.innerHTML = `
    <div class="provider-label"><span class="engine-status ${dotCls}"></span><strong>${eng.label}</strong></div>
    <div class="provider-status">${statusLabel}</div>
    <div class="provider-model-select" id="${msId}">${eng.available ? '<span class="spinner"></span>' : ""}</div>`;
  return row;
}

async function renderOCREnginesSection(engines) {
  const container = document.getElementById("ocr-engines-status-list");
  container.innerHTML = "";
  for (const eng of engines) {
    const msId = `ms-ocr-${eng.id}`;
    container.appendChild(_makeProviderRow(eng, msId));
    if (eng.available) {
      fetchModels(eng.id).then(models => {
        const div = document.getElementById(msId);
        if (!div) return;
        div.innerHTML = models.length === 0
          ? `<span style="color:var(--text-muted);font-size:11px;">—</span>`
          : `<span style="font-size:12px;">${models.slice(0,5).join(", ")}${models.length > 5 ? ` +${models.length-5}` : ""}</span>`;
      }).catch(() => {
        const div = document.getElementById(msId);
        if (div) div.innerHTML = `<span style="color:var(--danger);font-size:11px;">Erreur API</span>`;
      });
    }
  }
}

async function renderLLMSection(llms) {
  const container = document.getElementById("llm-status-list");
  container.innerHTML = "";
  for (const llm of llms) {
    const msId = `ms-llm-${llm.id}`;
    container.appendChild(_makeProviderRow(llm, msId));
    if (llm.available) {
      fetchModels(llm.id).then(models => {
        const div = document.getElementById(msId);
        if (!div) return;
        div.innerHTML = models.length === 0
          ? `<span style="color:var(--text-muted);font-size:11px;">—</span>`
          : `<span style="font-size:12px;">${models.slice(0,3).join(", ")}${models.length > 3 ? ` +${models.length-3}` : ""}</span>`;
      }).catch(() => {
        const div = document.getElementById(msId);
        if (div) div.innerHTML = `<span style="color:var(--danger);font-size:11px;">Erreur API</span>`;
      });
    }
  }
}

function startAutoRefresh() {
  if (_refreshIntervalId) clearInterval(_refreshIntervalId);
  _refreshIntervalId = setInterval(async () => {
    try {
      const r = await fetch("/api/engines");
      const d = await r.json();
      if (!_enginesData || JSON.stringify(d) !== JSON.stringify(_enginesData)) {
        _modelsCache = {};
        _enginesData = d;
        renderOCREnginesSection(d.engines);
        renderLLMSection(d.llms);
      }
    } catch(e) {}
  }, 10000);
}

// ─── Competitor composer ──────────────────────────────────────────────────────
async function onComposeOCRChange() {
  const engine = document.getElementById("compose-ocr-engine").value;
  const sp = document.getElementById("sp-ocr-model");
  sp.style.display = "inline-block";
  try {
    const models = await fetchModels(engine);
    populateSelect("compose-ocr-model", models, "sp-ocr-model");
  } catch(e) {
    sp.style.display = "none";
    document.getElementById("compose-ocr-model").innerHTML = '<option value="">Erreur</option>';
  }
}

async function onComposeLLMChange() {
  const provider = document.getElementById("compose-llm-provider").value;
  const sp = document.getElementById("sp-llm-model");
  sp.style.display = "inline-block";
  try {
    const models = await fetchModels(provider);
    populateSelect("compose-llm-model", models, "sp-llm-model");
  } catch(e) {
    sp.style.display = "none";
    document.getElementById("compose-llm-model").innerHTML = '<option value="">Erreur</option>';
  }
}

function onComposeModeChange() {
  const mode = document.querySelector("input[name=compose-mode]:checked").value;
  document.getElementById("compose-pipeline-section").style.display =
    mode === "pipeline" ? "block" : "none";
}

async function loadComposePrompts() {
  document.getElementById("sp-prompt").style.display = "inline-block";
  try {
    const models = await fetchModels("prompts");
    populateSelect("compose-prompt", models, "sp-prompt");
  } catch(e) {
    document.getElementById("sp-prompt").style.display = "none";
  }
}

function addCompetitor() {
  const ocrEngine = document.getElementById("compose-ocr-engine").value;
  const ocrModel = document.getElementById("compose-ocr-model").value;
  const mode = document.querySelector("input[name=compose-mode]:checked").value;
  const errEl = document.getElementById("compose-error");

  if (!ocrEngine) {
    errEl.textContent = lang === "fr" ? "Sélectionnez un moteur OCR." : "Select an OCR engine.";
    return;
  }

  const comp = { name: "", ocr_engine: ocrEngine, ocr_model: ocrModel,
                  llm_provider: "", llm_model: "", pipeline_mode: "", prompt_file: "" };

  if (mode === "pipeline") {
    comp.llm_provider = document.getElementById("compose-llm-provider").value;
    comp.llm_model = document.getElementById("compose-llm-model").value;
    comp.pipeline_mode = document.getElementById("compose-pipeline-mode").value;
    comp.prompt_file = document.getElementById("compose-prompt").value;
    if (!comp.llm_provider) {
      errEl.textContent = lang === "fr" ? "Sélectionnez un provider LLM." : "Select an LLM provider.";
      return;
    }
    comp.name = `${ocrEngine}${ocrModel ? ":"+ocrModel : ""} → ${comp.llm_provider}${comp.llm_model ? ":"+comp.llm_model : ""}`;
  } else {
    comp.name = `${ocrEngine}${ocrModel ? " ("+ocrModel+")" : ""}`;
  }

  errEl.textContent = "";
  _competitors.push(comp);
  renderCompetitors();
}

function removeCompetitor(idx) {
  _competitors.splice(idx, 1);
  renderCompetitors();
}

function renderCompetitors() {
  const container = document.getElementById("competitors-list");
  if (_competitors.length === 0) {
    container.innerHTML = `<div style="color:var(--text-muted);font-size:12px;">${t("compose_empty")}</div>`;
    return;
  }
  container.innerHTML = _competitors.map((c, i) => {
    const isPipeline = !!c.llm_provider;
    const badge = isPipeline ? "⛓ Pipeline" : "🔍 OCR";
    const detail = isPipeline
      ? `${c.ocr_engine}:${c.ocr_model} → ${c.llm_provider}:${c.llm_model} [${c.pipeline_mode}]`
      : `${c.ocr_engine}:${c.ocr_model}`;
    return `<div class="competitor-card">
      <div class="competitor-info">
        <span class="competitor-badge">${badge}</span>
        <span class="competitor-name">${c.name}</span>
        <span class="competitor-detail">${detail}</span>
      </div>
      <button class="btn btn-danger btn-sm" onclick="removeCompetitor(${i})">✕</button>
    </div>`;
  }).join("");
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
  if (_competitors.length === 0) {
    alert(lang === "fr" ? "Ajoutez au moins un concurrent (Section 4)." : "Add at least one competitor (Section 4).");
    return;
  }

  const payload = {
    corpus_path: corpusPath,
    competitors: _competitors,
    normalization_profile: document.getElementById("norm-profile").value,
    output_dir: document.getElementById("output-dir").value,
    report_name: document.getElementById("report-name").value,
  };

  document.getElementById("start-btn").disabled = true;
  document.getElementById("cancel-btn").style.display = "inline-flex";
  document.getElementById("bench-progress-section").style.display = "block";
  document.getElementById("bench-result-section").style.display = "none";
  document.getElementById("bench-log").textContent = "";
  document.getElementById("engine-progress-list").innerHTML = "";
  document.getElementById("bench-status-text").textContent = lang === "fr" ? "Démarrage…" : "Starting…";

  try {
    const r = await fetch("/api/benchmark/run", {
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
    _startSSE(_currentJobId);
  } catch(e) {
    appendLog(`Erreur : ${e.message}`, "error");
    document.getElementById("start-btn").disabled = false;
    document.getElementById("cancel-btn").style.display = "none";
    document.getElementById("bench-status-text").textContent = "";
  }
}

function _startSSE(jobId) {
  if (_eventSource) _eventSource.close();
  const pl = document.getElementById("engine-progress-list");
  pl.innerHTML = "";
  const seenEngines = {};

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
    const engId = d.engine.replace(/[^a-z0-9_-]/gi, "_");
    if (!seenEngines[engId]) {
      seenEngines[engId] = true;
      const div = document.createElement("div");
      div.style = "margin-bottom: 8px;";
      div.innerHTML = `<div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px;">
        <span>${d.engine}</span><span id="eng-pct-${engId}">0%</span></div>
        <div class="progress-bar-outer"><div class="progress-bar-inner" id="eng-bar-${engId}" style="width:0%"></div></div>`;
      pl.appendChild(div);
    }
    const bar = document.getElementById(`eng-bar-${engId}`);
    const pctEl = document.getElementById(`eng-pct-${engId}`);
    if (bar) bar.style.width = pct + "%";
    if (pctEl) pctEl.textContent = pct + "%";
    document.getElementById("bench-status-text").textContent =
      `${pct}% — ${d.engine} (${d.processed}/${d.total})`;
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

  _eventSource.addEventListener("done", e => { _finishBenchmark(); });
  _eventSource.onerror = () => { if (_currentJobId) _finishBenchmark(); };
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

// ─── Corpus upload ────────────────────────────────────────────────────────────
let _uploadMode = "zip";  // "zip" | "files"

function switchCorpusTab(tab) {
  document.getElementById("corpus-tab-browse").style.display = tab === "browse" ? "block" : "none";
  document.getElementById("corpus-tab-upload").style.display = tab === "upload" ? "block" : "none";
  document.getElementById("ctab-browse").classList.toggle("active", tab === "browse");
  document.getElementById("ctab-upload").classList.toggle("active", tab === "upload");
  if (tab === "upload") loadUploadedCorpora();
}

function onUploadModeChange() {
  _uploadMode = document.querySelector("input[name=upload-mode]:checked").value;
  const input = document.getElementById("upload-file-input");
  if (_uploadMode === "zip") {
    input.accept = ".zip";
    input.multiple = false;
    document.getElementById("upload-dropzone-text").textContent = t("upload_drop_zip");
  } else {
    input.accept = ".jpg,.jpeg,.png,.tif,.tiff,.webp,.gt.txt,.txt";
    input.multiple = true;
    document.getElementById("upload-dropzone-text").textContent = t("upload_drop_files");
  }
}

function onFileInputChange(event) {
  const files = Array.from(event.target.files);
  if (files.length > 0) uploadCorpus(files);
}

function onDropFiles(event) {
  event.preventDefault();
  document.getElementById("upload-dropzone").classList.remove("dragover");
  const files = Array.from(event.dataTransfer.files);
  if (files.length > 0) uploadCorpus(files);
}

async function uploadCorpus(files) {
  const progressContainer = document.getElementById("upload-progress-container");
  const progressBar = document.getElementById("upload-progress-bar");
  const progressText = document.getElementById("upload-progress-text");
  const previewEl = document.getElementById("upload-preview");

  progressContainer.style.display = "block";
  progressBar.style.width = "10%";
  progressText.textContent = t("upload_uploading");
  previewEl.innerHTML = "";

  const fd = new FormData();
  for (const f of files) fd.append("files", f);

  try {
    // Simulate progress during upload
    let pct = 10;
    const timer = setInterval(() => {
      pct = Math.min(pct + 5, 85);
      progressBar.style.width = pct + "%";
    }, 200);

    const r = await fetch("/api/corpus/upload", {method: "POST", body: fd});
    clearInterval(timer);
    progressBar.style.width = "100%";

    if (!r.ok) {
      const err = await r.json();
      throw new Error(err.detail || "Erreur serveur");
    }
    const d = await r.json();
    progressText.textContent = `✓ ${t("upload_success")} — ${d.doc_count} ${t("upload_pairs")}`;
    progressBar.style.background = "var(--success)";

    // Show preview
    renderUploadPreview(d, previewEl);

    // Set corpus path and auto-select
    setCorpusPath(d.corpus_path, `upload:${d.corpus_id} (${d.doc_count} docs)`);

    // Refresh list
    loadUploadedCorpora();
  } catch(e) {
    progressBar.style.width = "100%";
    progressBar.style.background = "var(--danger)";
    progressText.textContent = `✗ ${e.message}`;
  }
}

function renderUploadPreview(data, container) {
  const missingBadge = data.has_missing_gt
    ? `<span class="badge badge-err" style="margin-left:8px;">${data.missing_gt.length} ${t("upload_missing_gt")}</span>`
    : "";
  let html = `<div class="corpus-preview">
    <div class="corpus-preview-header">
      <span>📄 ${data.doc_count} ${t("upload_pairs")}</span>${missingBadge}
    </div>`;
  for (const p of data.pairs) {
    html += `<div class="corpus-preview-pair">
      <span style="color:var(--text-muted);">🖼</span><span>${p.image}</span>
      <span style="color:var(--text-muted); margin-left:auto;">↔</span>
      <span style="color:var(--success);">${p.gt}</span>
    </div>`;
  }
  if (data.total_pairs > data.pairs.length) {
    html += `<div class="corpus-preview-more">… et ${data.total_pairs - data.pairs.length} autres paires</div>`;
  }
  for (const w of (data.warnings || [])) {
    html += `<div style="padding:5px 12px; font-size:11px; color:var(--warning);">⚠ ${w}</div>`;
  }
  html += `</div>`;
  container.innerHTML = html;
}

function setCorpusPath(path, label) {
  document.getElementById("corpus-path").value = path;
  document.getElementById("corpus-info").textContent = `✓ ${label}`;
}

async function loadUploadedCorpora() {
  const container = document.getElementById("uploads-list");
  try {
    const r = await fetch("/api/corpus/uploads");
    const d = await r.json();
    if (d.uploads.length === 0) {
      container.innerHTML = `<div style="color:var(--text-muted); font-size:12px;">${t("upload_no_corpus")}</div>`;
      return;
    }
    const currentPath = document.getElementById("corpus-path").value;
    container.innerHTML = d.uploads.map(u => {
      const isSelected = u.corpus_path === currentPath;
      const missing = u.has_missing_gt
        ? `<span class="badge badge-warn" style="margin-left:6px;">${t("upload_missing_gt")}</span>` : "";
      return `<div class="upload-corpus-item${isSelected ? " selected" : ""}"
                   onclick="setCorpusPath('${u.corpus_path}', 'upload (${u.doc_count} docs)'); loadUploadedCorpora()">
        <span class="upload-corpus-label">
          <strong>${u.doc_count} ${t("upload_pairs")}</strong>${missing}
          <span style="display:block; font-size:11px; color:var(--text-muted); font-family:monospace;">${u.corpus_path}</span>
        </span>
        <button class="btn btn-danger btn-sm" onclick="event.stopPropagation(); deleteUploadedCorpus('${u.corpus_id}')"
                title="${t("upload_delete")}">✕</button>
      </div>`;
    }).join("");
  } catch(e) {
    container.innerHTML = `<div style="color:var(--danger); font-size:12px;">Erreur : ${e.message}</div>`;
  }
}

async function deleteUploadedCorpus(corpusId) {
  try {
    await fetch(`/api/corpus/uploads/${corpusId}`, {method: "DELETE"});
    loadUploadedCorpora();
    // Clear corpus path if it was the deleted one
    const p = document.getElementById("corpus-path").value;
    if (p.includes(corpusId)) {
      document.getElementById("corpus-path").value = "";
      document.getElementById("corpus-info").textContent = "";
    }
  } catch(e) {}
}

// ─── Init ────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  loadStatus();
  loadNormProfiles();
  initHTRFilters();
  // Load OCR engines, LLM models, initialize composer
  await loadBenchmarkSections();
  onComposeOCRChange();      // Pre-populate Tesseract languages
  loadComposePrompts();       // Pre-load prompt files
  startAutoRefresh();         // Auto-detect new API keys every 10 s
  // Close modal on backdrop click
  document.getElementById("import-modal").addEventListener("click", e => {
    if (e.target === document.getElementById("import-modal")) closeImportModal();
  });
});
</script>
</body>
</html>"""
