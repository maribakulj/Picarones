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
import logging
import os
import shutil
import threading
import uuid
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from fastapi import Cookie, FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from picarones import __version__
from picarones.web.security import (
    RateLimiter,
    assert_engines_allowed,
    assert_llm_provider_allowed,
    compute_browse_roots,
    csp_middleware,
    get_max_concurrent_jobs,
    get_rate_limit_per_hour,
    validate_image_safe,
)

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

# Sprint 24 — middleware CSP + en-têtes durcis (X-Frame-Options, etc.)
app.middleware("http")(csp_middleware)

# Fichiers statiques (CSS, icônes…)
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# Sprint 24 — rate limiter global (no-op si non public ou quota = 0)
_RATE_LIMITER = RateLimiter(max_per_hour=get_rate_limit_per_hour())

# Sprint 24 — sémaphore borné le nombre de benchmarks concurrents.
_JOBS_SEMAPHORE = threading.Semaphore(get_max_concurrent_jobs())


def _client_ip(request: Request) -> str:
    """Récupère l'IP client en respectant ``X-Forwarded-For`` derrière un proxy."""
    fwd = request.headers.get("x-forwarded-for") or ""
    if fwd:
        return fwd.split(",")[0].strip()
    return (request.client.host if request.client else "unknown")


def _enforce_rate_limit(request: Request) -> None:
    """Applique le rate limit ; lève HTTPException 429 si dépassé."""
    try:
        _RATE_LIMITER.check(_client_ip(request))
    except PermissionError as exc:
        raise HTTPException(status_code=429, detail=str(exc))

# ---------------------------------------------------------------------------
# Job management
# ---------------------------------------------------------------------------

_logger = logging.getLogger(__name__)


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
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _cancel_event: threading.Event = field(default_factory=threading.Event)

    def add_event(self, kind: str, data: Any) -> None:
        event = {"kind": kind, "data": data, "ts": _iso_now()}
        with self._lock:
            self.events.append(event)
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

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


_JOBS: dict[str, BenchmarkJob] = {}
_JOBS_MAX = 100  # Nombre max de jobs conservés en mémoire
_JOBS_LOCK = threading.Lock()


def _cleanup_old_jobs() -> None:
    """Supprime les jobs terminés les plus anciens si le nombre dépasse _JOBS_MAX."""
    with _JOBS_LOCK:
        if len(_JOBS) <= _JOBS_MAX:
            return
        finished = [
            (jid, j) for jid, j in _JOBS.items()
            if j.status in ("complete", "error", "cancelled")
        ]
        finished.sort(key=lambda x: x[1].finished_at or "")
        to_remove = len(_JOBS) - _JOBS_MAX
        for jid, _ in finished[:to_remove]:
            del _JOBS[jid]


_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"})
_UPLOADS_DIR = Path("./uploads")

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class BenchmarkRequest(BaseModel):
    corpus_path: str
    engines: list[str] = ["tesseract"]
    normalization_profile: str = "nfc"
    char_exclude: str = ""   # Caractères à ignorer (séparés par virgule, ex: "',–")
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
    ocr_engine: str = ""
    """Moteur OCR : 'tesseract', 'mistral_ocr', ... ou 'corpus' pour utiliser l'OCR pré-calculé."""
    ocr_model: str = ""
    llm_provider: str = ""
    llm_model: str = ""
    pipeline_mode: str = ""
    prompt_file: str = ""


class BenchmarkRunRequest(BaseModel):
    corpus_path: str
    competitors: list[CompetitorConfig]
    normalization_profile: str = "nfc"
    char_exclude: str = ""   # Caractères à ignorer (séparés par virgule, ex: "',–")
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

    # Ollama (un seul appel HTTP)
    ollama_available, ollama_models = _fetch_ollama_info()
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


def _fetch_ollama_info() -> tuple[bool, list[str]]:
    """Vérifie la disponibilité d'Ollama et liste ses modèles en un seul appel HTTP."""
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as r:
            if r.status != 200:
                return False, []
            data = json.loads(r.read().decode())
        models = [m.get("name", "") for m in data.get("models", [])]
        return True, models
    except Exception:
        return False, []


def _check_ollama() -> bool:
    available, _ = _fetch_ollama_info()
    return available


def _list_ollama_models() -> list[str]:
    _, models = _fetch_ollama_info()
    return models


def _get_tesseract_langs() -> list[str]:
    try:
        import pytesseract
        langs = pytesseract.get_languages(config="")
        return sorted(lg for lg in langs if lg != "osd")
    except Exception:
        return ["fra", "lat", "eng", "deu", "ita", "spa"]


# ---------------------------------------------------------------------------
# API — models (dynamic per provider, with capability metadata)
# ---------------------------------------------------------------------------

# Modèles Mistral text-only (pas de support vision)
_MISTRAL_TEXT_ONLY = frozenset({
    "ministral-3b-latest", "ministral-8b-latest", "mistral-tiny",
    "mistral-tiny-latest", "open-mistral-7b", "open-mixtral-8x7b",
    "mistral-small-latest", "mistral-small-2409",
})

# Préfixes de modèles Mistral qui sont text-only (pas de support vision)
_MISTRAL_TEXT_ONLY_PREFIXES = (
    "ministral", "open-mistral", "open-mixtral", "codestral",
    "mistral-embed", "mistral-tiny",
)

# Familles Ollama multimodales connues
_OLLAMA_VISION_FAMILIES = frozenset({
    "llava", "bakllava", "moondream", "minicpm-v", "llama3.2-vision",
    "llava-llama3", "llava-phi3", "nanollava",
})


def _model_entry(model_id: str, capabilities: list[str]) -> dict:
    """Crée une entrée modèle avec son ID et ses capacités."""
    return {"id": model_id, "capabilities": capabilities}


def _infer_mistral_capabilities(model_id: str) -> list[str]:
    mid = model_id.lower()
    # Modèles explicitement vision (Pixtral)
    if "pixtral" in mid:
        return ["text", "vision"]
    # Modèles explicitement text-only
    if mid in _MISTRAL_TEXT_ONLY or any(mid.startswith(p) for p in _MISTRAL_TEXT_ONLY_PREFIXES):
        return ["text"]
    # Mistral Large et modèles récents non-identifiés → vision par défaut
    if "mistral-large" in mid or "mistral-medium" in mid:
        return ["text", "vision"]
    # Par défaut, marquer comme text-only (plus sûr que de supposer vision)
    return ["text"]


def _infer_openai_capabilities(model_id: str) -> list[str]:
    mid = model_id.lower()
    if "gpt-4o" in mid or "gpt-4-turbo" in mid or "gpt-4.1" in mid or "o1" in mid or "o3" in mid:
        return ["text", "vision"]
    return ["text"]


def _infer_ollama_capabilities(model_name: str) -> list[str]:
    base = model_name.split(":")[0].lower()
    if any(base.startswith(family) for family in _OLLAMA_VISION_FAMILIES):
        return ["text", "vision"]
    return ["text"]


@app.get("/api/models/{provider}")
async def api_models(
    provider: str,
    capability: str = Query(default="", description="Filtre par capacité : 'text', 'vision', ou vide pour tout"),
) -> dict:
    """Retourne les modèles disponibles avec leurs capacités (text, vision).

    Interroge l'API du provider en temps réel.  Les capacités sont déterminées
    par heuristique sur le nom du modèle quand l'API ne fournit pas cette
    information directement.

    Le paramètre ``capability`` filtre les résultats (ex : ``?capability=vision``
    ne retourne que les modèles supportant la vision).
    """
    import urllib.request as _urlreq

    def _fetch_json(url: str, headers: dict) -> dict:
        req = _urlreq.Request(url, headers=headers)
        with _urlreq.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())

    def _filter_and_format(models: list[dict]) -> dict:
        if capability:
            models = [m for m in models if capability in m["capabilities"]]
        return {
            "provider": provider,
            "models": models,
            "model_ids": [m["id"] for m in models],
        }

    if provider == "tesseract":
        langs = _get_tesseract_langs()
        return {"provider": provider, "models": langs, "model_ids": langs}

    if provider == "mistral_ocr":
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            return {"provider": provider, "models": [], "model_ids": [], "error": "MISTRAL_API_KEY non définie"}
        try:
            data = _fetch_json(
                "https://api.mistral.ai/v1/models",
                {"Authorization": f"Bearer {api_key}"},
            )
            models = [
                _model_entry(m["id"], _infer_mistral_capabilities(m["id"]))
                for m in data.get("data", [])
                if "pixtral" in m["id"].lower() or "mistral-ocr" in m["id"].lower()
            ]
            return _filter_and_format(sorted(models, key=lambda m: m["id"]))
        except Exception as exc:
            fallback = [
                _model_entry("pixtral-12b-2409", ["text", "vision"]),
                _model_entry("pixtral-large-latest", ["text", "vision"]),
                _model_entry("mistral-ocr-latest", ["text", "vision"]),
            ]
            return {**_filter_and_format(fallback), "error": str(exc)}

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return {"provider": provider, "models": [], "model_ids": [], "error": "OPENAI_API_KEY non définie"}
        try:
            data = _fetch_json(
                "https://api.openai.com/v1/models",
                {"Authorization": f"Bearer {api_key}"},
            )
            models = [
                _model_entry(m["id"], _infer_openai_capabilities(m["id"]))
                for m in data.get("data", [])
                if "gpt-4" in m["id"].lower() or "o1" in m["id"].lower() or "o3" in m["id"].lower()
            ]
            return _filter_and_format(sorted(models, key=lambda m: m["id"], reverse=True))
        except Exception as exc:
            fallback = [
                _model_entry("gpt-4o", ["text", "vision"]),
                _model_entry("gpt-4o-mini", ["text", "vision"]),
                _model_entry("gpt-4-turbo", ["text", "vision"]),
            ]
            return {**_filter_and_format(fallback), "error": str(exc)}

    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return {"provider": provider, "models": [], "model_ids": [], "error": "ANTHROPIC_API_KEY non définie"}
        try:
            data = _fetch_json(
                "https://api.anthropic.com/v1/models",
                {"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            )
            # Tous les modèles Claude 3+ supportent la vision
            models = [_model_entry(m["id"], ["text", "vision"]) for m in data.get("data", [])]
            return _filter_and_format(models)
        except Exception as exc:
            fallback = [
                _model_entry("claude-sonnet-4-6", ["text", "vision"]),
                _model_entry("claude-haiku-4-5-20251001", ["text", "vision"]),
                _model_entry("claude-opus-4-6", ["text", "vision"]),
            ]
            return {**_filter_and_format(fallback), "error": str(exc)}

    if provider == "mistral":
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            return {"provider": provider, "models": [], "model_ids": [], "error": "MISTRAL_API_KEY non définie"}
        try:
            data = _fetch_json(
                "https://api.mistral.ai/v1/models",
                {"Authorization": f"Bearer {api_key}"},
            )
            # Inclure TOUS les modèles Mistral (y compris Pixtral pour la vision)
            # sauf mistral-ocr qui est un endpoint OCR dédié, pas un LLM chat
            models = [
                _model_entry(m["id"], _infer_mistral_capabilities(m["id"]))
                for m in data.get("data", [])
                if "mistral-ocr" not in m["id"].lower()
            ]
            return _filter_and_format(sorted(models, key=lambda m: m["id"]))
        except Exception as exc:
            fallback = [
                _model_entry("mistral-large-latest", ["text", "vision"]),
                _model_entry("pixtral-large-latest", ["text", "vision"]),
                _model_entry("pixtral-12b-2409", ["text", "vision"]),
                _model_entry("mistral-small-latest", ["text"]),
            ]
            return {**_filter_and_format(fallback), "error": str(exc)}

    if provider == "ollama":
        _, model_names = _fetch_ollama_info()
        models = [
            _model_entry(name, _infer_ollama_capabilities(name))
            for name in model_names
        ]
        return _filter_and_format(models)

    if provider == "google_vision":
        models = [
            _model_entry("document_text_detection", ["vision"]),
            _model_entry("text_detection", ["vision"]),
        ]
        return _filter_and_format(models)

    if provider == "azure_doc_intel":
        models = [
            _model_entry("prebuilt-document", ["vision"]),
            _model_entry("prebuilt-read", ["vision"]),
        ]
        return _filter_and_format(models)

    if provider == "prompts":
        prompts_dir = Path(__file__).parent.parent / "prompts"
        if prompts_dir.exists():
            prompts = sorted(f.name for f in prompts_dir.glob("*.txt"))
        else:
            prompts = []
        return {"provider": provider, "models": prompts, "model_ids": prompts}

    raise HTTPException(status_code=404, detail=f"Provider inconnu : {provider}")


# ---------------------------------------------------------------------------
# API — corpus browse
# ---------------------------------------------------------------------------

# Sprint 24 — racines configurables via PICARONES_BROWSE_ROOTS, sinon
# défaut restreint en mode public, défaut historique en mode dev.
_BROWSE_ROOTS = compute_browse_roots(_UPLOADS_DIR)


def _is_path_allowed(target: Path) -> bool:
    """Vérifie qu'un chemin résolu est sous un des répertoires autorisés (cross-plateforme)."""
    for root in _BROWSE_ROOTS:
        try:
            if target == root or target.is_relative_to(root):
                return True
        except (ValueError, TypeError):
            continue
    return False


@app.get("/api/corpus/browse")
async def api_corpus_browse(path: str = Query(default=".", description="Chemin à explorer")) -> dict:
    target = Path(path).resolve()
    if not target.exists() or not target.is_dir():
        raise HTTPException(status_code=404, detail=f"Dossier non trouvé : {path}")
    # Sécurité : restreindre la navigation aux répertoires autorisés
    if not _is_path_allowed(target):
        raise HTTPException(status_code=403, detail="Accès refusé : chemin hors des répertoires autorisés")

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

def _safe_parse_xml(xml_bytes: bytes) -> Optional[ET.Element]:
    """Parse du XML en désactivant les entités externes (protection XXE)."""
    try:
        import defusedxml.ElementTree as SafeET
        return SafeET.fromstring(xml_bytes)
    except ImportError:
        pass
    # Fallback : parser standard avec entités externes désactivées
    parser = ET.XMLParser()
    try:
        return ET.fromstring(xml_bytes, parser=parser)
    except ET.ParseError:
        return None


def _detect_xml_gt(xml_bytes: bytes) -> tuple[str, str] | None:
    """Détecte si xml_bytes est un fichier ALTO ou PAGE XML et extrait le texte GT.

    Retourne (format_label, texte_gt) ou None si le format n'est pas reconnu.
    """
    root = _safe_parse_xml(xml_bytes)
    if root is None:
        return None

    tag = root.tag  # peut être "{namespace}alto" ou "alto" ou "{ns}PcGts"

    # --- ALTO XML ---
    # Namespace contient loc.gov/standards/alto ou balise racine "alto"
    ns_alto = "http://www.loc.gov/standards/alto"
    is_alto = (
        ns_alto in tag
        or tag.lower() == "alto"
        or (tag.startswith("{") and tag.split("}")[1].lower() in ("alto",))
    )
    if is_alto:
        text = _extract_alto_text(root)
        return ("ALTO XML", text)

    # --- PAGE XML ---
    # Balise racine PcGts (avec ou sans namespace)
    local = tag.split("}")[-1] if "}" in tag else tag
    if local == "PcGts":
        text = _extract_page_text(root)
        return ("PAGE XML", text)

    return None


def _extract_alto_text(root: ET.Element) -> str:
    """Extrait le texte plein d'un arbre ALTO XML.

    Concatène les attributs CONTENT des balises <String> dans l'ordre de lecture
    (bloc → ligne → mot), avec un espace entre mots et une newline entre lignes.
    """
    # Chercher les éléments TextLine (avec ou sans namespace)
    lines: list[str] = []
    for elem in root.iter():
        local = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if local == "TextLine":
            words: list[str] = []
            for child in elem.iter():
                child_local = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                if child_local == "String":
                    content = child.get("CONTENT", "")
                    if content:
                        words.append(content)
            if words:
                lines.append(" ".join(words))
    return "\n".join(lines)


def _extract_page_text(root: ET.Element) -> str:
    """Extrait le texte plein d'un arbre PAGE XML.

    Concatène le contenu des balises <Unicode> dans l'ordre de lecture.
    """
    texts: list[str] = []
    for elem in root.iter():
        local = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if local == "Unicode" and elem.text:
            texts.append(elem.text.strip())
    return "\n".join(t for t in texts if t)


def _analyze_corpus_dir(path: Path) -> dict:
    """Analyse un dossier et retourne un résumé des paires image/GT détectées."""
    # Exclure les fichiers cachés macOS (._* AppleDouble) et tout fichier débutant par .
    images = sorted(
        f.name for f in path.iterdir()
        if f.suffix.lower() in _IMAGE_EXTS and not f.name.startswith(".")
    )
    pairs: list[dict] = []
    missing_gt: list[str] = []
    for img in images:
        stem = Path(img).stem
        gt_txt = path / (stem + ".gt.txt")
        gt_xml = path / (stem + ".xml")
        if gt_txt.exists():
            pairs.append({"image": img, "gt": stem + ".gt.txt", "gt_format": "texte brut"})
        elif gt_xml.exists():
            result = _detect_xml_gt(gt_xml.read_bytes())
            if result is not None:
                fmt, text = result
                # Matérialiser le GT en .gt.txt pour le chargeur de corpus
                gt_txt.write_text(text, encoding="utf-8")
                pairs.append({"image": img, "gt": stem + ".gt.txt", "gt_format": fmt})
            else:
                missing_gt.append(img)
        else:
            missing_gt.append(img)

    # Détecter le format dominant pour le résumé global
    formats = {p["gt_format"] for p in pairs}
    if len(formats) == 1:
        dominant_format: str = formats.pop()
    elif formats:
        dominant_format = "mixte"
    else:
        dominant_format = "texte brut"

    # Détecter les fichiers OCR bruité (.ocr.txt) pour les corpus triplets
    ocr_text_count = sum(
        1 for p in pairs
        if (path / (Path(p["image"]).stem + ".ocr.txt")).exists()
    )

    return {
        "doc_count": len(pairs),
        "pairs": pairs[:20],
        "total_pairs": len(pairs),
        "missing_gt": missing_gt[:10],
        "has_missing_gt": len(missing_gt) > 0,
        "warnings": [f"GT manquant : {img}" for img in missing_gt[:5]],
        "usable": len(pairs) > 0,
        "gt_format": dominant_format,
        "has_ocr_text": ocr_text_count > 0,
        "ocr_text_count": ocr_text_count,
    }


_MAX_ZIP_TOTAL_SIZE = 500 * 1024 * 1024  # 500 Mo décompressé max
_MAX_ZIP_FILES = 2000  # nombre max de fichiers extraits


def _flatten_zip_to_dir(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extrait un ZIP en aplatissant les paires image/.gt.txt/.xml dans dest."""
    dest.mkdir(parents=True, exist_ok=True)
    total_size = 0
    file_count = 0
    for member in zf.infolist():
        if member.is_dir():
            continue
        p = Path(member.filename)
        name = p.name
        # Ignorer les fichiers cachés macOS (._* créés par AppleDouble dans les ZIPs)
        if name.startswith("."):
            continue
        # Accepter images, .gt.txt, .ocr.txt et .xml (ALTO/PAGE)
        if p.suffix.lower() in _IMAGE_EXTS or name.endswith(".gt.txt") or name.endswith(".ocr.txt") or p.suffix.lower() == ".xml":
            # Protection ZIP bomb : vérifier la taille décompressée
            total_size += member.file_size
            if total_size > _MAX_ZIP_TOTAL_SIZE:
                raise ValueError(
                    f"ZIP trop volumineux : taille décompressée > {_MAX_ZIP_TOTAL_SIZE // (1024*1024)} Mo"
                )
            file_count += 1
            if file_count > _MAX_ZIP_FILES:
                raise ValueError(f"ZIP contient trop de fichiers (> {_MAX_ZIP_FILES})")
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
            # Sprint 24 — empêcher la traversée via le nom de fichier reçu
            # depuis le client (multipart). On garde uniquement ``basename``.
            safe_name = Path(filename).name
            data = await uf.read()
            suffix = Path(safe_name).suffix.lower()

            if suffix == ".zip":
                # Extraire le ZIP en aplatissant les paires
                import io
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    _flatten_zip_to_dir(zf, corpus_dir)
            elif suffix in _IMAGE_EXTS:
                # Sprint 24 — valider l'image avant écriture (Pillow.verify,
                # taille max, rejet des bombes de décompression).
                try:
                    validate_image_safe(data, filename=safe_name)
                except ValueError as exc:
                    raise HTTPException(status_code=415, detail=str(exc))
                (corpus_dir / safe_name).write_bytes(data)
            elif safe_name.endswith(".gt.txt") or safe_name.endswith(".ocr.txt") or suffix in (".txt", ".xml"):
                (corpus_dir / safe_name).write_bytes(data)
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
        except Exception as e:
            _logger.warning(
                "[api_corpus_uploads] upload '%s' ignoré — inspection impossible : %s",
                d.name, e,
            )
    return {"uploads": uploads}


@app.get("/api/corpus/image/{upload_id}/{filename}")
async def api_corpus_image(upload_id: str, filename: str) -> FileResponse:
    """Sert une image depuis le dossier d'upload."""
    # Sécurité : interdire les path traversal
    if "/" in upload_id or "\\" in upload_id or ".." in upload_id:
        raise HTTPException(status_code=400, detail="upload_id invalide")
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="filename invalide")
    image_path = _UPLOADS_DIR / upload_id / filename
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image non trouvée")
    suffix = image_path.suffix.lower()
    media_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                   ".tif": "image/tiff", ".tiff": "image/tiff", ".webp": "image/webp"}
    media_type = media_types.get(suffix, "application/octet-stream")
    return FileResponse(str(image_path), media_type=media_type)


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
            "exclude_chars": sorted(p.exclude_chars),
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
    # Sécurité : interdire les path traversal
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Nom de fichier invalide")
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
    if not _JOBS_SEMAPHORE.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail=(
                f"Trop de benchmarks concurrents (max "
                f"{get_max_concurrent_jobs()}). Réessayer plus tard."
            ),
        )

    job_id = str(uuid.uuid4())
    job = BenchmarkJob(job_id=job_id)
    _JOBS[job_id] = job
    _cleanup_old_jobs()

    def _release_after(job_, fn, *args):
        try:
            fn(job_, *args)
        finally:
            _JOBS_SEMAPHORE.release()

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
    job._cancel_event.set()  # Signal d'annulation pour run_benchmark
    job.add_event("cancelled", {"message": "Benchmark annulé par l'utilisateur."})
    return {"job_id": job_id, "status": "cancelled"}


@app.get("/api/benchmark/{job_id}/stream")
async def api_benchmark_stream(job_id: str) -> StreamingResponse:
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job non trouvé : {job_id}")

    async def event_generator() -> AsyncIterator[str]:
        # S'abonner AVANT de lire les événements existants pour ne rien perdre
        queue = job.subscribe()
        try:
            # Envoie les événements déjà produits (snapshot thread-safe)
            with job._lock:
                past_events = list(job.events)
            for event in past_events:
                yield _sse_format(event["kind"], event["data"])

            if job.status in ("complete", "error", "cancelled"):
                yield _sse_format("done", {"status": job.status})
                return
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
    if not _JOBS_SEMAPHORE.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail=(
                f"Trop de benchmarks concurrents (max "
                f"{get_max_concurrent_jobs()}). Réessayer plus tard."
            ),
        )

    job_id = str(uuid.uuid4())
    job = BenchmarkJob(job_id=job_id)
    _JOBS[job_id] = job

    def _release_after(job_, fn, *args):
        try:
            fn(job_, *args)
        finally:
            _JOBS_SEMAPHORE.release()

    thread = threading.Thread(
        target=_release_after,
        args=(job, _run_benchmark_thread_v2, req),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "pending"}


def _build_llm_adapter(comp: CompetitorConfig) -> Any:
    """Instancie un adaptateur LLM depuis la config d'un concurrent."""
    if comp.llm_provider == "openai":
        from picarones.llm.openai_adapter import OpenAIAdapter
        return OpenAIAdapter(model=comp.llm_model or None)
    elif comp.llm_provider == "anthropic":
        from picarones.llm.anthropic_adapter import AnthropicAdapter
        return AnthropicAdapter(model=comp.llm_model or None)
    elif comp.llm_provider == "mistral":
        from picarones.llm.mistral_adapter import MistralAdapter
        return MistralAdapter(model=comp.llm_model or None)
    elif comp.llm_provider == "ollama":
        from picarones.llm.ollama_adapter import OllamaAdapter
        return OllamaAdapter(model=comp.llm_model or None)
    else:
        raise ValueError(f"Provider LLM inconnu : {comp.llm_provider}")


def _engine_from_competitor(comp: CompetitorConfig) -> Any:
    """Instancie un moteur OCR (ou pipeline OCR+LLM) depuis une CompetitorConfig.

    Modes supportés :
    - ``ocr_engine`` = 'tesseract', 'mistral_ocr', etc. → moteur OCR seul
    - ``ocr_engine`` + ``llm_provider`` → pipeline OCR live + LLM
    - ``ocr_engine`` = 'corpus' + ``llm_provider`` → post-correction LLM
      avec OCR pré-calculé (fichiers .ocr.txt du corpus triplet)
    - ``ocr_engine`` = '' + ``llm_provider`` → LLM seul (zero-shot ou post-correction)
    """
    engine_id = comp.ocr_engine

    # Pipeline post-correction avec OCR pré-calculé (corpus triplet)
    is_corpus_ocr = engine_id in ("corpus", "")

    if is_corpus_ocr and not comp.llm_provider:
        raise ValueError(
            "ocr_engine='corpus' nécessite un llm_provider "
            "(pour la post-correction ou le zero-shot)"
        )

    ocr = None
    if not is_corpus_ocr:
        from picarones.engines.tesseract import TesseractEngine
        from picarones.engines.mistral_ocr import MistralOCREngine

        if engine_id == "tesseract":
            ocr = TesseractEngine(config={"lang": comp.ocr_model or "fra", "psm": 6})
        elif engine_id == "mistral_ocr":
            ocr = MistralOCREngine(config={"model": comp.ocr_model or "mistral-ocr-latest"})
        elif engine_id == "google_vision":
            try:
                from picarones.engines.google_vision import GoogleVisionEngine
                ocr = GoogleVisionEngine(config={"detection_type": comp.ocr_model or "document_text_detection"})
            except ImportError as exc:
                raise RuntimeError("Google Vision non disponible.") from exc
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

    # Pipeline OCR+LLM (live ou post-correction)
    _mode_map = {
        "text_only": "text_only",
        "post_correction_text": "text_only",
        "text_and_image": "text_and_image",
        "post_correction_image": "text_and_image",
        "zero_shot": "zero_shot",
    }
    mode = _mode_map.get(comp.pipeline_mode, "text_only")

    llm = _build_llm_adapter(comp)

    from picarones.pipelines.base import OCRLLMPipeline
    prompt = comp.prompt_file or "correction_medieval_french.txt"

    if is_corpus_ocr:
        pipeline_name = comp.name or f"corpus_ocr → {comp.llm_model or comp.llm_provider}"
    else:
        pipeline_name = comp.name or f"{engine_id} → {comp.llm_model or comp.llm_provider}"

    return OCRLLMPipeline(
        ocr_engine=ocr,
        llm_adapter=llm,
        mode=mode,
        prompt=prompt,
        pipeline_name=pipeline_name,
    )


def _run_benchmark_thread_v2(job: BenchmarkJob, req: BenchmarkRunRequest) -> None:
    """Exécute un benchmark à partir d'une liste de CompetitorConfig."""

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

        from picarones.core.normalization import _parse_exclude_chars
        char_excl = _parse_exclude_chars(req.char_exclude) if req.char_exclude else None

        result = run_benchmark(
            corpus=corpus,
            engines=engines,
            output_json=output_json,
            show_progress=False,
            progress_callback=_progress_callback,
            char_exclude=char_excl,
            cancel_event=job._cancel_event,
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

        from picarones.core.normalization import _parse_exclude_chars
        char_excl = _parse_exclude_chars(req.char_exclude) if req.char_exclude else None

        # Lancer le benchmark
        result = run_benchmark(
            corpus=corpus,
            engines=ocr_engines,
            output_json=output_json,
            show_progress=False,
            progress_callback=_progress_callback,
            char_exclude=char_excl,
            cancel_event=job._cancel_event,
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

# Sprint 25 — environnement Jinja2 partagé pour la SPA.
# Le HTML/CSS/JS inline qui vivait dans ``_HTML_TEMPLATE`` (3000+ lignes
# de string Python) est maintenant découpé en :
#   - picarones/web/templates/  (base + 6 partials Jinja2)
#   - picarones/web/static/web-app.js  (toute la logique JS)
# Ce découpage permet :
#   1. de tester chaque vue indépendamment ;
#   2. de durcir la CSP à ``script-src 'self'`` (le JS n'est plus inline) ;
#   3. de toucher l'UI sans relire un fichier de 3000 lignes.
from jinja2 import Environment, FileSystemLoader, select_autoescape

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_jinja_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "j2"]),
    trim_blocks=False,
    lstrip_blocks=False,
)


def _render_index(lang: str) -> str:
    """Rend la SPA depuis ``base.html.j2``. Déterministe pour un même couple
    (lang, version) — utilisé par le test de non-régression Sprint 25."""
    return _jinja_env.get_template("base.html.j2").render(
        lang=lang,
        version=__version__,
    )


@app.get("/", response_class=HTMLResponse)
async def index(picarones_lang: str = Cookie(default="fr")) -> HTMLResponse:
    lang = picarones_lang if picarones_lang in _SUPPORTED_LANGS else "fr"
    return HTMLResponse(content=_render_index(lang))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

