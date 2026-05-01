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
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from contextlib import asynccontextmanager

from fastapi import Cookie, FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from picarones import __version__
from picarones.web.corpus_utils import (
    analyze_corpus_dir as _analyze_corpus_dir,
    flatten_zip_to_dir as _flatten_zip_to_dir,
)
from picarones.web.engine_utils import (
    check_engine as _check_engine,
    fetch_ollama_info as _fetch_ollama_info,
    get_tesseract_langs as _get_tesseract_langs,
    infer_mistral_capabilities as _infer_mistral_capabilities,
    infer_ollama_capabilities as _infer_ollama_capabilities,
    infer_openai_capabilities as _infer_openai_capabilities,
    model_entry as _model_entry,
)
from picarones.web.models import (
    BenchmarkRequest,
    BenchmarkRunRequest,
    CompetitorConfig,
    HTRUnitedImportRequest,
    HuggingFaceImportRequest,
)
from picarones.web.security import (
    assert_engines_allowed,
    assert_llm_provider_allowed,
    compute_browse_roots,
    csp_middleware,
    get_max_concurrent_jobs,
    validate_image_safe,
)
from picarones.web.state import (
    IMAGE_EXTS as _IMAGE_EXTS,
    JOB_STORE as _JOB_STORE,
    JOBS as _JOBS,
    JOBS_SEMAPHORE as _JOBS_SEMAPHORE,
    SUPPORTED_LANGS as _SUPPORTED_LANGS,
    UPLOADS_DIR as _UPLOADS_DIR,
    BenchmarkJob,
    cleanup_old_jobs as _cleanup_old_jobs,
    enforce_rate_limit as _enforce_rate_limit,
    iso_now as _iso_now,
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
        _JOB_STORE.mark_orphaned_jobs_interrupted()
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

from picarones.web.state import LANG_COOKIE as _LANG_COOKIE


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


# ---------------------------------------------------------------------------
# API — models (dynamic per provider, with capability metadata)
# ---------------------------------------------------------------------------


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
    from picarones.measurements.normalization import NORMALIZATION_PROFILES

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

# ---------------------------------------------------------------------------
# API — config save/load (Sprint 28)
# ---------------------------------------------------------------------------

#: Schéma versionné des configs utilisateur. Si on change le format,
#: bumpez ce nombre et rajoutez un upgrade path dans ``_upgrade_config``.
_CONFIG_SCHEMA_VERSION = 1

#: Champs autorisés dans une config sauvegardée. On filtre explicitement
#: pour ne pas embarquer des secrets ou des clefs serveur si le client
#: pousse un dict trop riche.
_ALLOWED_CONFIG_FIELDS: frozenset[str] = frozenset({
    "schema_version",
    "saved_at",
    "label",
    "corpus_path",
    "engines",
    "normalization_profile",
    "char_exclude",
    "lang",
    "report_lang",
    "output_dir",
    "report_name",
    "competitors",
})


def _filter_config(payload: dict) -> dict:
    """Ne garde que les champs autorisés, dans un ordre stable pour les diffs."""
    out: dict[str, Any] = {}
    for k in sorted(_ALLOWED_CONFIG_FIELDS):
        if k in payload:
            out[k] = payload[k]
    return out


def _upgrade_config(payload: dict) -> dict:
    """Migre les anciennes configs vers le schéma courant.

    Schéma 1 (Sprint 28) : pas de migration nécessaire — on retourne tel quel.
    """
    return payload


@app.post("/api/config/save")
async def api_config_save(payload: dict) -> Response:
    """Sérialise un dict de config en JSON téléchargeable.

    Sprint 28 — supprime la friction *« reconfigurer chaque session »*.
    Le client envoie sa config courante (engines, profil, options),
    le serveur retourne un fichier JSON à télécharger ; un autre
    utilisateur peut le réimporter via ``/api/config/load``.
    """
    cleaned = _filter_config(payload or {})
    cleaned["schema_version"] = _CONFIG_SCHEMA_VERSION
    cleaned["saved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    body = json.dumps(cleaned, ensure_ascii=False, indent=2, sort_keys=True)
    label = str(cleaned.get("label") or "picarones-config")
    # Sanitisation du nom : pas de "/" ni "..", longueur bornée
    safe_label = "".join(c for c in label if c.isalnum() or c in "-_") or "picarones-config"
    safe_label = safe_label[:80]
    filename = f"{safe_label}-v{_CONFIG_SCHEMA_VERSION}.json"
    return Response(
        content=body,
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


@app.post("/api/config/load")
async def api_config_load(payload: dict) -> dict:
    """Valide et normalise une config uploadée.

    Le client envoie le contenu JSON déjà parsé (le frontend lit le
    fichier via ``FileReader``). On filtre les champs autorisés,
    applique l'upgrade path éventuel, et retourne le dict normalisé.
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Le corps doit être un objet JSON.")

    schema = payload.get("schema_version")
    if not isinstance(schema, int) or schema < 1 or schema > _CONFIG_SCHEMA_VERSION:
        raise HTTPException(
            status_code=400,
            detail=(
                f"schema_version invalide ({schema!r}) — "
                f"attendu entre 1 et {_CONFIG_SCHEMA_VERSION}."
            ),
        )

    upgraded = _upgrade_config(payload)
    return {
        "config": _filter_config(upgraded),
        "schema_version": _CONFIG_SCHEMA_VERSION,
    }


# ---------------------------------------------------------------------------
# API — synthèse narrative en preview (Sprint 28)
# ---------------------------------------------------------------------------

@app.get("/api/benchmark/{job_id}/synthesis_preview")
async def api_benchmark_synthesis_preview(job_id: str, lang: str = "fr") -> dict:
    """Rend la synthèse narrative d'un job terminé sans rouvrir le HTML.

    Sprint 28 — un chercheur attend 20 minutes la fin d'un benchmark
    et veut savoir d'un coup d'œil *« le moteur narratif a-t-il quelque
    chose d'intéressant à dire ? »* avant d'ouvrir le rapport HTML
    complet. Cet endpoint :

    1. Charge le ``BenchmarkJob`` (RAM ou DB) ;
    2. Lit le JSON de résultats associé via ``output_path`` ;
    3. Appelle ``build_synthesis()`` côté serveur ;
    4. Retourne ``{sentences, facts, lang}``.

    Renvoie ``409 Conflict`` si le job n'est pas terminé, ``404`` si
    introuvable, ``422`` si le JSON associé est manquant ou cassé.
    """
    if lang not in _SUPPORTED_LANGS:
        lang = "fr"

    # 1. Statut courant : RAM si dispo, sinon DB.
    ram_job = _JOBS.get(job_id)
    db_job = _JOB_STORE.get_job(job_id)
    if ram_job is None and db_job is None:
        raise HTTPException(status_code=404, detail=f"Job non trouvé : {job_id}")

    status = ram_job.status if ram_job is not None else db_job["status"]
    if status not in ("complete",):
        raise HTTPException(
            status_code=409,
            detail=f"Synthèse indisponible : statut courant = {status!r} (attendu 'complete').",
        )

    output_path = (
        ram_job.output_path if ram_job is not None
        else (db_job or {}).get("output_path", "")
    )
    if not output_path:
        raise HTTPException(status_code=422, detail="Aucun rapport produit pour ce job.")

    # 2. Le HTML est à ``output_path`` ; le JSON associé est à côté
    # (convention ``picarones run -o results.json --output-html``).
    html_path = Path(output_path)
    json_candidates = [
        html_path.with_suffix(".json"),
        html_path.with_name(html_path.stem + "_results.json"),
        html_path.parent / "results.json",
    ]
    json_path: Optional[Path] = next((p for p in json_candidates if p.exists()), None)
    if json_path is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "JSON de résultats introuvable à côté du rapport HTML. "
                f"Cherché : {[str(p) for p in json_candidates]}"
            ),
        )

    try:
        report_json = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=f"Lecture JSON échouée : {exc}")

    from picarones.measurements.narrative import build_synthesis

    synthesis = build_synthesis(report_json, lang=lang)
    return {
        "job_id": job_id,
        "lang": lang,
        "source_json": str(json_path),
        "sentences": synthesis.get("sentences", []),
        "facts": synthesis.get("facts", []),
    }


# ---------------------------------------------------------------------------
# API — historique des régressions (Sprint 28)
# ---------------------------------------------------------------------------

@app.get("/api/history/regressions")
async def api_history_regressions(
    engine: Optional[str] = Query(default=None, description="Filtre par moteur"),
    threshold: float = Query(default=0.01, description="Seuil régression CER absolu"),
    db_path: Optional[str] = Query(default=None, description="Chemin SQLite history"),
) -> dict:
    """Liste les régressions détectées dans l'historique longitudinal.

    Sprint 28 — surface de l'infrastructure ``BenchmarkHistory`` du
    Sprint 8, qui était limitée au CLI ``picarones history --regression``.
    Le rapport HTML peut désormais consommer cet endpoint pour afficher
    un encart *« ⚠ Tesseract a régressé de 0,8 pp depuis le 12 janvier »*
    en tête de page.
    """
    from picarones.measurements.history import BenchmarkHistory

    try:
        history = BenchmarkHistory(db_path) if db_path else BenchmarkHistory()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ouverture historique échouée : {exc}")

    # Si aucun moteur n'est passé, on liste tous les moteurs présents
    # dans l'historique et on tente une détection sur chacun.
    if engine:
        targets = [engine]
    else:
        # Récupération des engines distincts via la query publique.
        try:
            entries = history.query(limit=10000)
            targets = sorted({e.engine for e in entries if e.engine})
        except Exception:
            targets = []

    out: list[dict[str, Any]] = []
    for eng in targets:
        try:
            res = history.detect_regression(engine=eng, threshold=threshold)
        except Exception as exc:
            _logger.warning("[regressions] detect_regression(%s) échoué : %s", eng, exc)
            continue
        if res is None:
            continue
        d = {
            "engine": eng,
            "is_regression": getattr(res, "is_regression", False),
            "delta_cer": getattr(res, "delta_cer", None),
            "current_cer": getattr(res, "current_cer", None),
            "baseline_cer": getattr(res, "baseline_cer", None),
            "current_run_id": getattr(res, "current_run_id", None),
            "baseline_run_id": getattr(res, "baseline_run_id", None),
        }
        if d["is_regression"]:
            out.append(d)

    return {
        "threshold": float(threshold),
        "regressions": out,
        "count": len(out),
    }


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
    from picarones.extras.importers.htr_united import HTRUnitedCatalogue

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
    from picarones.extras.importers.htr_united import HTRUnitedCatalogue, import_htr_united_corpus

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
    from picarones.extras.importers.huggingface import HuggingFaceImporter

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
    from picarones.extras.importers.huggingface import HuggingFaceImporter

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
    job = BenchmarkJob(job_id=job_id, _store=_JOB_STORE)
    _JOB_STORE.create_job(job_id)
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
    if job is not None:
        return job.as_dict()
    # Sprint 26 — fallback DB : le job n'est pas (plus) en RAM dans ce
    # worker mais peut exister en base (autre worker, ou redémarrage).
    db_job = _JOB_STORE.get_job(job_id)
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
    job = _JOBS.get(job_id)
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

    job = _JOBS.get(job_id)
    db_job = _JOB_STORE.get_job(job_id)
    if job is None and db_job is None:
        raise HTTPException(status_code=404, detail=f"Job non trouvé : {job_id}")

    async def event_generator() -> AsyncIterator[str]:
        queue: Optional[asyncio.Queue] = None
        if job is not None:
            queue = job.subscribe()
        try:
            # 1) Backlog depuis la base — l'autorité de vérité (Sprint 26).
            backlog = _JOB_STORE.get_events_after(job_id, last_seq=last_seq)
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


def _sse_format(event_type: str, data: Any, seq: Optional[int] = None) -> str:
    """Format SSE.

    Sprint 26 — émet une ligne ``id: <seq>`` quand le ``seq`` est connu.
    C'est la valeur que le navigateur renvoie automatiquement dans
    ``Last-Event-ID`` à la prochaine connexion (cf.
    https://html.spec.whatwg.org/multipage/server-sent-events.html).
    """
    payload = json.dumps(data, ensure_ascii=False)
    head = f"id: {seq}\n" if seq is not None else ""
    return f"{head}event: {event_type}\ndata: {payload}\n\n"


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
    job = BenchmarkJob(job_id=job_id, _store=_JOB_STORE)
    _JOB_STORE.create_job(job_id)
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

    job.set_status("running")
    job.started_at = _iso_now()
    job.add_event("start", {"message": "Démarrage du benchmark…", "corpus": req.corpus_path})

    try:
        from picarones.core.corpus import load_corpus_from_directory
        from picarones.measurements.runner import run_benchmark

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

        from picarones.measurements.normalization import _parse_exclude_chars
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
        job.set_status("complete")

        ranking = result.ranking()
        job.add_event("complete", {
            "message": "Benchmark terminé.",
            "output_html": output_html,
            "output_json": output_json,
            "ranking": ranking,
        })

    except Exception as exc:
        job.set_status("error", error=str(exc))
        job.add_event("error", {"message": f"Erreur : {exc}"})


def _run_benchmark_thread(job: BenchmarkJob, req: BenchmarkRequest) -> None:
    """Exécute le benchmark dans un thread et envoie des événements SSE."""

    job.set_status("running")
    job.started_at = _iso_now()
    job.add_event("start", {"message": "Démarrage du benchmark…", "corpus": req.corpus_path})

    try:
        from picarones.core.corpus import load_corpus_from_directory
        from picarones.measurements.runner import run_benchmark

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

        from picarones.measurements.normalization import _parse_exclude_chars
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
        job.set_status("complete")

        # Classement final
        ranking = result.ranking()
        job.add_event("complete", {
            "message": "Benchmark terminé.",
            "output_html": output_html,
            "output_json": output_json,
            "ranking": ranking,
        })

    except Exception as exc:
        job.set_status("error", error=str(exc))
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

