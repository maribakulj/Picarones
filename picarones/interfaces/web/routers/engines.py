"""Router de découverte des moteurs OCR/LLM et de leurs modèles."""

from __future__ import annotations

import asyncio
import json
import os
import urllib.request
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from picarones.interfaces.web.engine_utils import (
    check_engine,
    fetch_ollama_info,
    get_tesseract_langs,
    infer_ollama_capabilities,
    infer_openai_capabilities,
    mistral_capabilities_from_model,
    model_entry,
)

router = APIRouter()


# ──────────────────────────────────────────────────────────────────────────
# Disponibilité des moteurs OCR + LLM
# ──────────────────────────────────────────────────────────────────────────

@router.get("/api/engines")
async def api_engines() -> dict:
    """Statut des 5 moteurs OCR (locaux + cloud) et 4 providers LLM.

    Les détections (``check_engine``, ``fetch_ollama_info``,
    ``get_tesseract_langs``) font des imports + un appel HTTP local
    qui peuvent bloquer plusieurs centaines de ms ; déléguées à un
    thread via ``asyncio.to_thread`` pour ne pas figer l'event loop.
    """
    return await asyncio.to_thread(_collect_engines_sync)


def _collect_engines_sync() -> dict:
    """Implémentation synchrone de :func:`api_engines`."""
    engines = []

    # OCR locaux
    tess = check_engine("tesseract", "pytesseract")
    tess["langs"] = get_tesseract_langs()
    engines.append(tess)
    engines.append(check_engine("pero_ocr", "pero_ocr", label="Pero OCR"))
    engines.append(check_engine("kraken", "kraken", label="Kraken"))
    engines.append(check_engine("calamari", "calamari_ocr", label="Calamari"))

    # OCR cloud — déduits de la présence d'une clé API en environnement
    cloud_ocr_specs = (
        ("mistral_ocr", "Mistral OCR (Pixtral / mistral-ocr-latest)", "MISTRAL_API_KEY"),
        ("google_vision", "Google Vision API",
         "GOOGLE_APPLICATION_CREDENTIALS"),
        ("azure_doc_intel", "Azure Document Intelligence", "AZURE_DOC_INTEL_KEY"),
    )
    for engine_id, label, primary_env in cloud_ocr_specs:
        # Google Vision accepte deux variables alternatives
        if engine_id == "google_vision":
            key = (
                os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                or os.environ.get("GOOGLE_API_KEY")
            )
        else:
            key = os.environ.get(primary_env)
        engines.append({
            "id": engine_id,
            "label": label,
            "type": "ocr_cloud",
            "available": bool(key),
            "key_env": primary_env,
            "status": "configured" if key else "missing_key",
            "version": "",
        })

    # LLMs cloud
    llms = []
    cloud_llm_specs = (
        ("openai", "OpenAI (GPT-4o, GPT-4o mini)", "OPENAI_API_KEY"),
        ("anthropic", "Anthropic (Claude Sonnet, Haiku)", "ANTHROPIC_API_KEY"),
        ("mistral", "Mistral LLM (Mistral Large, Small…)", "MISTRAL_API_KEY"),
    )
    for llm_id, label, env in cloud_llm_specs:
        key = os.environ.get(env)
        llms.append({
            "id": llm_id,
            "label": label,
            "type": "llm",
            "available": bool(key),
            "key_env": env,
            "status": "configured" if key else "missing_key",
        })

    # Ollama local (un seul appel HTTP qui sert aussi à lister les modèles)
    ollama_available, ollama_models = fetch_ollama_info()
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


# ──────────────────────────────────────────────────────────────────────────
# Modèles disponibles par provider (avec capacités text / vision)
# ──────────────────────────────────────────────────────────────────────────

def _fetch_json(url: str, headers: dict) -> dict:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


def _format_models(provider: str, models: list[dict], capability: str) -> dict:
    """Filtre et formate la liste des modèles avant envoi au client."""
    if capability:
        models = [m for m in models if capability in m["capabilities"]]
    return {
        "provider": provider,
        "models": models,
        "model_ids": [m["id"] for m in models],
    }


@router.get("/api/models/{provider}")
async def api_models(
    provider: str,
    capability: str = Query(
        default="",
        description="Filtre par capacité : 'text', 'vision', ou vide pour tout",
    ),
) -> dict:
    """Modèles disponibles avec leurs capacités (text, vision).

    Interroge l'API du provider en temps réel. Les capacités sont
    déterminées par heuristique sur le nom du modèle quand l'API ne
    fournit pas cette information directement. Le paramètre
    ``capability`` filtre les résultats.

    Délègue à un thread (``asyncio.to_thread``) car ``urllib`` est
    synchrone — un timeout de 10 s figerait sinon tout l'event loop
    FastAPI pendant l'attente.
    """
    return await asyncio.to_thread(_models_for_provider_sync, provider, capability)


def _models_for_provider_sync(provider: str, capability: str) -> dict:
    """Implémentation synchrone de :func:`api_models`."""
    if provider == "tesseract":
        langs = get_tesseract_langs()
        return {"provider": provider, "models": langs, "model_ids": langs}

    if provider == "mistral_ocr":
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            return {
                "provider": provider, "models": [], "model_ids": [],
                "error": "MISTRAL_API_KEY non définie",
            }
        try:
            data = _fetch_json(
                "https://api.mistral.ai/v1/models",
                {"Authorization": f"Bearer {api_key}"},
            )
            models = [
                model_entry(m["id"], mistral_capabilities_from_model(m))
                for m in data.get("data", [])
                if "pixtral" in m["id"].lower() or "mistral-ocr" in m["id"].lower()
            ]
            return _format_models(
                provider, sorted(models, key=lambda m: m["id"]), capability,
            )
        except Exception as exc:  # noqa: BLE001
            fallback = [
                model_entry("pixtral-12b-2409", ["text", "vision"]),
                model_entry("pixtral-large-latest", ["text", "vision"]),
                model_entry("mistral-ocr-latest", ["text", "vision"]),
            ]
            return {**_format_models(provider, fallback, capability), "error": str(exc)}

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return {
                "provider": provider, "models": [], "model_ids": [],
                "error": "OPENAI_API_KEY non définie",
            }
        try:
            data = _fetch_json(
                "https://api.openai.com/v1/models",
                {"Authorization": f"Bearer {api_key}"},
            )
            models = [
                model_entry(m["id"], infer_openai_capabilities(m["id"]))
                for m in data.get("data", [])
                if "gpt-4" in m["id"].lower()
                or "o1" in m["id"].lower()
                or "o3" in m["id"].lower()
            ]
            return _format_models(
                provider, sorted(models, key=lambda m: m["id"], reverse=True),
                capability,
            )
        except Exception as exc:  # noqa: BLE001
            fallback = [
                model_entry("gpt-4o", ["text", "vision"]),
                model_entry("gpt-4o-mini", ["text", "vision"]),
                model_entry("gpt-4-turbo", ["text", "vision"]),
            ]
            return {**_format_models(provider, fallback, capability), "error": str(exc)}

    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return {
                "provider": provider, "models": [], "model_ids": [],
                "error": "ANTHROPIC_API_KEY non définie",
            }
        try:
            data = _fetch_json(
                "https://api.anthropic.com/v1/models",
                {"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            )
            # Tous les modèles Claude 3+ supportent la vision
            models = [
                model_entry(m["id"], ["text", "vision"])
                for m in data.get("data", [])
            ]
            return _format_models(provider, models, capability)
        except Exception as exc:  # noqa: BLE001
            fallback = [
                model_entry("claude-sonnet-4-6", ["text", "vision"]),
                model_entry("claude-haiku-4-5-20251001", ["text", "vision"]),
                model_entry("claude-opus-4-6", ["text", "vision"]),
            ]
            return {**_format_models(provider, fallback, capability), "error": str(exc)}

    if provider == "mistral":
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            return {
                "provider": provider, "models": [], "model_ids": [],
                "error": "MISTRAL_API_KEY non définie",
            }
        try:
            data = _fetch_json(
                "https://api.mistral.ai/v1/models",
                {"Authorization": f"Bearer {api_key}"},
            )
            # Inclure TOUS les modèles Mistral (y compris Pixtral pour la vision)
            # sauf mistral-ocr qui est un endpoint OCR dédié, pas un LLM chat.
            # Capacités lues depuis le champ ``capabilities`` de l'API
            # (source de vérité auto-maintenue), heuristique nom en
            # repli seulement.
            models = [
                model_entry(m["id"], mistral_capabilities_from_model(m))
                for m in data.get("data", [])
                if "mistral-ocr" not in m["id"].lower()
            ]
            return _format_models(
                provider, sorted(models, key=lambda m: m["id"]), capability,
            )
        except Exception as exc:  # noqa: BLE001
            fallback = [
                model_entry("mistral-large-latest", ["text", "vision"]),
                model_entry("mistral-medium-latest", ["text", "vision"]),
                model_entry("pixtral-large-latest", ["text", "vision"]),
                model_entry("pixtral-12b-2409", ["text", "vision"]),
                # Mistral Small 3.1+ est multimodal (cf.
                # engine_utils.MISTRAL_SMALL_VISION).
                model_entry("mistral-small-latest", ["text", "vision"]),
            ]
            return {**_format_models(provider, fallback, capability), "error": str(exc)}

    if provider == "ollama":
        _, model_names = fetch_ollama_info()
        models = [
            model_entry(name, infer_ollama_capabilities(name))
            for name in model_names
        ]
        return _format_models(provider, models, capability)

    if provider == "google_vision":
        models = [
            model_entry("document_text_detection", ["vision"]),
            model_entry("text_detection", ["vision"]),
        ]
        return _format_models(provider, models, capability)

    if provider == "azure_doc_intel":
        models = [
            model_entry("prebuilt-document", ["vision"]),
            model_entry("prebuilt-read", ["vision"]),
        ]
        return _format_models(provider, models, capability)

    if provider == "prompts":
        prompts_dir = Path(__file__).resolve().parent.parent.parent.parent / "prompts"
        prompts = (
            sorted(f.name for f in prompts_dir.glob("*.txt"))
            if prompts_dir.exists()
            else []
        )
        return {"provider": provider, "models": prompts, "model_ids": prompts}

    raise HTTPException(status_code=404, detail=f"Provider inconnu : {provider}")
