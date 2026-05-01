"""Helpers métier des routeurs ``engines`` et ``models``.

Ces utilitaires détectent la disponibilité des moteurs OCR et LLM
locaux, listent leurs modèles, et inférent leurs capacités (text /
vision) à partir du nom — quand l'API du provider ne fournit pas
cette information directement.
"""

from __future__ import annotations

import json


# ──────────────────────────────────────────────────────────────────────────
# Tables de capacités par famille
# ──────────────────────────────────────────────────────────────────────────

MISTRAL_TEXT_ONLY = frozenset({
    "ministral-3b-latest", "ministral-8b-latest", "mistral-tiny",
    "mistral-tiny-latest", "open-mistral-7b", "open-mixtral-8x7b",
    "mistral-small-latest", "mistral-small-2409",
})
"""Modèles Mistral explicitement text-only (pas de support vision)."""

MISTRAL_TEXT_ONLY_PREFIXES = (
    "ministral", "open-mistral", "open-mixtral", "codestral",
    "mistral-embed", "mistral-tiny",
)
"""Préfixes de modèles Mistral à traiter comme text-only."""

OLLAMA_VISION_FAMILIES = frozenset({
    "llava", "bakllava", "moondream", "minicpm-v", "llama3.2-vision",
    "llava-llama3", "llava-phi3", "nanollava",
})
"""Familles Ollama multimodales connues."""


# ──────────────────────────────────────────────────────────────────────────
# Disponibilité des moteurs locaux
# ──────────────────────────────────────────────────────────────────────────

def check_engine(engine_id: str, module_name: str, label: str = "") -> dict:
    """Vérifie qu'un moteur OCR local est installé et retourne son statut."""
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
            version = str(pytesseract.get_tesseract_version())
        except Exception:  # noqa: BLE001
            version = "installé"
    elif installed:
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "installé")
        except Exception:  # noqa: BLE001
            version = "installé"

    return {
        "id": engine_id,
        "label": label,
        "type": "ocr",
        "available": installed,
        "version": version,
        "status": "available" if installed else "not_installed",
    }


def fetch_ollama_info() -> tuple[bool, list[str]]:
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
    except Exception:  # noqa: BLE001
        return False, []


def get_tesseract_langs() -> list[str]:
    """Liste les langues Tesseract installées (avec fallback éditorial)."""
    try:
        import pytesseract
        langs = pytesseract.get_languages(config="")
        return sorted(lg for lg in langs if lg != "osd")
    except Exception:  # noqa: BLE001
        return ["fra", "lat", "eng", "deu", "ita", "spa"]


# ──────────────────────────────────────────────────────────────────────────
# Inférence de capacités par nom de modèle
# ──────────────────────────────────────────────────────────────────────────

def model_entry(model_id: str, capabilities: list[str]) -> dict:
    """Crée une entrée modèle avec son ID et ses capacités."""
    return {"id": model_id, "capabilities": capabilities}


def infer_mistral_capabilities(model_id: str) -> list[str]:
    mid = model_id.lower()
    # Modèles explicitement vision (Pixtral)
    if "pixtral" in mid:
        return ["text", "vision"]
    # Modèles explicitement text-only
    if mid in MISTRAL_TEXT_ONLY or any(mid.startswith(p) for p in MISTRAL_TEXT_ONLY_PREFIXES):
        return ["text"]
    # Mistral Large et modèles récents non-identifiés → vision par défaut
    if "mistral-large" in mid or "mistral-medium" in mid:
        return ["text", "vision"]
    # Par défaut, marquer comme text-only (plus sûr que de supposer vision)
    return ["text"]


def infer_openai_capabilities(model_id: str) -> list[str]:
    mid = model_id.lower()
    if "gpt-4o" in mid or "gpt-4-turbo" in mid or "gpt-4.1" in mid or "o1" in mid or "o3" in mid:
        return ["text", "vision"]
    return ["text"]


def infer_ollama_capabilities(model_name: str) -> list[str]:
    base = model_name.split(":")[0].lower()
    if any(base.startswith(family) for family in OLLAMA_VISION_FAMILIES):
        return ["text", "vision"]
    return ["text"]


__all__ = [
    "MISTRAL_TEXT_ONLY",
    "MISTRAL_TEXT_ONLY_PREFIXES",
    "OLLAMA_VISION_FAMILIES",
    "check_engine",
    "fetch_ollama_info",
    "get_tesseract_langs",
    "model_entry",
    "infer_mistral_capabilities",
    "infer_openai_capabilities",
    "infer_ollama_capabilities",
]
