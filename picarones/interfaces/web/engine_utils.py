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
    # Mistral Small versions antérieures à 3.1 : text-only.
    # ``mistral-small-2402`` (fév 2024), ``mistral-small-2409``
    # (Small v2, sept 2024), ``mistral-small-2501`` (Small 3,
    # janv 2025) n'ont pas de support vision.
    "mistral-small-2402", "mistral-small-2409", "mistral-small-2501",
})
"""Modèles Mistral explicitement text-only (pas de support vision).

Note (mai 2026) : ``mistral-small-latest`` a été RETIRÉ de cette
liste — l'alias pointe désormais vers Mistral Small 3.1+
(``mistral-small-2503`` puis ``2506``) qui sont multimodaux.
Seules les versions datées antérieures à 3.1 restent text-only.
``ministral-*`` reste text-only (modèles edge sans vision,
cf. ``data/pricing.yaml``)."""

MISTRAL_SMALL_VISION = frozenset({
    "mistral-small-latest", "mistral-small-2503", "mistral-small-2506",
})
"""Mistral Small 3.1+ : multimodaux (vision).  Le runtime
``MistralAdapter`` envoie effectivement l'image pour ces modèles
(cf. ``mistral_adapter.py:_TEXT_ONLY_MODELS`` qui ne les exclut
pas)."""

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
    """Vérifie qu'un moteur OCR local est installé et retourne son statut.

    On ne fait que ``__import__(module_name)`` + ``getattr(mod, "__version__")`` —
    seules ``ImportError`` et ``AttributeError`` peuvent légitimement
    survenir. Tout autre type d'exception (panne disque, OSError…) est
    propagé pour ne pas masquer un vrai bug.
    """
    label = label or engine_id.replace("_", " ").title()
    try:
        __import__(module_name)
        installed = True
    except ImportError:
        installed = False

    version = ""
    if installed and engine_id == "tesseract":
        try:
            import importlib
            pytesseract = importlib.import_module("pytesseract")
            version = str(pytesseract.get_tesseract_version())
        except (ImportError, pytesseract.TesseractNotFoundError, OSError):
            # ``TesseractNotFoundError`` : binaire absent ; ``OSError`` :
            # ``PATH`` manquant ; ``ImportError`` : racine du sous-import.
            version = "installé"
    elif installed:
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "installé")
        except (ImportError, AttributeError):
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
    """Vérifie la disponibilité d'Ollama et liste ses modèles en un seul appel HTTP.

    Capture explicitement ``URLError`` (Ollama pas démarré, port fermé,
    timeout) et ``json.JSONDecodeError`` (réponse non-JSON inattendue).
    Toute autre exception (par ex. ``OSError`` sur lecture réseau,
    ``UnicodeDecodeError``) est aussi traitée comme "Ollama
    indisponible" — c'est l'intention du caller (UX dégradée gracieuse).
    """
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as r:
            if r.status != 200:
                return False, []
            data = json.loads(r.read().decode())
    except (urllib.error.URLError, OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False, []
    models = [m.get("name", "") for m in data.get("models", [])]
    return True, models


def get_tesseract_langs() -> list[str]:
    """Liste les langues Tesseract installées (avec fallback éditorial).

    ``TesseractNotFoundError`` quand le binaire est absent du ``PATH``,
    ``ImportError`` si pytesseract n'est pas installé, ``OSError`` sur
    appel système échoué — tous traités comme "lister les langues
    indisponible, fallback à la liste éditoriale historique".
    """
    try:
        import importlib
        pytesseract = importlib.import_module("pytesseract")
        langs = pytesseract.get_languages(config="")
        return sorted(lg for lg in langs if lg != "osd")
    except (ImportError, OSError):
        # ``pytesseract.TesseractNotFoundError`` hérite d'``OSError``.
        return ["fra", "lat", "eng", "deu", "ita", "spa"]


# ──────────────────────────────────────────────────────────────────────────
# Inférence de capacités par nom de modèle
# ──────────────────────────────────────────────────────────────────────────

def model_entry(model_id: str, capabilities: list[str]) -> dict:
    """Crée une entrée modèle avec son ID et ses capacités."""
    return {"id": model_id, "capabilities": capabilities}


def mistral_capabilities_from_model(model: dict) -> list[str]:
    """Dérive ``["text"]`` / ``["text", "vision"]`` d'un objet modèle
    Mistral renvoyé par ``GET /v1/models``.

    Source de vérité : le champ ``capabilities`` que l'API Mistral
    expose pour chaque modèle (``{"completion_chat": bool,
    "vision": bool, ...}``).  C'est l'unique source fiable : elle
    suit automatiquement les nouveaux modèles (``mistral-small-2509``
    futur, etc.) sans maintenance d'une liste hardcodée.

    Repli sur l'heuristique nom-de-modèle :func:`infer_mistral_capabilities`
    UNIQUEMENT si l'API ne renvoie pas de ``capabilities`` exploitable
    (version d'API ancienne, changement de schéma, ou liste de
    fallback statique offline).  L'heuristique est volontairement
    conservée comme garde-fou, pas comme source primaire.
    """
    caps = model.get("capabilities")
    if isinstance(caps, dict) and "vision" in caps:
        # ``completion_chat`` indique le support chat/texte ; on
        # considère tout modèle listé ici comme capable de texte
        # (ce sont des modèles chat), et ``vision`` ajoute l'image.
        out = ["text"]
        if caps.get("vision") is True:
            out.append("vision")
        return out
    # Pas de capabilities exploitable → heuristique nom (fallback).
    return infer_mistral_capabilities(model.get("id", ""))


def infer_mistral_capabilities(model_id: str) -> list[str]:
    """Heuristique de SECOURS basée sur le nom du modèle.

    Utilisée uniquement quand l'API Mistral ne fournit pas le champ
    ``capabilities`` (cf. :func:`mistral_capabilities_from_model` qui
    est la voie primaire) et pour la liste de fallback statique
    offline.  Les sets hardcodés ci-dessous sont des garde-fous
    best-effort, pas la source de vérité — ils peuvent dériver et
    c'est acceptable car l'API prime quand elle est joignable.
    """
    mid = model_id.lower()
    # Modèles explicitement vision (Pixtral)
    if "pixtral" in mid:
        return ["text", "vision"]
    # Mistral Small 3.1+ multimodaux — vérifié AVANT le test
    # text-only pour ne pas se faire écraser par un éventuel
    # préfixe.  ``mistral-small-latest`` est l'alias courant
    # (Small 3.2) ; les versions datées 2503/2506 sont aussi
    # multimodales.
    if mid in MISTRAL_SMALL_VISION:
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
    "mistral_capabilities_from_model",
    "infer_mistral_capabilities",
    "infer_openai_capabilities",
    "infer_ollama_capabilities",
]
