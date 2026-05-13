"""Modèles Pydantic partagés par les routers FastAPI.

Ces schémas décrivent les payloads des requêtes ``POST`` consommées
par plusieurs endpoints du serveur web. Les sortir d'``app.py``
permet à chaque routeur de les importer sans dépendance vers
l'application elle-même.

Validation stricte
------------------
Tous les champs ``str`` ont une borne ``max_length`` proportionnée
à leur usage attendu (chemin filesystem, identifiant HuggingFace,
nom de rapport…) pour empêcher qu'un payload géant n'épuise la
mémoire avant validation. Les énumérations finies (langue OCR,
langue de rapport) sont typées en ``Literal[...]`` pour rejeter au
plus tôt les valeurs invalides.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Bornes éditoriales — ajustées au plus large raisonnable, pas plus.
_MAX_PATH = 1024
"""Longueur max d'un chemin filesystem (limite POSIX généralement 4096)."""

_MAX_NAME = 256
"""Longueur max d'un identifiant ou nom court (rapport, label, dataset)."""

_MAX_PROMPT_FILENAME = 256
"""Nom de fichier prompt — ``"correction_medieval_french.txt"`` etc."""

_MAX_CHAR_EXCLUDE = 256
"""Liste de caractères à exclure (séparés par virgules)."""

_MAX_ENGINE_LIST = 32
"""Nombre max de moteurs OCR par requête legacy."""

_MAX_COMPETITORS = 32
"""Nombre max de concurrents composés par benchmark/run."""

# Codes ISO Tesseract acceptés pour le paramètre ``lang`` (historiquement
# transporté par ``BenchmarkRequest``, désormais via ``PipelineConfig.ocr_model``
# selon le moteur cible).  Liste explicite plutôt que ``str`` ouvert
# pour rejeter au plus tôt une valeur fantaisiste qui transiterait
# vers ``pytesseract`` en pure perte.
TesseractLang = Literal[
    "fra", "lat", "eng", "deu", "ita", "spa", "por", "nld", "cat",
    "rum", "ell", "ara", "heb", "rus", "ukr", "pol", "ces", "swe",
]

ReportLang = Literal["fr", "en"]
"""Langue du rapport HTML."""

NormalizationProfileId = Literal[
    "nfc", "caseless", "minimal",
    "medieval_french", "early_modern_french",
    "medieval_latin",
    "early_modern_english", "medieval_english",
    "secretary_hand",
    "sans_ponctuation", "sans_apostrophes",
]
"""Identifiants des profils de normalisation Unicode disponibles.

Liste alignée sur ``measurements.normalization.NORMALIZATION_PROFILES``
(11 profils). Toute addition côté ``normalization.py`` doit être
répercutée ici sous peine de rejet Pydantic au niveau API web.
Sprint A14-S1 — alignement README ↔ web models ↔ runtime."""

# Phase 7.1 audit code-quality (2026-05) : ``PipelineMode`` est désormais
# importé depuis :data:`picarones.domain.pipeline_spec.PipelineMode`
# (source de vérité unique).  L'alias local préserve l'API publique
# de ce module pour les callers historiques.
from picarones.domain.pipeline_spec import PipelineMode  # noqa: E402


# ``BenchmarkRequest`` (mode legacy à liste de moteurs plats) a été
# supprimé en Phase 4.2 audit code-quality (2026-05) — les clients
# utilisent désormais ``BenchmarkRunRequest`` avec
# ``competitors: list[PipelineConfig]``.  Rupture API documentée
# dans CHANGELOG v2.0.


class HTRUnitedImportRequest(BaseModel):
    entry_id: str = Field(min_length=1, max_length=_MAX_NAME)
    output_dir: str = Field(default="./corpus/", max_length=_MAX_PATH)
    max_samples: int = Field(default=100, ge=1, le=10_000)


class HuggingFaceImportRequest(BaseModel):
    dataset_id: str = Field(min_length=1, max_length=_MAX_NAME)
    output_dir: str = Field(default="./corpus/", max_length=_MAX_PATH)
    split: str = Field(default="train", max_length=_MAX_NAME)
    max_samples: int = Field(default=100, ge=1, le=10_000)


class PipelineConfig(BaseModel):
    name: str = Field(default="", max_length=_MAX_NAME)
    engine_name: str = Field(default="", max_length=_MAX_NAME)
    """Identifiant du moteur de transcription : ``tesseract``,
    ``mistral_ocr``, ``kraken``, ``calamari``, … ou ``corpus`` pour
    utiliser l'OCR pré-calculé.  Vide (``""``) pour un pipeline LLM
    seul (zero-shot VLM).

    Phase 5b du chantier post-rewrite : renommé depuis ``ocr_engine``
    car le field accepte aussi des VLMs (zero_shot) et des sources
    pré-calculées (``corpus``) — le préfixe ``ocr_`` était trompeur.
    Rupture API : les clients qui envoyaient ``ocr_engine`` reçoivent
    désormais 422.
    """
    ocr_model: str = Field(default="", max_length=_MAX_NAME)
    llm_provider: str = Field(default="", max_length=_MAX_NAME)
    llm_model: str = Field(default="", max_length=_MAX_NAME)
    pipeline_mode: PipelineMode | Literal[""] = ""
    """Mode du pipeline OCR+LLM — vide si pas de pipeline LLM (OCR seul).

    Typage strict (Phase 2 chantier post-rewrite) : Pydantic rejette
    en 422 toute valeur hors de la matrice canonique au lieu d'aliaser
    silencieusement sur ``text_only``.  La chaîne vide (``""``) reste
    autorisée pour indiquer qu'aucun LLM n'est attaché au moteur OCR.
    """
    prompt_file: str = Field(default="", max_length=_MAX_PROMPT_FILENAME)


class BenchmarkRunRequest(BaseModel):
    corpus_path: str = Field(min_length=1, max_length=_MAX_PATH)
    competitors: list[PipelineConfig] = Field(
        min_length=1, max_length=_MAX_COMPETITORS,
    )
    normalization_profile: NormalizationProfileId = "nfc"
    char_exclude: str = Field(default="", max_length=_MAX_CHAR_EXCLUDE)
    output_dir: str = Field(default="./rapports/", max_length=_MAX_PATH)
    report_name: str = Field(default="", max_length=_MAX_NAME)
    report_lang: ReportLang = "fr"


__all__ = [
    "TesseractLang",
    "ReportLang",
    "NormalizationProfileId",
    "PipelineMode",
    "HTRUnitedImportRequest",
    "HuggingFaceImportRequest",
    "PipelineConfig",
    "BenchmarkRunRequest",
]
