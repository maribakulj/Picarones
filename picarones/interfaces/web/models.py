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

from pydantic import BaseModel, Field, field_validator

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
    expose_alto: bool = False
    """Phase B3-final corr-B (mai 2026) — active la production native
    d'ALTO XML par Tesseract via ``pytesseract.image_to_alto_xml``.

    Combiné avec ``BenchmarkRunRequest.views`` contenant
    ``alto_documentary``, débloque les sections multi-vues du rapport
    HTML.  Ignoré pour les engines non-Tesseract."""


# Phase B3-final corr-A — vues canoniques d'évaluation acceptées.
ViewName = Literal["text_final", "alto_documentary", "searchability"]


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
    # Phase B3-final corr-A/B/C (mai 2026) — exposition des features
    # B2/B5/B6 aux clients de l'API REST.
    views: list[ViewName] = Field(default_factory=lambda: ["text_final"])
    """Liste des vues d'évaluation à appliquer.  Défaut :
    ``["text_final"]`` (compat ascendante).  Pour activer le rapport
    HTML multi-vues (AltoView, SearchView), passer ``["text_final",
    "alto_documentary", "searchability"]``.  Nécessite que les
    pipelines produisent les artefacts éligibles (ex :
    ``alto_documentary`` requiert ``PipelineConfig.expose_alto=true``
    côté Tesseract)."""
    profile: Literal[
        "minimal", "standard", "philological", "diagnostics",
        "economics", "pipeline", "full",
    ] = "standard"
    """Phase B2.6 — profil de hooks document-level / corpus aggregators.
    Sélectionne quels ``@register_document_metric`` /
    ``@register_corpus_aggregator`` s'exécutent."""
    partial_dir: str = Field(default="", max_length=_MAX_PATH)
    """Phase B2.3 — répertoire pour la reprise sur interruption.
    Vide = pas de resume."""
    entity_extractor: str = Field(default="", max_length=_MAX_NAME * 4)
    """Phase B2.4 — dotted path vers une factory d'extracteur d'entités
    (ex : ``mypkg.ner:SpacyExtractor``).  Vide = pas de NER attach."""
    output_json: str = Field(default="", max_length=_MAX_PATH)
    """Phase B2.7 — chemin facultatif où sérialiser le BenchmarkResult
    legacy en JSON.  Vide = pas de sortie JSON additionnelle (le
    rapport HTML reste produit normalement)."""

    # Phase D2 audit B3-final (mai 2026) — durcissement sécurité.
    # Les champs path-like et le dotted path sont validés pour bloquer
    # les patterns dangereux (path traversal, chemins absolus, segments
    # ``..``).  Refus en 422 plutôt que dégradation silencieuse en aval.

    @field_validator("partial_dir", "output_json")
    @classmethod
    def _validate_no_path_traversal(cls, v: str, info) -> str:
        """Refuse ``..`` segments et chemins absolus.

        Le runner web confine ses opérations à ``WorkspaceManager``.
        Un payload qui contient ``../../etc/passwd`` ou
        ``/etc/passwd`` doit être rejeté immédiatement (422 Pydantic)
        plutôt qu'arriver jusqu'à ``Path()`` côté serveur.
        """
        if not v:
            return v
        import os.path
        if ".." in v.replace("\\", "/").split("/"):
            raise ValueError(
                f"{info.field_name} : segment '..' interdit "
                f"(path traversal).  Reçu : {v!r}",
            )
        if os.path.isabs(v):
            raise ValueError(
                f"{info.field_name} : chemin absolu interdit, "
                f"utiliser un chemin relatif au workspace.  Reçu : {v!r}",
            )
        return v

    @field_validator("entity_extractor")
    @classmethod
    def _validate_entity_extractor_format(cls, v: str) -> str:
        """Refuse les dotted paths mal formés.

        Format accepté : ``module.submodule:Symbol`` ou
        ``module.submodule.Symbol`` — composants alphanumériques +
        ``_``.  Refus de ``..``, ``/``, espaces, caractères spéciaux
        (vecteur d'injection).
        """
        if not v:
            return v
        import re
        # Strict : alphanum + underscore + un seul ``:`` ou ``.``
        # comme séparateur final.  Refuse explicitement ``..``,
        # slashes, espaces.
        if ".." in v or "/" in v or "\\" in v or " " in v:
            raise ValueError(
                "entity_extractor : segments '..' / slashes / "
                f"espaces interdits.  Reçu : {v!r}",
            )
        # Doit matcher le même regex que ``RunSpec._DOTTED_PATH_RE``.
        pattern = re.compile(
            r"^[a-zA-Z_][a-zA-Z0-9_]*"
            r"(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*"
            r"(?:[:.][a-zA-Z_][a-zA-Z0-9_]*)$"
        )
        if not pattern.match(v):
            raise ValueError(
                f"entity_extractor : format invalide {v!r}.  "
                "Attendu : ``module.submodule:Symbol`` ou "
                "``module.submodule.Symbol``.",
            )
        return v


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
