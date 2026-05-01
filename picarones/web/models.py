"""Modèles Pydantic partagés par les routers FastAPI.

Ces schémas décrivent les payloads des requêtes ``POST`` consommées par
plusieurs endpoints du serveur web. Les sortir d'``app.py`` permet à
chaque routeur de les importer sans dépendance vers l'application
elle-même.
"""

from __future__ import annotations

from pydantic import BaseModel


class BenchmarkRequest(BaseModel):
    corpus_path: str
    engines: list[str] = ["tesseract"]
    normalization_profile: str = "nfc"
    char_exclude: str = ""
    """Caractères à ignorer (séparés par virgule, ex: ``"',–"``)."""
    output_dir: str = "./rapports/"
    report_name: str = ""
    lang: str = "fra"
    report_lang: str = "fr"
    """Langue du rapport HTML : ``fr`` ou ``en``."""


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
    """Moteur OCR : ``tesseract``, ``mistral_ocr``, … ou ``corpus`` pour
    utiliser l'OCR pré-calculé."""
    ocr_model: str = ""
    llm_provider: str = ""
    llm_model: str = ""
    pipeline_mode: str = ""
    prompt_file: str = ""


class BenchmarkRunRequest(BaseModel):
    corpus_path: str
    competitors: list[CompetitorConfig]
    normalization_profile: str = "nfc"
    char_exclude: str = ""
    output_dir: str = "./rapports/"
    report_name: str = ""
    report_lang: str = "fr"


__all__ = [
    "BenchmarkRequest",
    "HTRUnitedImportRequest",
    "HuggingFaceImportRequest",
    "CompetitorConfig",
    "BenchmarkRunRequest",
]
