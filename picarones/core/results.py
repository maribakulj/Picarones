"""Modèle de données des résultats et export JSON.

Hiérarchie
----------
BenchmarkResult
  └── EngineReport          (un par moteur)
        └── DocumentResult  (un par document)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from picarones import __version__
from picarones.core.metrics import MetricsResult, aggregate_metrics


@dataclass
class DocumentResult:
    """Résultat d'un moteur sur un seul document."""

    doc_id: str
    image_path: str
    ground_truth: str
    hypothesis: str
    metrics: MetricsResult
    duration_seconds: float
    engine_error: Optional[str] = None
    # Champs spécifiques aux pipelines OCR+LLM
    ocr_intermediate: Optional[str] = None
    """Sortie OCR brute avant correction LLM (None pour les moteurs OCR seuls)."""
    pipeline_metadata: dict = field(default_factory=dict)
    """Métadonnées du pipeline : mode, prompt, over-normalization…"""
    # Champs Sprint 5 — métriques avancées patrimoniales
    confusion_matrix: Optional[dict] = None
    """Matrice de confusion unicode sérialisée."""
    char_scores: Optional[dict] = None
    """Scores ligatures et diacritiques."""
    taxonomy: Optional[dict] = None
    """Classification taxonomique des erreurs (classes 1-9)."""
    structure: Optional[dict] = None
    """Analyse structurelle (segmentation lignes, ordre lecture)."""
    image_quality: Optional[dict] = None
    """Métriques de qualité image."""

    def as_dict(self) -> dict:
        d = {
            "doc_id": self.doc_id,
            "image_path": self.image_path,
            "ground_truth": self.ground_truth,
            "hypothesis": self.hypothesis,
            "metrics": self.metrics.as_dict(),
            "duration_seconds": self.duration_seconds,
            "engine_error": self.engine_error,
        }
        if self.ocr_intermediate is not None:
            d["ocr_intermediate"] = self.ocr_intermediate
        if self.pipeline_metadata:
            d["pipeline_metadata"] = self.pipeline_metadata
        if self.confusion_matrix is not None:
            d["confusion_matrix"] = self.confusion_matrix
        if self.char_scores is not None:
            d["char_scores"] = self.char_scores
        if self.taxonomy is not None:
            d["taxonomy"] = self.taxonomy
        if self.structure is not None:
            d["structure"] = self.structure
        if self.image_quality is not None:
            d["image_quality"] = self.image_quality
        return d


@dataclass
class EngineReport:
    """Rapport complet d'un moteur (ou pipeline) sur l'ensemble du corpus."""

    engine_name: str
    engine_version: str
    engine_config: dict
    document_results: list[DocumentResult]
    aggregated_metrics: dict = field(default_factory=dict)
    pipeline_info: dict = field(default_factory=dict)
    """Métadonnées du pipeline OCR+LLM (vide pour les moteurs OCR seuls).
    Clés typiques : mode, prompt_file, llm_model, llm_provider, pipeline_steps,
    over_normalization (score agrégé, classe 10 de la taxonomie).
    """
    # Métriques agrégées Sprint 5
    aggregated_confusion: Optional[dict] = None
    """Matrice de confusion unicode agrégée sur le corpus."""
    aggregated_char_scores: Optional[dict] = None
    """Scores ligatures/diacritiques agrégés."""
    aggregated_taxonomy: Optional[dict] = None
    """Distribution taxonomique des erreurs agrégée."""
    aggregated_structure: Optional[dict] = None
    """Métriques structurelles agrégées."""
    aggregated_image_quality: Optional[dict] = None
    """Métriques de qualité image agrégées."""

    def __post_init__(self) -> None:
        if not self.aggregated_metrics and self.document_results:
            self.aggregated_metrics = aggregate_metrics(
                [dr.metrics for dr in self.document_results]
            )

    @property
    def mean_cer(self) -> Optional[float]:
        cer_stats = self.aggregated_metrics.get("cer", {})
        return cer_stats.get("mean")

    @property
    def mean_wer(self) -> Optional[float]:
        wer_stats = self.aggregated_metrics.get("wer", {})
        return wer_stats.get("mean")

    @property
    def ligature_score(self) -> Optional[float]:
        """Score de ligatures agrégé (None si non calculé)."""
        if self.aggregated_char_scores:
            return self.aggregated_char_scores.get("ligature", {}).get("score")
        return None

    @property
    def diacritic_score(self) -> Optional[float]:
        """Score diacritique agrégé (None si non calculé)."""
        if self.aggregated_char_scores:
            return self.aggregated_char_scores.get("diacritic", {}).get("score")
        return None

    @property
    def is_pipeline(self) -> bool:
        """Vrai si ce rapport correspond à un pipeline OCR+LLM."""
        return bool(self.pipeline_info)

    def as_dict(self) -> dict:
        d = {
            "engine_name": self.engine_name,
            "engine_version": self.engine_version,
            "engine_config": self.engine_config,
            "aggregated_metrics": self.aggregated_metrics,
            "document_results": [dr.as_dict() for dr in self.document_results],
        }
        if self.pipeline_info:
            d["pipeline_info"] = self.pipeline_info
        if self.aggregated_confusion is not None:
            d["aggregated_confusion"] = self.aggregated_confusion
        if self.aggregated_char_scores is not None:
            d["aggregated_char_scores"] = self.aggregated_char_scores
        if self.aggregated_taxonomy is not None:
            d["aggregated_taxonomy"] = self.aggregated_taxonomy
        if self.aggregated_structure is not None:
            d["aggregated_structure"] = self.aggregated_structure
        if self.aggregated_image_quality is not None:
            d["aggregated_image_quality"] = self.aggregated_image_quality
        return d


@dataclass
class BenchmarkResult:
    """Résultat complet d'un benchmark multi-moteurs sur un corpus."""

    corpus_name: str
    corpus_source: Optional[str]
    document_count: int
    engine_reports: list[EngineReport]
    run_date: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())
    picarones_version: str = __version__
    metadata: dict = field(default_factory=dict)

    def ranking(self) -> list[dict]:
        """Retourne le classement des moteurs trié par CER croissant."""
        ranked = []
        for report in self.engine_reports:
            ranked.append(
                {
                    "engine": report.engine_name,
                    "mean_cer": report.mean_cer,
                    "mean_wer": report.mean_wer,
                    "documents": len(report.document_results),
                    "failed": report.aggregated_metrics.get("failed_count", 0),
                }
            )
        return sorted(
            ranked,
            key=lambda x: (x["mean_cer"] is None, x["mean_cer"] or float("inf")),
        )

    def as_dict(self) -> dict:
        return {
            "picarones_version": self.picarones_version,
            "run_date": self.run_date,
            "corpus": {
                "name": self.corpus_name,
                "source": self.corpus_source,
                "document_count": self.document_count,
            },
            "ranking": self.ranking(),
            "engine_reports": [r.as_dict() for r in self.engine_reports],
            "metadata": self.metadata,
        }

    def to_json(self, path: str | Path, indent: int = 2) -> Path:
        """Sérialise le benchmark en JSON et l'écrit sur disque.

        Parameters
        ----------
        path:
            Chemin du fichier JSON de sortie.
        indent:
            Indentation JSON (défaut : 2 espaces).

        Returns
        -------
        Path
            Chemin absolu du fichier écrit.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(self.as_dict(), fh, ensure_ascii=False, indent=indent)
        return output_path.resolve()

    @classmethod
    def from_json(cls, path: str | Path) -> dict:
        """Charge un résultat JSON brut depuis le disque (pour le rapport HTML).

        Retourne le dict Python — la reconstruction complète en objets
        est réservée aux sprints suivants.
        """
        with Path(path).open(encoding="utf-8") as fh:
            return json.load(fh)
