"""Scatter plots du rapport (Sprint 10).

- ``gini_vs_cer`` — corrélation Gini (concentration des erreurs)
  vs CER moyen, par moteur.
- ``ratio_vs_anchor`` — ratio de longueur OCR/GT vs score d'ancrage,
  par moteur (révèle les hallucinations VLM).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from picarones.report.report_data._helpers import safe_round

if TYPE_CHECKING:
    from picarones.core.results import BenchmarkResult


def build_gini_vs_cer(benchmark: "BenchmarkResult") -> list[dict]:
    """Scatter Gini de la distribution d'erreurs vs CER moyen."""
    gini_vs_cer: list[dict] = []
    for report in benchmark.engine_reports:
        line_metrics = report.aggregated_line_metrics
        gini_val = line_metrics.get("gini_mean") if line_metrics else None
        cer_val = report.mean_cer
        if gini_val is not None and cer_val is not None:
            gini_vs_cer.append({
                "engine": report.engine_name,
                "cer": safe_round(cer_val),
                "gini": safe_round(gini_val),
                "is_pipeline": report.is_pipeline,
            })
    return gini_vs_cer


def build_ratio_vs_anchor(benchmark: "BenchmarkResult") -> list[dict]:
    """Scatter ratio de longueur vs score d'ancrage (détection VLM)."""
    ratio_vs_anchor: list[dict] = []
    for report in benchmark.engine_reports:
        halluc = report.aggregated_hallucination
        if not halluc:
            continue
        ratio_vs_anchor.append({
            "engine": report.engine_name,
            "length_ratio": safe_round(halluc.get("length_ratio_mean", 1.0)),
            "anchor_score": safe_round(halluc.get("anchor_score_mean", 1.0)),
            "hallucinating_rate": safe_round(halluc.get("hallucinating_doc_rate", 0.0)),
            "is_vlm": (
                report.pipeline_info.get("is_vlm", False)
                if report.pipeline_info else False
            ),
        })
    return ratio_vs_anchor


__all__ = ["build_gini_vs_cer", "build_ratio_vs_anchor"]
