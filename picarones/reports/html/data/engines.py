"""Construction du résumé par moteur (``engines_summary``).

Pour chaque ``EngineReport``, accumule métriques agrégées (CER, WER,
MER, WIL), distribution CER pour l'histogramme, métriques avancées
patrimoniales (Sprint 5), distribution d'erreurs (Sprint 10), NER
(Sprint 41), calibration (Sprint 43), profil philologique (Sprint
62), recherchabilité + séquences numériques (Sprint 86), lisibilité
(Sprint 87) et indicateurs pipeline OCR+LLM.

Les coûts (durée moyenne, prix par 1k pages, CO₂) sont ajoutés
ultérieurement par :mod:`picarones.report.report_data.pareto` qui
en a besoin pour calculer les fronts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from picarones.reports.html.data._helpers import safe_round

if TYPE_CHECKING:
    from picarones.evaluation.benchmark_result import BenchmarkResult


def build_engines_summary(benchmark: "BenchmarkResult") -> list[dict]:
    """Retourne la liste des dicts moteur, une entrée par ``EngineReport``."""
    engines_summary: list[dict] = []
    for report in benchmark.engine_reports:
        agg = report.aggregated_metrics
        diplo_agg = agg.get("cer_diplomatic", {})

        line_metrics = report.aggregated_line_metrics
        halluc = report.aggregated_hallucination

        entry: dict = {
            "name": report.engine_name,
            "version": report.engine_version,
            "cer":  safe_round(agg.get("cer", {}).get("mean")),
            "wer":  safe_round(agg.get("wer", {}).get("mean")),
            "mer":  safe_round(agg.get("mer", {}).get("mean")),
            "wil":  safe_round(agg.get("wil", {}).get("mean")),
            "cer_median": safe_round(agg.get("cer", {}).get("median")),
            "cer_min":    safe_round(agg.get("cer", {}).get("min")),
            "cer_max":    safe_round(agg.get("cer", {}).get("max")),
            "doc_count":  agg.get("document_count", 0),
            "failed":     agg.get("failed_count", 0),
            # CER diplomatique (après normalisation historique : ſ=s, u=v, i=j…)
            "cer_diplomatic": safe_round(diplo_agg.get("mean")) if diplo_agg else None,
            "cer_diplomatic_profile": diplo_agg.get("profile"),
            # Distribution pour l'histogramme : liste des CER individuels
            "cer_values": [
                safe_round(dr.metrics.cer)
                for dr in report.document_results
                if dr.metrics.error is None
            ],
            "cer_diplomatic_values": [
                safe_round(dr.metrics.cer_diplomatic)
                for dr in report.document_results
                if dr.metrics.error is None and dr.metrics.cer_diplomatic is not None
            ],
            # Champs pipeline OCR+LLM (vides pour les moteurs OCR seuls)
            "is_pipeline": report.is_pipeline,
            "pipeline_info": report.pipeline_info,
            # Sprint 5 — métriques avancées patrimoniales
            "ligature_score": safe_round(report.ligature_score) if report.ligature_score is not None else None,
            "diacritic_score": safe_round(report.diacritic_score) if report.diacritic_score is not None else None,
            "aggregated_confusion": report.aggregated_confusion,
            "aggregated_taxonomy": report.aggregated_taxonomy,
            "aggregated_structure": report.aggregated_structure,
            "aggregated_image_quality": report.aggregated_image_quality,
            # Sprint 10 — distribution des erreurs + hallucinations VLM
            "gini": safe_round(line_metrics.get("gini_mean")) if line_metrics else None,
            "cer_p90": safe_round(line_metrics.get("percentiles", {}).get("p90")) if line_metrics else None,
            "cer_p99": safe_round(line_metrics.get("percentiles", {}).get("p99")) if line_metrics else None,
            "catastrophic_rate_30": safe_round(line_metrics.get("catastrophic_rate", {}).get("0.3")) if line_metrics else None,
            "aggregated_line_metrics": line_metrics,
            "anchor_score": safe_round(halluc.get("anchor_score_mean")) if halluc else None,
            "length_ratio": safe_round(halluc.get("length_ratio_mean")) if halluc else None,
            "hallucinating_doc_rate": safe_round(halluc.get("hallucinating_doc_rate")) if halluc else None,
            "aggregated_hallucination": halluc,
            # Sprint 41 — NER agrégé (None si aucun calcul effectué)
            "aggregated_ner": report.aggregated_ner,
            # Sprint 43 — calibration agrégée (None si aucune confidence
            # n'a été exposée par le moteur sur ce corpus)
            "aggregated_calibration": report.aggregated_calibration,
            # Sprint 62 — profil philologique agrégé (None si aucun
            # signal philologique sur le corpus pour ce moteur)
            "aggregated_philological": report.aggregated_philological,
            # Sprint 86 — A.II.5 (recherchabilité fuzzy + séquences
            # numériques). None si aucun document n'a de signal.
            "aggregated_searchability": report.aggregated_searchability,
            "aggregated_numerical_sequences": (
                report.aggregated_numerical_sequences
            ),
            # Sprint 87 — A.II.2 (delta Flesch agrégé)
            "aggregated_readability": report.aggregated_readability,
            "is_vlm": report.pipeline_info.get("is_vlm", False) if report.pipeline_info else False,
        }
        engines_summary.append(entry)
    return engines_summary


__all__ = ["build_engines_summary"]
