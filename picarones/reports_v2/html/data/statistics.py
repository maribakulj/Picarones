"""Sections statistiques du rapport (Sprint 7 + Sprint 17).

Construit les blocs :

- ``pairwise_wilcoxon`` — tests de Wilcoxon par paire de moteurs.
- ``bootstrap_cis`` — intervalles de confiance bootstrap par moteur.
- ``friedman`` + ``nemenyi`` — Sprint 17, multi-moteurs.
- ``reliability_curves`` — courbes de fiabilité par moteur.
- ``venn_data`` — diagramme de Venn des erreurs communes/exclusives.
- ``error_clusters`` — clustering des patterns d'erreurs.
- ``correlation_per_engine`` — matrice de corrélation par moteur.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from picarones.evaluation import compute_word_diff
from picarones.evaluation.statistics import (
    bootstrap_ci,
    cluster_errors,
    compute_correlation_matrix,
    compute_pairwise_stats,
    compute_reliability_curve,
    compute_venn_data,
    friedman_test,
    nemenyi_posthoc,
)
from picarones.reports_v2.html.data._helpers import safe_round

if TYPE_CHECKING:
    from picarones.evaluation.benchmark_result import BenchmarkResult


def _engine_cer_values(benchmark: "BenchmarkResult") -> dict[str, list[float]]:
    """Map ``engine_name → [cer_individuels valides]``."""
    out: dict[str, list[float]] = {}
    for report in benchmark.engine_reports:
        vals = [
            safe_round(dr.metrics.cer)
            for dr in report.document_results
            if dr.metrics.error is None
        ]
        if vals:
            out[report.engine_name] = vals
    return out


def build_pairwise_wilcoxon(benchmark: "BenchmarkResult") -> list[dict]:
    """Tests de Wilcoxon par paire de moteurs (Sprint 7)."""
    return compute_pairwise_stats(_engine_cer_values(benchmark))


def build_bootstrap_cis(benchmark: "BenchmarkResult") -> list[dict]:
    """Intervalles de confiance bootstrap par moteur (Sprint 7)."""
    bootstrap_cis: list[dict] = []
    for engine_name, vals in _engine_cer_values(benchmark).items():
        lo, hi = bootstrap_ci(vals)
        mean_v = sum(vals) / len(vals) if vals else 0.0
        bootstrap_cis.append({
            "engine": engine_name,
            "mean": safe_round(mean_v),
            "ci_lower": safe_round(lo),
            "ci_upper": safe_round(hi),
        })
    return bootstrap_cis


def build_friedman_and_nemenyi(benchmark: "BenchmarkResult") -> dict:
    """Test de Friedman + post-hoc Nemenyi (Sprint 17, multi-moteurs).

    Alignement strict sur le même ordre de documents : on reconstruit
    la map à partir des documents communs à tous les moteurs, sinon
    Friedman n'est pas applicable.

    Returns
    -------
    dict
        ``{"friedman": {...}, "nemenyi": {...}}`` à fusionner dans
        la section ``statistics`` du rapport.
    """
    # Liste ordonnée des doc_ids selon l'ordre d'apparition.
    seen: set[str] = set()
    doc_ids_ordered: list[str] = []
    for report in benchmark.engine_reports:
        for dr in report.document_results:
            if dr.doc_id not in seen:
                seen.add(dr.doc_id)
                doc_ids_ordered.append(dr.doc_id)

    common_doc_ids: Optional[set[str]] = None
    for report in benchmark.engine_reports:
        doc_ids = {dr.doc_id for dr in report.document_results if dr.metrics.error is None}
        common_doc_ids = doc_ids if common_doc_ids is None else common_doc_ids & doc_ids

    engine_cer_aligned: dict[str, list[float]] = {}
    if common_doc_ids:
        ordered_common = [d for d in doc_ids_ordered if d in common_doc_ids]
        for report in benchmark.engine_reports:
            dr_by_id = {dr.doc_id: dr for dr in report.document_results}
            engine_cer_aligned[report.engine_name] = [
                safe_round(dr_by_id[d].metrics.cer) for d in ordered_common
            ]

    if engine_cer_aligned:
        friedman = friedman_test(engine_cer_aligned)
        nemenyi = nemenyi_posthoc(engine_cer_aligned)
    else:
        friedman = {
            "statistic": 0.0, "p_value": 1.0, "significant": False,
            "df": 0, "n_blocks": 0, "n_engines": 0, "mean_ranks": {},
            "interpretation": "Test de Friedman non calculé — aucun document commun.",
            "error": "no_common_documents",
        }
        nemenyi = {
            "alpha": 0.05, "critical_distance": 0.0, "q_alpha": 0.0,
            "n_blocks": 0, "n_engines": 0, "mean_ranks": {},
            "engines_sorted": [], "significant_matrix": [], "tied_groups": [],
            "error": "no_common_documents",
        }
    return {"friedman": friedman, "nemenyi": nemenyi}


def build_reliability_curves(benchmark: "BenchmarkResult") -> list[dict]:
    """Courbes de fiabilité par moteur (Sprint 7)."""
    reliability_curves: list[dict] = []
    for report in benchmark.engine_reports:
        vals = [
            safe_round(dr.metrics.cer)
            for dr in report.document_results
            if dr.metrics.error is None
        ]
        curve = compute_reliability_curve(vals)
        reliability_curves.append({
            "engine": report.engine_name,
            "points": curve,
        })
    return reliability_curves


def build_venn_data(benchmark: "BenchmarkResult") -> dict:
    """Venn des erreurs communes / exclusives (Sprint 7).

    Construit les ensembles d'erreurs par moteur :
    ``{engine → set("doc_id:gt_tok:hyp_tok")}``.
    """
    venn_error_sets: dict[str, set[str]] = {}
    for report in benchmark.engine_reports:
        error_set: set[str] = set()
        for dr in report.document_results:
            ops = compute_word_diff(dr.ground_truth, dr.hypothesis)
            for op in ops:
                if op["op"] in ("replace", "delete", "insert"):
                    key = (
                        f"{dr.doc_id}:"
                        f"{op.get('old', op.get('text', ''))}:"
                        f"{op.get('new', op.get('text', ''))}"
                    )
                    error_set.add(key)
        venn_error_sets[report.engine_name] = error_set
    return compute_venn_data(venn_error_sets)


def build_error_clusters(benchmark: "BenchmarkResult") -> list[dict]:
    """Clustering des patterns d'erreurs (Sprint 7)."""
    error_data_all: list[dict] = []
    for report in benchmark.engine_reports:
        for dr in report.document_results:
            error_data_all.append({
                "engine": report.engine_name,
                "gt": dr.ground_truth,
                "hypothesis": dr.hypothesis,
            })
    error_clusters_raw = cluster_errors(error_data_all, max_clusters=8)
    return [c.as_dict() for c in error_clusters_raw]


def build_correlation_per_engine(benchmark: "BenchmarkResult") -> list[dict]:
    """Matrice de corrélation par moteur entre métriques métiers (Sprint 7)."""
    correlation_per_engine: list[dict] = []
    for report in benchmark.engine_reports:
        metrics_list: list[dict[str, float]] = []
        for dr in report.document_results:
            if dr.metrics.error is not None:
                continue
            entry: dict[str, float] = {
                "cer": safe_round(dr.metrics.cer),
                "wer": safe_round(dr.metrics.wer),
                "mer": safe_round(dr.metrics.mer),
                "wil": safe_round(dr.metrics.wil),
            }
            if dr.image_quality:
                entry["quality_score"] = safe_round(dr.image_quality.get("quality_score", 0.5))
                entry["sharpness"] = safe_round(dr.image_quality.get("sharpness_score", 0.5))
            if dr.char_scores:
                entry["ligature"] = safe_round(dr.char_scores.get("ligature", {}).get("score", 0.5))
                entry["diacritic"] = safe_round(dr.char_scores.get("diacritic", {}).get("score", 0.5))
            metrics_list.append(entry)
        if metrics_list:
            corr = compute_correlation_matrix(metrics_list)
            correlation_per_engine.append({
                "engine": report.engine_name,
                **corr,
            })
    return correlation_per_engine


__all__ = [
    "build_pairwise_wilcoxon",
    "build_bootstrap_cis",
    "build_friedman_and_nemenyi",
    "build_reliability_curves",
    "build_venn_data",
    "build_error_clusters",
    "build_correlation_per_engine",
]
