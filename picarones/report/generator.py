"""Générateur du rapport HTML interactif auto-contenu.

Le rapport produit est un fichier HTML unique embarquant :
- Toutes les données (JSON inline)
- Chart.js et diff2html (depuis cdnjs)
- CSS et JavaScript de l'application

Vues disponibles
----------------
1. Classement  — tableau triable par colonne (CER, WER, MER, WIL)
2. Galerie     — grille d'images avec badge CER coloré
3. Document    — image zoomable + diff coloré GT / OCR par moteur
4. Analyses    — histogramme CER + graphique radar
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Ressources vendor (embarquées dans le rapport HTML)
# ---------------------------------------------------------------------------

_VENDOR_DIR = Path(__file__).parent / "vendor"


def _load_vendor_js(name: str) -> str:
    """Lit un fichier JS vendorisé et retourne son contenu."""
    p = _VENDOR_DIR / name
    if p.exists():
        return p.read_text(encoding="utf-8")
    return f"/* vendor/{name} non trouvé */"

from picarones.core.results import BenchmarkResult
from picarones.report.diff_utils import compute_char_diff, compute_word_diff
from picarones.core.statistics import (
    compute_pairwise_stats,
    compute_reliability_curve,
    compute_correlation_matrix,
    compute_venn_data,
    cluster_errors,
    bootstrap_ci,
    friedman_test,
    nemenyi_posthoc,
    build_critical_difference_svg,
    compute_pareto_front,
)
from picarones.core.pricing import build_costs_for_benchmark, load_pricing_database
from picarones.core.difficulty import compute_all_difficulties, difficulty_label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_image_b64(image_path: str, max_width: int = 1200) -> str:
    """Lit une image, la redimensionne si besoin, et retourne un data-URI base64."""
    try:
        from PIL import Image
        p = Path(image_path)
        if not p.exists():
            return ""
        with Image.open(p) as img:
            if img.width > max_width:
                ratio = max_width / img.width
                new_h = max(1, int(img.height * ratio))
                img = img.resize((max_width, new_h), Image.LANCZOS)
            # Convertir en RGB pour éviter les problèmes de mode (RGBA, palette…)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            buf = io.BytesIO()
            fmt = "JPEG" if p.suffix.lower() in (".jpg", ".jpeg") else "PNG"
            img.save(buf, format=fmt, optimize=True, quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            mime = "image/jpeg" if fmt == "JPEG" else "image/png"
            return f"data:{mime};base64,{b64}"
    except Exception:
        return ""


def _encode_images_b64_from_result(benchmark: "BenchmarkResult", max_width: int = 1200) -> dict[str, str]:
    """Encode toutes les images d'un BenchmarkResult en base64.

    Returns
    -------
    dict
        ``{doc_id: data_uri}``
    """
    images: dict[str, str] = {}
    if not benchmark.engine_reports:
        return images
    for dr in benchmark.engine_reports[0].document_results:
        if dr.image_path and dr.doc_id not in images:
            uri = _encode_image_b64(dr.image_path, max_width=max_width)
            if uri:
                images[dr.doc_id] = uri
    return images


def _cer_color(cer: float) -> str:
    """Retourne une couleur CSS pour un score CER donné (0→vert, 1→rouge)."""
    from picarones.core.colors import COLOR_GREEN, COLOR_YELLOW, COLOR_ORANGE, COLOR_RED
    if cer < 0.05:
        return COLOR_GREEN
    if cer < 0.15:
        return COLOR_YELLOW
    if cer < 0.30:
        return COLOR_ORANGE
    return COLOR_RED


def _cer_bg(cer: float) -> str:
    from picarones.core.colors import BG_GREEN, BG_YELLOW, BG_ORANGE, BG_RED
    if cer < 0.05:
        return BG_GREEN
    if cer < 0.15:
        return BG_YELLOW
    if cer < 0.30:
        return BG_ORANGE
    return BG_RED


def _pct(v: Optional[float], decimals: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.{decimals}f} %"


def _safe(v: Optional[float], decimals: int = 4) -> float:
    return round(v or 0.0, decimals)


# ---------------------------------------------------------------------------
# Préparation des données
# ---------------------------------------------------------------------------

def _build_report_data(benchmark: BenchmarkResult, images_b64: dict[str, str]) -> dict:
    """Transforme un BenchmarkResult en dict JSON pour le rapport HTML."""

    engines_summary = []
    for report in benchmark.engine_reports:
        agg = report.aggregated_metrics
        diplo_agg = agg.get("cer_diplomatic", {})
        entry: dict = {
            "name": report.engine_name,
            "version": report.engine_version,
            "cer":  _safe(agg.get("cer", {}).get("mean")),
            "wer":  _safe(agg.get("wer", {}).get("mean")),
            "mer":  _safe(agg.get("mer", {}).get("mean")),
            "wil":  _safe(agg.get("wil", {}).get("mean")),
            "cer_median": _safe(agg.get("cer", {}).get("median")),
            "cer_min":    _safe(agg.get("cer", {}).get("min")),
            "cer_max":    _safe(agg.get("cer", {}).get("max")),
            "doc_count":  agg.get("document_count", 0),
            "failed":     agg.get("failed_count", 0),
            # CER diplomatique (après normalisation historique : ſ=s, u=v, i=j…)
            "cer_diplomatic": _safe(diplo_agg.get("mean")) if diplo_agg else None,
            "cer_diplomatic_profile": diplo_agg.get("profile"),
            # Distribution pour l'histogramme : liste des CER individuels
            "cer_values": [
                _safe(dr.metrics.cer)
                for dr in report.document_results
                if dr.metrics.error is None
            ],
            "cer_diplomatic_values": [
                _safe(dr.metrics.cer_diplomatic)
                for dr in report.document_results
                if dr.metrics.error is None and dr.metrics.cer_diplomatic is not None
            ],
            # Champs pipeline OCR+LLM (vides pour les moteurs OCR seuls)
            "is_pipeline": report.is_pipeline,
            "pipeline_info": report.pipeline_info,
            # Sprint 5 — métriques avancées patrimoniales
            "ligature_score": _safe(report.ligature_score) if report.ligature_score is not None else None,
            "diacritic_score": _safe(report.diacritic_score) if report.diacritic_score is not None else None,
            "aggregated_confusion": report.aggregated_confusion,
            "aggregated_taxonomy": report.aggregated_taxonomy,
            "aggregated_structure": report.aggregated_structure,
            "aggregated_image_quality": report.aggregated_image_quality,
            # Sprint 10 — distribution des erreurs + hallucinations VLM
            "gini": _safe(report.aggregated_line_metrics.get("gini_mean")) if report.aggregated_line_metrics else None,
            "cer_p90": _safe(report.aggregated_line_metrics.get("percentiles", {}).get("p90")) if report.aggregated_line_metrics else None,
            "cer_p99": _safe(report.aggregated_line_metrics.get("percentiles", {}).get("p99")) if report.aggregated_line_metrics else None,
            "catastrophic_rate_30": _safe(report.aggregated_line_metrics.get("catastrophic_rate", {}).get("0.3")) if report.aggregated_line_metrics else None,
            "aggregated_line_metrics": report.aggregated_line_metrics,
            "anchor_score": _safe(report.aggregated_hallucination.get("anchor_score_mean")) if report.aggregated_hallucination else None,
            "length_ratio": _safe(report.aggregated_hallucination.get("length_ratio_mean")) if report.aggregated_hallucination else None,
            "hallucinating_doc_rate": _safe(report.aggregated_hallucination.get("hallucinating_doc_rate")) if report.aggregated_hallucination else None,
            "aggregated_hallucination": report.aggregated_hallucination,
            # Sprint 41 — NER agrégé (None si aucun calcul effectué)
            "aggregated_ner": report.aggregated_ner,
            # Sprint 43 — calibration agrégée (None si aucune confidence
            # n'a été exposée par le moteur sur ce corpus)
            "aggregated_calibration": report.aggregated_calibration,
            # Sprint 62 — profil philologique agrégé (None si aucun
            # signal philologique sur le corpus pour ce moteur)
            "aggregated_philological": report.aggregated_philological,
            "is_vlm": report.pipeline_info.get("is_vlm", False) if report.pipeline_info else False,
        }
        engines_summary.append(entry)

    # Documents (vue galerie + vue détail)
    # On collecte tous les doc_ids depuis l'union de tous les moteurs,
    # en préservant l'ordre d'apparition (premier moteur d'abord, puis compléments).
    seen_doc_ids: set[str] = set()
    doc_ids_ordered: list[str] = []
    for report in benchmark.engine_reports:
        for dr in report.document_results:
            if dr.doc_id not in seen_doc_ids:
                seen_doc_ids.add(dr.doc_id)
                doc_ids_ordered.append(dr.doc_id)

    # Index croisé : doc_id → {engine_name → DocumentResult}
    doc_engine_map: dict[str, dict] = {did: {} for did in doc_ids_ordered}
    for report in benchmark.engine_reports:
        for dr in report.document_results:
            doc_engine_map.setdefault(dr.doc_id, {})[report.engine_name] = dr

    documents = []
    for doc_id in doc_ids_ordered:
        engine_results = []
        gt = ""
        image_path = ""
        for engine_name in [r.engine_name for r in benchmark.engine_reports]:
            dr = doc_engine_map[doc_id].get(engine_name)
            if dr is None:
                continue
            gt = dr.ground_truth
            image_path = dr.image_path
            diff_ops = compute_char_diff(dr.ground_truth, dr.hypothesis)
            er_entry: dict = {
                "engine": engine_name,
                "hypothesis": dr.hypothesis,
                "cer": _safe(dr.metrics.cer),
                "cer_diplomatic": _safe(dr.metrics.cer_diplomatic) if dr.metrics.cer_diplomatic is not None else None,
                "wer": _safe(dr.metrics.wer),
                "mer": _safe(dr.metrics.mer),
                "wil": _safe(dr.metrics.wil),
                "duration": dr.duration_seconds,
                "error": dr.engine_error,
                "diff": diff_ops,
            }
            # Champs spécifiques aux pipelines OCR+LLM
            if dr.ocr_intermediate is not None:
                er_entry["ocr_intermediate"] = dr.ocr_intermediate
                er_entry["ocr_diff"] = compute_word_diff(dr.ground_truth, dr.ocr_intermediate)
                er_entry["llm_correction_diff"] = compute_word_diff(dr.ocr_intermediate, dr.hypothesis)
            if dr.pipeline_metadata:
                on = dr.pipeline_metadata.get("over_normalization")
                if on is not None:
                    er_entry["over_normalization"] = on
                er_entry["pipeline_mode"] = dr.pipeline_metadata.get("pipeline_mode")
            # Sprint 5 — métriques avancées par document
            if dr.char_scores is not None:
                er_entry["ligature_score"] = _safe(dr.char_scores.get("ligature", {}).get("score"))
                er_entry["diacritic_score"] = _safe(dr.char_scores.get("diacritic", {}).get("score"))
            if dr.taxonomy is not None:
                er_entry["taxonomy"] = dr.taxonomy
            if dr.structure is not None:
                er_entry["structure"] = dr.structure
            if dr.image_quality is not None:
                er_entry["image_quality"] = dr.image_quality
            # Sprint 10
            if dr.line_metrics is not None:
                er_entry["line_metrics"] = dr.line_metrics
            if dr.hallucination_metrics is not None:
                er_entry["hallucination_metrics"] = dr.hallucination_metrics
            engine_results.append(er_entry)

        # CER moyen sur ce document (pour le badge galerie)
        cer_values = [er["cer"] for er in engine_results if er["error"] is None]
        mean_cer = sum(cer_values) / len(cer_values) if cer_values else 1.0
        best_engine = min(engine_results, key=lambda x: x["cer"], default=None)

        # Script type (depuis metadata par document si disponible)
        script_type = ""
        first_dr = doc_engine_map[doc_id].get(
            benchmark.engine_reports[0].engine_name if benchmark.engine_reports else None
        )
        if first_dr and first_dr.image_quality:
            script_type = first_dr.image_quality.get("script_type", "")

        documents.append({
            "doc_id": doc_id,
            "image_path": image_path,
            "image_b64": images_b64.get(doc_id, ""),
            "ground_truth": gt,
            "mean_cer": _safe(mean_cer),
            "best_engine": best_engine["engine"] if best_engine else "",
            "engine_results": engine_results,
            "script_type": script_type,
        })

    # ── Sprint 7 — Score de difficulté intrinsèque ───────────────────────
    gt_map = {d["doc_id"]: d["ground_truth"] for d in documents}
    cer_map: dict[str, dict[str, float]] = {d["doc_id"]: {} for d in documents}
    iq_map: dict[str, float] = {}
    for report in benchmark.engine_reports:
        for dr in report.document_results:
            cer_map.setdefault(dr.doc_id, {})[report.engine_name] = _safe(dr.metrics.cer)
            if dr.image_quality and "quality_score" in dr.image_quality:
                iq_map[dr.doc_id] = dr.image_quality["quality_score"]
    difficulty_scores = compute_all_difficulties(
        doc_ids=doc_ids_ordered,
        ground_truths=gt_map,
        cer_map=cer_map,
        image_quality_map=iq_map or None,
    )
    # Ajouter difficulty_score à chaque document
    for doc in documents:
        ds = difficulty_scores.get(doc["doc_id"])
        if ds:
            doc["difficulty_score"] = _safe(ds.score)
            doc["difficulty_label"] = difficulty_label(ds.score)
        else:
            doc["difficulty_score"] = 0.5
            doc["difficulty_label"] = "Modéré"

    # ── Sprint 7 — Tests statistiques (Wilcoxon pairwise + bootstrap CI) ─
    engine_cer_map_stats: dict[str, list[float]] = {}
    for report in benchmark.engine_reports:
        vals = [_safe(dr.metrics.cer) for dr in report.document_results if dr.metrics.error is None]
        if vals:
            engine_cer_map_stats[report.engine_name] = vals

    pairwise_stats = compute_pairwise_stats(engine_cer_map_stats)

    # ── Sprint 17 — Friedman + Nemenyi ──────────────────────────────────
    # Alignement strict sur le même ordre de documents : on reconstruit la
    # map à partir des documents communs à tous les moteurs, sinon Friedman
    # n'est pas applicable.
    engine_cer_aligned: dict[str, list[float]] = {}
    common_doc_ids: Optional[set[str]] = None
    for report in benchmark.engine_reports:
        doc_ids = {dr.doc_id for dr in report.document_results if dr.metrics.error is None}
        common_doc_ids = doc_ids if common_doc_ids is None else common_doc_ids & doc_ids
    if common_doc_ids:
        ordered_common = [d for d in doc_ids_ordered if d in common_doc_ids]
        for report in benchmark.engine_reports:
            dr_by_id = {dr.doc_id: dr for dr in report.document_results}
            engine_cer_aligned[report.engine_name] = [
                _safe(dr_by_id[d].metrics.cer) for d in ordered_common
            ]

    friedman = friedman_test(engine_cer_aligned) if engine_cer_aligned else {
        "statistic": 0.0, "p_value": 1.0, "significant": False,
        "df": 0, "n_blocks": 0, "n_engines": 0, "mean_ranks": {},
        "interpretation": "Test de Friedman non calculé — aucun document commun.",
        "error": "no_common_documents",
    }
    nemenyi = nemenyi_posthoc(engine_cer_aligned) if engine_cer_aligned else {
        "alpha": 0.05, "critical_distance": 0.0, "q_alpha": 0.0,
        "n_blocks": 0, "n_engines": 0, "mean_ranks": {},
        "engines_sorted": [], "significant_matrix": [], "tied_groups": [],
        "error": "no_common_documents",
    }

    bootstrap_cis: list[dict] = []
    for engine_name, vals in engine_cer_map_stats.items():
        lo, hi = bootstrap_ci(vals)
        mean_v = sum(vals) / len(vals) if vals else 0.0
        bootstrap_cis.append({
            "engine": engine_name,
            "mean": _safe(mean_v),
            "ci_lower": _safe(lo),
            "ci_upper": _safe(hi),
        })

    # ── Sprint 7 — Courbes de fiabilité ──────────────────────────────────
    reliability_curves: list[dict] = []
    for report in benchmark.engine_reports:
        vals = [_safe(dr.metrics.cer) for dr in report.document_results if dr.metrics.error is None]
        curve = compute_reliability_curve(vals)
        reliability_curves.append({
            "engine": report.engine_name,
            "points": curve,
        })

    # ── Sprint 7 — Venn des erreurs communes / exclusives ────────────────
    # Construire les ensembles d'erreurs par moteur : {engine → set(doc_id:gt_tok:hyp_tok)}
    venn_error_sets: dict[str, set[str]] = {}
    for report in benchmark.engine_reports:
        error_set: set[str] = set()
        for dr in report.document_results:
            ops = compute_word_diff(dr.ground_truth, dr.hypothesis)
            for op in ops:
                if op["op"] in ("replace", "delete", "insert"):
                    key = f"{dr.doc_id}:{op.get('old', op.get('text',''))}:{op.get('new', op.get('text',''))}"
                    error_set.add(key)
        venn_error_sets[report.engine_name] = error_set

    venn_data = compute_venn_data(venn_error_sets)

    # ── Sprint 7 — Clustering des patterns d'erreurs ─────────────────────
    error_data_all: list[dict] = []
    for report in benchmark.engine_reports:
        for dr in report.document_results:
            error_data_all.append({
                "engine": report.engine_name,
                "gt": dr.ground_truth,
                "hypothesis": dr.hypothesis,
            })
    error_clusters_raw = cluster_errors(error_data_all, max_clusters=8)
    error_clusters = [c.as_dict() for c in error_clusters_raw]

    # ── Sprint 7 — Matrice de corrélation ────────────────────────────────
    # Pour chaque moteur : une liste de dicts métriques par document
    correlation_per_engine: list[dict] = []
    for report in benchmark.engine_reports:
        metrics_list = []
        for dr in report.document_results:
            if dr.metrics.error is not None:
                continue
            entry: dict[str, float] = {
                "cer": _safe(dr.metrics.cer),
                "wer": _safe(dr.metrics.wer),
                "mer": _safe(dr.metrics.mer),
                "wil": _safe(dr.metrics.wil),
            }
            if dr.image_quality:
                entry["quality_score"] = _safe(dr.image_quality.get("quality_score", 0.5))
                entry["sharpness"] = _safe(dr.image_quality.get("sharpness_score", 0.5))
            if dr.char_scores:
                entry["ligature"] = _safe(dr.char_scores.get("ligature", {}).get("score", 0.5))
                entry["diacritic"] = _safe(dr.char_scores.get("diacritic", {}).get("score", 0.5))
            metrics_list.append(entry)
        if metrics_list:
            corr = compute_correlation_matrix(metrics_list)
            correlation_per_engine.append({
                "engine": report.engine_name,
                **corr,
            })

    # ── Sprint 10 — Données scatter plots ─────────────────────────────────
    # Scatter 1 : Gini vs CER moyen (moteurs)
    gini_vs_cer = []
    for report in benchmark.engine_reports:
        gini_val = report.aggregated_line_metrics.get("gini_mean") if report.aggregated_line_metrics else None
        cer_val = report.mean_cer
        if gini_val is not None and cer_val is not None:
            gini_vs_cer.append({
                "engine": report.engine_name,
                "cer": _safe(cer_val),
                "gini": _safe(gini_val),
                "is_pipeline": report.is_pipeline,
            })

    # ── Sprint 19 — Coûts et frontière de Pareto ────────────────────────
    # Durée moyenne mesurée par moteur sur le benchmark courant (sec/page)
    durations_by_engine: dict[str, float] = {}
    for report in benchmark.engine_reports:
        durs = [dr.duration_seconds for dr in report.document_results
                if dr.duration_seconds is not None]
        if durs:
            durations_by_engine[report.engine_name] = sum(durs) / len(durs)

    pricing_defaults, _ = load_pricing_database()
    costs_by_engine = build_costs_for_benchmark(
        engines_summary, durations_by_engine,
    )
    # Annoter chaque résumé moteur avec son coût et sa durée
    for entry in engines_summary:
        name = entry["name"]
        entry["mean_duration_seconds"] = round(durations_by_engine.get(name, 0.0), 4) \
            if name in durations_by_engine else None
        entry["cost"] = costs_by_engine.get(name)

    # Front Pareto sur (CER moyen, coût €/1000 pages) — moteurs avec les deux dispos
    pareto_points = []
    for entry in engines_summary:
        cer = entry.get("cer")
        cost = (entry.get("cost") or {}).get("cost_per_1k_pages_eur")
        if cer is None or cost is None:
            continue
        pareto_points.append({"engine": entry["name"], "cer": cer, "cost": cost})
    pareto_front_engines = compute_pareto_front(
        pareto_points, objectives=("cer", "cost"),
    )

    # Front Pareto secondaire (CER, vitesse) pour le toggle "vitesse"
    pareto_speed_points = []
    for entry in engines_summary:
        cer = entry.get("cer")
        dur = entry.get("mean_duration_seconds")
        if cer is None or dur is None:
            continue
        pareto_speed_points.append({"engine": entry["name"], "cer": cer, "dur": dur})
    pareto_front_speed = compute_pareto_front(
        pareto_speed_points, objectives=("cer", "dur"),
    )

    # Front Pareto carbone (CER, g CO2 / 1000 pages) — étiqueté expérimental
    pareto_co2_points = []
    for entry in engines_summary:
        cer = entry.get("cer")
        co2 = (entry.get("cost") or {}).get("co2_per_1k_pages_g")
        if cer is None or co2 is None:
            continue
        pareto_co2_points.append({"engine": entry["name"], "cer": cer, "co2": co2})
    pareto_front_co2 = compute_pareto_front(
        pareto_co2_points, objectives=("cer", "co2"),
    )

    pareto_data = {
        "cost": {
            "points": pareto_points,
            "front": pareto_front_engines,
            "axis_label": "Coût (€ / 1000 pages)",
        },
        "speed": {
            "points": pareto_speed_points,
            "front": pareto_front_speed,
            "axis_label": "Temps moyen (s / page)",
        },
        "co2": {
            "points": pareto_co2_points,
            "front": pareto_front_co2,
            "axis_label": "Empreinte carbone (g CO₂ / 1000 pages, expérimental)",
        },
        "pricing_meta": {
            "last_updated": pricing_defaults.last_updated,
            "currency": pricing_defaults.currency,
            "hourly_rate_local_cpu_eur": pricing_defaults.hourly_rate_local_cpu_eur,
            "hourly_rate_local_gpu_eur": pricing_defaults.hourly_rate_local_gpu_eur,
            "grid_intensity_local": pricing_defaults.grid_intensity_local,
            "grid_intensity_cloud": pricing_defaults.grid_intensity_cloud,
        },
    }

    # Scatter 2 : ratio longueur vs score d'ancrage (moteurs)
    ratio_vs_anchor = []
    for report in benchmark.engine_reports:
        if report.aggregated_hallucination:
            ratio_vs_anchor.append({
                "engine": report.engine_name,
                "length_ratio": _safe(report.aggregated_hallucination.get("length_ratio_mean", 1.0)),
                "anchor_score": _safe(report.aggregated_hallucination.get("anchor_score_mean", 1.0)),
                "hallucinating_rate": _safe(report.aggregated_hallucination.get("hallucinating_doc_rate", 0.0)),
                "is_vlm": report.pipeline_info.get("is_vlm", False) if report.pipeline_info else False,
            })

    return {
        "meta": {
            "corpus_name": benchmark.corpus_name,
            "corpus_source": benchmark.corpus_source,
            "document_count": benchmark.document_count,
            "run_date": benchmark.run_date,
            "picarones_version": benchmark.picarones_version,
            "metadata": benchmark.metadata,
        },
        "ranking": benchmark.ranking(),
        "engines": engines_summary,
        "documents": documents,
        # Sprint 7
        "statistics": {
            "pairwise_wilcoxon": pairwise_stats,
            "bootstrap_cis": bootstrap_cis,
            # Sprint 17 — Friedman multi-moteurs + post-hoc Nemenyi + CDD
            "friedman": friedman,
            "nemenyi": nemenyi,
        },
        "reliability_curves": reliability_curves,
        "venn_data": venn_data,
        "error_clusters": error_clusters,
        "correlation_per_engine": correlation_per_engine,
        # Sprint 10
        "gini_vs_cer": gini_vs_cer,
        "ratio_vs_anchor": ratio_vs_anchor,
        # Sprint 19 — vue Pareto coût/qualité avec variantes d'axe
        "pareto": pareto_data,
        # Sprint 36 — analyse inter-moteurs (divergence taxonomique +
        # complémentarité / oracle).  ``None`` si moins de 2 moteurs.
        "inter_engine_analysis": benchmark.inter_engine_analysis,
        # Sprint 45-46 — stratification par script_type
        "available_strata": benchmark.available_strata(),
        "stratified_ranking": benchmark.stratified_ranking() or None,
        "corpus_homogeneity": benchmark.corpus_homogeneity(),
    }


# ---------------------------------------------------------------------------
# Rendu Jinja2
# ---------------------------------------------------------------------------

# Depuis le Sprint 16, le template monolithique ~3100 lignes a été découpé en
# fichiers externes dans ``picarones/report/templates/`` (CSS, JS, vues HTML).
# ``base.html.j2`` assemble le tout via ``{% include %}``.

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _build_jinja_env():
    """Construit l'Environment Jinja2 pour le rapport.

    Autoescape désactivé : le comportement est équivalent à celui du
    ``_HTML_TEMPLATE.format()`` historique. Les variables injectées
    (JSON embarqué, SVG généré, synthèse narrative issue de templates
    internes) sont toutes produites par le code Picarones et ne nécessitent
    pas d'échappement HTML.
    """
    from jinja2 import Environment, FileSystemLoader
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=False,
        keep_trailing_newline=True,
    )
    return env


# ---------------------------------------------------------------------------
# Classe principale
# ---------------------------------------------------------------------------

class ReportGenerator:
    """Génère un rapport HTML interactif depuis un BenchmarkResult.

    Usage
    -----
    >>> from picarones.report import ReportGenerator
    >>> gen = ReportGenerator(benchmark_result)
    >>> path = gen.generate("rapport.html")
    >>> # Rapport en anglais :
    >>> gen_en = ReportGenerator(benchmark_result, lang="en")
    >>> path_en = gen_en.generate("report.html")
    """

    def __init__(
        self,
        benchmark: BenchmarkResult,
        images_b64: Optional[dict[str, str]] = None,
        lang: str = "fr",
        normalization_profile: Any = None,
    ) -> None:
        """
        Parameters
        ----------
        benchmark:
            Résultat de benchmark à visualiser.
        images_b64:
            Dictionnaire {doc_id: data-URI base64} des images.
            Si None, le générateur cherche dans ``benchmark.metadata["_images_b64"]``.
        lang:
            Code langue du rapport : ``"fr"`` (défaut) ou ``"en"``.
        normalization_profile:
            Profil de normalisation effectivement utilisé (Sprint 27 — pour
            le snapshot de reproductibilité). ``None`` retombe sur le
            profil mentionné dans ``benchmark.metadata["normalization_profile"]``
            s'il est présent, sinon snapshot indisponible.
        """
        self.benchmark = benchmark
        self.images_b64: dict[str, str] = images_b64 or {}
        self.lang = lang
        self.normalization_profile = normalization_profile

        # Récupérer les images embarquées dans les metadata (fixtures)
        if not self.images_b64:
            self.images_b64 = benchmark.metadata.get("_images_b64", {})  # type: ignore[assignment]

        # Sprint 27 — fallback : profil de normalisation depuis les metadata
        if self.normalization_profile is None:
            self.normalization_profile = benchmark.metadata.get("normalization_profile")

    def generate(self, output_path: str | Path) -> Path:
        """Génère le fichier HTML et le sauvegarde sur disque.

        Parameters
        ----------
        output_path:
            Chemin du fichier HTML à écrire.

        Returns
        -------
        Path
            Chemin absolu du fichier généré.
        """
        from picarones.i18n import get_labels

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Auto-encoder les images si aucune n'est fournie
        images_b64 = self.images_b64
        if not images_b64:
            images_b64 = _encode_images_b64_from_result(self.benchmark)

        labels = get_labels(self.lang)
        report_data = _build_report_data(self.benchmark, images_b64)

        # Sprint 27 — snapshots de reproductibilité (pricing, glossaire,
        # profil de normalisation, environnement). Embarqués dans le JSON
        # du rapport pour qu'un lecteur puisse régénérer la synthèse, le
        # Pareto et le glossaire sans accès au code source.
        from picarones.report.snapshot import snapshot_all
        report_data["snapshots"] = snapshot_all(
            lang=self.lang,
            normalization_profile=self.normalization_profile,
        )

        report_json = json.dumps(report_data, ensure_ascii=False, separators=(",", ":"))
        i18n_json = json.dumps(labels, ensure_ascii=False, separators=(",", ":"))
        chartjs_js = _load_vendor_js("chart.umd.min.js")

        # Sprint 17 — rendu SVG du CDD côté serveur (statique, pas de JS)
        cdd_svg = build_critical_difference_svg(
            report_data.get("statistics", {}).get("nemenyi", {}),
        )

        # Sprint 18 — synthèse factuelle narrative (déterministe, sans LLM)
        from picarones.core.narrative import build_synthesis
        synthesis = build_synthesis(report_data, lang=self.lang)

        # Sprint 20 — glossaire contextuel chargé depuis YAML
        from picarones.report.glossary import load_glossary
        glossary = load_glossary(self.lang)
        glossary_json = json.dumps(glossary, ensure_ascii=False, separators=(",", ":"))

        # Sprint 37 — section inter-moteurs (matrice de divergence + oracle)
        # rendue côté serveur. Vide si moins de 2 moteurs ou taxonomie absente.
        from picarones.report.inter_engine_render import (
            build_divergence_matrix_html,
            build_oracle_gap_html,
        )
        divergence_matrix_html = build_divergence_matrix_html(
            report_data.get("inter_engine_analysis"),
            labels=labels,
        )
        oracle_gap_html = build_oracle_gap_html(
            report_data.get("inter_engine_analysis"),
            labels=labels,
        )

        # Sprint 41 — section NER (résumé F1 par moteur + heatmap par
        # catégorie). Vide si aucun moteur n'a de aggregated_ner.
        from picarones.report.ner_render import (
            build_ner_per_category_html,
            build_ner_summary_html,
        )
        ner_summary_html = build_ner_summary_html(
            report_data.get("engines", []),
            labels=labels,
        )
        ner_per_category_html = build_ner_per_category_html(
            report_data.get("engines", []),
            labels=labels,
        )

        # Sprint 43 — section calibration (tableau ECE/MCE + grille de
        # reliability diagrams par moteur). Vide si aucun moteur n'a
        # de aggregated_calibration.
        from picarones.report.calibration_render import (
            build_calibration_summary_html,
            build_reliability_diagrams_grid_html,
        )
        calibration_summary_html = build_calibration_summary_html(
            report_data.get("engines", []),
            labels=labels,
        )
        reliability_diagrams_html = build_reliability_diagrams_grid_html(
            report_data.get("engines", []),
            labels=labels,
        )

        # Sprint 46 — section stratifiée (tableau par strate). Vide si
        # aucune strate disponible.
        from picarones.report.stratification_render import (
            build_stratified_ranking_html,
        )
        stratified_ranking_html = build_stratified_ranking_html(
            report_data.get("stratified_ranking"),
            report_data.get("available_strata"),
            report_data.get("corpus_homogeneity"),
            labels=labels,
        )

        # Sprint 62 — profil philologique (6 sections adaptive sur les
        # modules philologiques Sprints 55-60). Vide si aucun moteur
        # n'a de aggregated_philological.
        from picarones.report.philological_render import (
            build_philological_profile_html,
        )
        philological_profile_html = build_philological_profile_html(
            report_data.get("engines", []),
            labels=labels,
        )

        env = _build_jinja_env()
        template = env.get_template("base.html.j2")
        html = template.render(
            corpus_name=self.benchmark.corpus_name,
            picarones_version=self.benchmark.picarones_version,
            report_data_json=report_json,
            i18n_json=i18n_json,
            html_lang=labels.get("html_lang", "fr"),
            chartjs_inline=chartjs_js,
            critical_difference_svg=cdd_svg,
            friedman=report_data.get("statistics", {}).get("friedman", {}),
            synthesis=synthesis,
            glossary_json=glossary_json,
            divergence_matrix_html=divergence_matrix_html,
            oracle_gap_html=oracle_gap_html,
            ner_summary_html=ner_summary_html,
            ner_per_category_html=ner_per_category_html,
            calibration_summary_html=calibration_summary_html,
            reliability_diagrams_html=reliability_diagrams_html,
            stratified_ranking_html=stratified_ranking_html,
            philological_profile_html=philological_profile_html,
        )

        output_path.write_text(html, encoding="utf-8")
        return output_path.resolve()

    @classmethod
    def from_json(cls, json_path: str | Path, **kwargs) -> "ReportGenerator":
        """Crée un générateur depuis un fichier JSON de résultats.

        Compatible avec les fichiers produits par ``BenchmarkResult.to_json()``.
        Les images base64 doivent être passées via ``kwargs["images_b64"]``
        si elles ne sont pas dans le JSON.
        """
        import json as _json

        data = _json.loads(Path(json_path).read_text(encoding="utf-8"))

        # Reconstruction minimale d'un BenchmarkResult depuis le dict
        from picarones.core.metrics import MetricsResult
        from picarones.core.results import DocumentResult, EngineReport

        engine_reports = []
        for er_data in data.get("engine_reports", []):
            doc_results = []
            for dr_data in er_data.get("document_results", []):
                m = dr_data["metrics"]
                metrics = MetricsResult(
                    cer=m["cer"], cer_nfc=m["cer_nfc"], cer_caseless=m["cer_caseless"],
                    wer=m["wer"], wer_normalized=m["wer_normalized"],
                    mer=m["mer"], wil=m["wil"],
                    reference_length=m["reference_length"],
                    hypothesis_length=m["hypothesis_length"],
                    error=m.get("error"),
                )
                doc_results.append(DocumentResult(
                    doc_id=dr_data["doc_id"],
                    image_path=dr_data["image_path"],
                    ground_truth=dr_data["ground_truth"],
                    hypothesis=dr_data["hypothesis"],
                    metrics=metrics,
                    duration_seconds=dr_data.get("duration_seconds", 0.0),
                    engine_error=dr_data.get("engine_error"),
                ))
            engine_reports.append(EngineReport(
                engine_name=er_data["engine_name"],
                engine_version=er_data.get("engine_version", "unknown"),
                engine_config=er_data.get("engine_config", {}),
                document_results=doc_results,
            ))

        corpus_info = data.get("corpus", {})
        bm = BenchmarkResult(
            corpus_name=corpus_info.get("name", "Corpus"),
            corpus_source=corpus_info.get("source"),
            document_count=corpus_info.get("document_count", 0),
            engine_reports=engine_reports,
            run_date=data.get("run_date", ""),
            picarones_version=data.get("picarones_version", ""),
            metadata=data.get("metadata", {}),
        )

        images_b64 = kwargs.pop("images_b64", {})
        return cls(bm, images_b64=images_b64, **kwargs)
