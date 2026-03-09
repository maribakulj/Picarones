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
import math
from pathlib import Path
from typing import Optional

from picarones.core.results import BenchmarkResult
from picarones.report.diff_utils import compute_char_diff, compute_word_diff
from picarones.core.statistics import (
    compute_pairwise_stats,
    compute_reliability_curve,
    compute_correlation_matrix,
    compute_venn_data,
    cluster_errors,
    bootstrap_ci,
)
from picarones.core.difficulty import compute_all_difficulties, difficulty_label, difficulty_color


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
    if cer < 0.05:
        return "#16a34a"   # vert
    if cer < 0.15:
        return "#ca8a04"   # jaune-orangé
    if cer < 0.30:
        return "#ea580c"   # orange
    return "#dc2626"       # rouge


def _cer_bg(cer: float) -> str:
    if cer < 0.05:
        return "#dcfce7"
    if cer < 0.15:
        return "#fef9c3"
    if cer < 0.30:
        return "#ffedd5"
    return "#fee2e2"


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
            "is_vlm": report.pipeline_info.get("is_vlm", False) if report.pipeline_info else False,
        }
        engines_summary.append(entry)

    # Documents (vue galerie + vue détail)
    # On collecte tous les doc_ids depuis le premier moteur
    doc_ids_ordered = []
    if benchmark.engine_reports:
        doc_ids_ordered = [dr.doc_id for dr in benchmark.engine_reports[0].document_results]

    # Index croisé : doc_id → {engine_name → DocumentResult}
    doc_engine_map: dict[str, dict] = {did: {} for did in doc_ids_ordered}
    for report in benchmark.engine_reports:
        for dr in report.document_results:
            doc_engine_map[dr.doc_id][report.engine_name] = dr

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
        },
        "reliability_curves": reliability_curves,
        "venn_data": venn_data,
        "error_clusters": error_clusters,
        "correlation_per_engine": correlation_per_engine,
        # Sprint 10
        "gini_vs_cer": gini_vs_cer,
        "ratio_vs_anchor": ratio_vs_anchor,
    }


# ---------------------------------------------------------------------------
# Template HTML
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="{html_lang}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Picarones — {corpus_name}</title>

<!-- Chart.js -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"
  integrity="sha512-CQBWl4fJHWbryGE+Pc3UJWW1h3Q8IkkvNnPTozals+S49OTEQPoQj/m1LZRM28Wr/7bJCMlpYS3/Zp4hHuWQ=="
  crossorigin="anonymous"></script>

<!-- diff2html -->
<link rel="stylesheet"
  href="https://cdnjs.cloudflare.com/ajax/libs/diff2html/3.4.47/diff2html.min.css"
  crossorigin="anonymous">
<script src="https://cdnjs.cloudflare.com/ajax/libs/diff2html/3.4.47/diff2html.min.js"
  crossorigin="anonymous"></script>

<style>
/* ── Reset & base ─────────────────────────────────────────────────── */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
:root {{
  --bg:         #f1f5f9;
  --surface:    #ffffff;
  --border:     #e2e8f0;
  --primary:    #1e40af;
  --primary-lt: #dbeafe;
  --text:       #1e293b;
  --text-muted: #64748b;
  --ins:        #16a34a;
  --ins-bg:     #dcfce7;
  --del:        #dc2626;
  --del-bg:     #fee2e2;
  --rep:        #c2410c;
  --rep-bg:     #ffedd5;
  --radius:     8px;
  --shadow:     0 1px 3px rgba(0,0,0,.08), 0 1px 2px rgba(0,0,0,.05);
  --nav-h:      56px;
}}
html {{ font-size: 14px; scroll-behavior: smooth; }}
body {{
  font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
}}

/* ── Navigation ───────────────────────────────────────────────────── */
nav {{
  position: fixed; top: 0; left: 0; right: 0; z-index: 100;
  height: var(--nav-h);
  background: var(--primary);
  display: flex; align-items: center;
  padding: 0 1.5rem;
  gap: 2rem;
  box-shadow: 0 2px 8px rgba(0,0,0,.25);
}}
nav .brand {{
  color: #fff; font-weight: 700; font-size: 1.1rem;
  letter-spacing: -.3px; white-space: nowrap;
  display: flex; align-items: center; gap: .4rem;
}}
nav .brand span {{ opacity: .7; font-weight: 400; font-size: .85rem; }}
nav .tabs {{
  display: flex; gap: .25rem; flex: 1;
}}
.tab-btn {{
  background: transparent; border: none; cursor: pointer;
  color: rgba(255,255,255,.7);
  padding: .4rem .9rem; border-radius: 6px;
  font-size: .9rem; font-weight: 500;
  transition: background .15s, color .15s;
}}
.tab-btn:hover  {{ background: rgba(255,255,255,.12); color: #fff; }}
.tab-btn.active {{ background: rgba(255,255,255,.18); color: #fff; }}
nav .meta {{
  color: rgba(255,255,255,.6); font-size: .78rem;
  white-space: nowrap; margin-left: auto;
}}

/* ── Layout ───────────────────────────────────────────────────────── */
main {{
  margin-top: var(--nav-h);
  padding: 1.5rem;
  max-width: 1400px;
  margin-left: auto; margin-right: auto;
}}
.view {{ display: none; }}
.view.active {{ display: block; }}
.card {{
  background: var(--surface);
  border-radius: var(--radius);
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
  padding: 1.25rem;
  margin-bottom: 1.25rem;
}}
h2 {{
  font-size: 1rem; font-weight: 700;
  color: var(--text); margin-bottom: .75rem;
  border-bottom: 2px solid var(--primary-lt);
  padding-bottom: .4rem;
}}
h3 {{ font-size: .9rem; font-weight: 600; margin-bottom: .5rem; }}

/* ── Ranking table ────────────────────────────────────────────────── */
.table-wrap {{ overflow-x: auto; }}
table {{
  width: 100%; border-collapse: collapse;
  font-size: .88rem;
}}
thead tr {{ background: var(--bg); }}
th {{
  text-align: left; padding: .6rem .75rem;
  border-bottom: 2px solid var(--border);
  cursor: pointer; white-space: nowrap;
  color: var(--text-muted); font-weight: 600; font-size: .8rem;
  text-transform: uppercase; letter-spacing: .04em;
  user-select: none;
}}
th.sortable:hover {{ color: var(--primary); }}
th .sort-icon {{ opacity: .4; margin-left: .25rem; font-style: normal; }}
th.sorted .sort-icon {{ opacity: 1; color: var(--primary); }}
td {{
  padding: .55rem .75rem;
  border-bottom: 1px solid var(--border);
  vertical-align: middle;
}}
tr:last-child td {{ border-bottom: none; }}
tbody tr:hover {{ background: #f8fafc; }}
.rank-badge {{
  display: inline-flex; align-items: center; justify-content: center;
  width: 1.6rem; height: 1.6rem; border-radius: 50%;
  font-weight: 700; font-size: .75rem;
  background: var(--primary-lt); color: var(--primary);
}}
.rank-badge.rank-1 {{ background: #fef3c7; color: #92400e; }}
.engine-name {{ font-weight: 600; }}
.engine-version {{ color: var(--text-muted); font-size: .78rem; margin-left: .3rem; }}
.cer-badge {{
  display: inline-block;
  padding: .15rem .5rem; border-radius: 4px;
  font-weight: 600; font-size: .82rem;
}}
.bar {{
  display: inline-block; height: 8px; border-radius: 4px;
  vertical-align: middle; margin-right: .4rem;
}}

/* ── Gallery ──────────────────────────────────────────────────────── */
.gallery-controls {{
  display: flex; align-items: center; gap: .75rem;
  margin-bottom: 1rem; flex-wrap: wrap;
}}
.gallery-controls label {{ font-size: .82rem; color: var(--text-muted); }}
.gallery-controls input[type=range] {{ width: 120px; }}
.gallery-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 1rem;
}}
.gallery-card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  cursor: pointer;
  transition: transform .15s, box-shadow .15s;
}}
.gallery-card:hover {{
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0,0,0,.12);
  border-color: var(--primary);
}}
.gallery-card img, .gallery-card .img-placeholder {{
  width: 100%; aspect-ratio: 4/3; object-fit: cover;
  display: block; background: #e8e0d4;
}}
.img-placeholder {{
  display: flex; align-items: center; justify-content: center;
  font-size: 2rem; color: #94a3b8;
}}
.gallery-card-body {{
  padding: .6rem .75rem;
}}
.gallery-card-title {{
  font-size: .8rem; font-weight: 600; margin-bottom: .35rem;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.gallery-card-badges {{
  display: flex; gap: .3rem; flex-wrap: wrap;
}}
.engine-cer-badge {{
  font-size: .7rem; font-weight: 700;
  padding: .1rem .35rem; border-radius: 3px;
}}

/* ── Document detail ──────────────────────────────────────────────── */
.doc-layout {{
  display: grid;
  grid-template-columns: 220px 1fr;
  gap: 1rem;
  align-items: start;
}}
@media (max-width: 768px) {{
  .doc-layout {{ grid-template-columns: 1fr; }}
}}
.doc-sidebar {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  max-height: calc(100vh - var(--nav-h) - 3rem);
  overflow-y: auto;
  position: sticky;
  top: calc(var(--nav-h) + 1.5rem);
}}
.doc-sidebar-header {{
  padding: .6rem .75rem;
  font-size: .8rem; font-weight: 700; color: var(--text-muted);
  text-transform: uppercase; letter-spacing: .05em;
  border-bottom: 1px solid var(--border);
  position: sticky; top: 0; background: var(--surface);
}}
.doc-list-item {{
  padding: .5rem .75rem;
  cursor: pointer;
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between;
  gap: .5rem;
  transition: background .1s;
}}
.doc-list-item:last-child {{ border-bottom: none; }}
.doc-list-item:hover {{ background: var(--bg); }}
.doc-list-item.active {{ background: var(--primary-lt); }}
.doc-list-label {{ font-size: .82rem; font-weight: 500; }}
.doc-list-cer {{
  font-size: .72rem; font-weight: 700;
  padding: .1rem .3rem; border-radius: 3px;
  flex-shrink: 0;
}}

/* Image zone */
.doc-image-wrap {{
  position: relative; overflow: hidden;
  border: 1px solid var(--border); border-radius: var(--radius);
  background: #e8e0d4; cursor: zoom-in;
  aspect-ratio: 4/3;
}}
.doc-image-wrap img {{
  width: 100%; height: 100%; object-fit: contain;
  transform-origin: center center;
  transition: transform .2s;
  user-select: none;
}}
.doc-image-placeholder {{
  width: 100%; height: 100%;
  display: flex; align-items: center; justify-content: center;
  flex-direction: column; gap: .5rem; color: #94a3b8;
  font-size: .9rem;
}}
.zoom-controls {{
  position: absolute; bottom: .5rem; right: .5rem;
  display: flex; gap: .3rem;
}}
.zoom-btn {{
  background: rgba(0,0,0,.5); color: #fff;
  border: none; border-radius: 4px; cursor: pointer;
  width: 28px; height: 28px; font-size: .9rem;
  display: flex; align-items: center; justify-content: center;
  transition: background .1s;
}}
.zoom-btn:hover {{ background: rgba(0,0,0,.75); }}

/* Diff panels */
.diff-panels {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: .75rem;
  margin-top: .75rem;
}}
.diff-panel {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
}}
.diff-panel-header {{
  padding: .5rem .75rem;
  background: var(--bg);
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between;
}}
.diff-panel-title {{ font-size: .83rem; font-weight: 700; }}
.diff-panel-metrics {{
  display: flex; gap: .4rem;
  font-size: .72rem;
}}
.diff-panel-body {{
  padding: .75rem; font-size: .82rem; line-height: 1.7;
  font-family: 'Georgia', serif;
  max-height: 260px; overflow-y: auto;
}}
/* Diff spans */
.d-eq {{ color: var(--text); }}
.d-ins {{ color: var(--ins); background: var(--ins-bg); border-radius: 2px; padding: 0 1px; }}
.d-del {{ color: var(--del); background: var(--del-bg); border-radius: 2px; padding: 0 1px; text-decoration: line-through; }}
.d-rep-old {{ color: var(--del); background: var(--del-bg); border-radius: 2px 0 0 2px; padding: 0 1px; text-decoration: line-through; }}
.d-rep-new {{ color: var(--rep); background: var(--rep-bg); border-radius: 0 2px 2px 0; padding: 0 1px; }}

/* Side-by-side diff */
.sbs-header {{
  display: flex; align-items: center; justify-content: space-between;
  flex-wrap: wrap; gap: .5rem; margin-bottom: .75rem;
}}
.sbs-engine-select {{
  display: flex; align-items: center; gap: .4rem; font-size: .82rem;
}}
.sbs-engine-select select {{
  border: 1px solid var(--border); border-radius: 4px;
  padding: .2rem .4rem; font-size: .82rem; background: var(--surface);
}}
.sbs-columns {{
  display: grid; grid-template-columns: 1fr 1fr; gap: .75rem;
}}
@media (max-width: 700px) {{
  .sbs-columns {{ grid-template-columns: 1fr; }}
}}
.sbs-col {{
  border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden;
}}
.sbs-col-header {{
  padding: .45rem .75rem;
  display: flex; align-items: center; justify-content: space-between; gap: .5rem;
  font-size: .83rem; font-weight: 700;
}}
.sbs-gt-header {{
  background: #f0fdf4; border-bottom: 1px solid #bbf7d0; color: #15803d;
}}
.sbs-ocr-header {{
  background: #eff6ff; border-bottom: 1px solid #bfdbfe; color: #1d4ed8;
}}
.sbs-col-body {{
  padding: .75rem; font-size: .82rem; line-height: 1.8;
  font-family: 'Georgia', serif;
  max-height: 340px; overflow-y: auto;
  color: var(--text); white-space: pre-wrap; word-break: break-word;
}}
/* Caractères manquants dans GT (orange) */
.d-miss {{ color: #92400e; background: #fef3c7; border-radius: 2px; padding: 0 1px; }}
/* Caractères erronés dans OCR (rouge) */
.d-err  {{ color: var(--del); background: var(--del-bg); border-radius: 2px; padding: 0 1px; }}
/* Insertions dans OCR (vert) */
.d-ins-ocr {{ color: var(--ins); background: var(--ins-bg); border-radius: 2px; padding: 0 1px; }}

/* ── Analyses ─────────────────────────────────────────────────────── */
.charts-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
  gap: 1rem;
}}
.chart-card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem;
}}
.chart-canvas-wrap {{ position: relative; height: 280px; }}

/* ── Pipeline badges ──────────────────────────────────────────────── */
.pipeline-tag {{
  display: inline-flex; align-items: center; gap: .25rem;
  padding: .12rem .38rem;
  border-radius: 4px; font-size: .67rem; font-weight: 700;
  background: #ede9fe; color: #6d28d9;
  letter-spacing: .02em; vertical-align: middle;
}}
.pipeline-tag .pipe-arrow {{ opacity: .7; }}
.over-norm-badge {{
  display: inline-block; padding: .12rem .38rem;
  border-radius: 4px; font-size: .67rem; font-weight: 700;
  background: #fef3c7; color: #b45309;
}}
.over-norm-badge.high {{ background: #fee2e2; color: #b91c1c; }}
/* Vue triple-diff (pipeline) */
.triple-diff-wrap {{
  display: grid; grid-template-columns: 1fr 1fr; gap: .5rem;
  margin-top: .5rem;
}}
.triple-diff-section {{ background: var(--bg); border-radius: 6px; padding: .5rem; }}
.triple-diff-section h5 {{
  font-size: .73rem; font-weight: 700; color: var(--text-muted);
  margin-bottom: .35rem; text-transform: uppercase; letter-spacing: .04em;
}}
.pipeline-steps {{
  display: flex; align-items: center; gap: .3rem; flex-wrap: wrap;
  margin-top: .25rem;
}}
.step-chip {{
  padding: .12rem .4rem; border-radius: 4px; font-size: .68rem; font-weight: 600;
}}
.step-chip.ocr  {{ background: #e0f2fe; color: #0369a1; }}
.step-chip.llm  {{ background: #ede9fe; color: #6d28d9; }}
.step-arrow {{ color: var(--text-muted); font-size: .8rem; }}

/* ── Misc ─────────────────────────────────────────────────────────── */
.badge {{
  display: inline-block; padding: .15rem .45rem;
  border-radius: 4px; font-size: .72rem; font-weight: 700;
}}
.pill {{
  display: inline-block; padding: .1rem .4rem;
  border-radius: 12px; font-size: .72rem;
  background: var(--primary-lt); color: var(--primary);
}}
.empty-state {{
  text-align: center; padding: 3rem 1rem;
  color: var(--text-muted); font-size: .9rem;
}}
.legend-dot {{
  display: inline-block; width: 8px; height: 8px;
  border-radius: 50%; margin-right: .3rem;
}}
.legend-row {{
  display: flex; align-items: center; gap: .4rem;
  font-size: .78rem; color: var(--text-muted);
}}
footer {{
  text-align: center; padding: 1.5rem;
  color: var(--text-muted); font-size: .75rem;
  border-top: 1px solid var(--border); margin-top: 2rem;
}}
.stat-row {{
  display: flex; gap: 1.5rem; flex-wrap: wrap; margin-bottom: .75rem;
}}
.stat {{
  background: var(--bg); border-radius: 6px; padding: .4rem .75rem;
  font-size: .8rem;
}}
.stat b {{ color: var(--primary); }}

/* ── Difficulty badge ─────────────────────────────────────────── */
.diff-badge {{
  display: inline-flex; align-items: center; gap: .2rem;
  padding: .1rem .4rem; border-radius: 4px;
  font-size: .7rem; font-weight: 700;
}}

/* ── Presentation mode ────────────────────────────────────────── */
.btn-present {{
  background: rgba(255,255,255,.15); border: 1px solid rgba(255,255,255,.3);
  color: #fff; padding: .3rem .7rem; border-radius: 6px;
  font-size: .8rem; font-weight: 600; cursor: pointer;
  transition: background .15s;
  white-space: nowrap;
}}
.btn-present:hover {{ background: rgba(255,255,255,.28); }}
.btn-present.active {{ background: rgba(255,255,255,.35); }}
.btn-export-csv {{
  background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.25);
  color: rgba(255,255,255,.85); padding: .3rem .7rem; border-radius: 6px;
  font-size: .8rem; font-weight: 600; cursor: pointer;
  transition: background .15s; white-space: nowrap;
}}
.btn-export-csv:hover {{ background: rgba(255,255,255,.22); color:#fff; }}
body.present-mode .technical {{ display: none !important; }}
body.present-mode .chart-card {{ page-break-inside: avoid; }}
body.present-mode nav .meta {{ display: none; }}

/* ── Cluster cards ─────────────────────────────────────────────── */
.cluster-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: .75rem; margin-top: .75rem;
}}
.cluster-card {{
  background: var(--bg); border: 1px solid var(--border);
  border-radius: var(--radius); padding: .75rem;
}}
.cluster-label {{ font-weight: 700; font-size: .88rem; color: var(--primary); margin-bottom: .3rem; }}
.cluster-count {{ font-size: .75rem; color: var(--text-muted); margin-bottom: .5rem; }}
.cluster-examples {{
  display: flex; flex-direction: column; gap: .2rem;
}}
.cluster-ex {{
  font-family: monospace; font-size: .78rem;
  background: var(--surface); border-radius: 3px; padding: .15rem .35rem;
  display: flex; align-items: center; gap: .35rem; color: var(--text-muted);
}}
.cluster-ex .ex-old {{ color: var(--del); background: var(--del-bg); border-radius: 2px; padding: 0 3px; }}
.cluster-ex .ex-new {{ color: var(--rep); background: var(--rep-bg); border-radius: 2px; padding: 0 3px; }}

/* ── Statistical tests table ─────────────────────────────────────*/
.stat-sig {{ color: #dc2626; font-weight: 700; }}
.stat-ns  {{ color: #64748b; }}

/* ── Venn diagram ────────────────────────────────────────────────*/
.venn-wrap {{ display: flex; justify-content: center; padding: 1rem; }}

/* ── Correlation matrix ──────────────────────────────────────────*/
.corr-table {{ border-collapse: collapse; font-size: .8rem; margin: .5rem auto; }}
.corr-table th, .corr-table td {{
  padding: .35rem .5rem; text-align: center; border: 1px solid var(--border);
  min-width: 60px;
}}
.corr-table th {{ background: var(--bg); font-weight: 600; font-size: .75rem; }}

/* ── Sprint 10 — heatmap erreurs ─────────────────────────────────*/
.heatmap-wrap {{
  display: flex; gap: 3px; align-items: flex-end;
  height: 60px; margin: .5rem 0;
}}
.heatmap-bar {{
  flex: 1; border-radius: 3px 3px 0 0;
  min-height: 4px;
  transition: opacity .15s;
}}
.heatmap-bar:hover {{ opacity: .75; }}
.heatmap-labels {{
  display: flex; justify-content: space-between;
  font-size: .65rem; color: var(--text-muted); margin-top: .15rem;
}}

/* ── Sprint 10 — hallucination badge ─────────────────────────────*/
.hallucination-badge {{
  display: inline-flex; align-items: center; gap: .25rem;
  padding: .15rem .45rem; border-radius: 4px;
  font-size: .72rem; font-weight: 700;
  background: #fce7f3; color: #9d174d;
  border: 1px solid #fbcfe8;
}}
.hallucination-badge.ok {{
  background: #f0fdf4; color: #15803d;
  border-color: #bbf7d0;
}}

/* ── Sprint 10 — bloc halluciné ──────────────────────────────────*/
.halluc-block {{
  background: #fce7f3; border: 1px solid #f9a8d4;
  border-radius: 4px; padding: .35rem .6rem;
  margin: .25rem 0; font-size: .78rem;
  font-family: 'Georgia', serif; color: #9d174d;
}}
.halluc-block-meta {{
  font-size: .65rem; color: #be185d; font-family: system-ui, sans-serif;
  margin-bottom: .15rem; font-weight: 600;
}}

/* ── Sprint 10 — percentile bars ─────────────────────────────────*/
.pct-bars {{ display: flex; flex-direction: column; gap: .25rem; margin: .4rem 0; }}
.pct-bar-row {{ display: flex; align-items: center; gap: .4rem; font-size: .72rem; }}
.pct-bar-label {{ width: 2.5rem; color: var(--text-muted); text-align: right; flex-shrink: 0; }}
.pct-bar-track {{
  flex: 1; height: 8px; background: var(--bg);
  border-radius: 4px; overflow: hidden;
}}
.pct-bar-fill {{ height: 100%; border-radius: 4px; }}
.pct-bar-val {{ width: 3rem; color: var(--text); font-weight: 600; }}
</style>
</head>

<body>

<!-- ── Navigation ─────────────────────────────────────────────────── -->
<nav>
  <div class="brand">
    Picarones
    <span data-i18n="nav_report">| rapport OCR</span>
  </div>
  <div class="tabs">
    <button class="tab-btn active" onclick="showView('ranking')" data-i18n="tab_ranking">Classement</button>
    <button class="tab-btn" onclick="showView('gallery')" data-i18n="tab_gallery">Galerie</button>
    <button class="tab-btn" onclick="showView('document')" data-i18n="tab_document">Document</button>
    <button class="tab-btn" onclick="showView('characters')" data-i18n="tab_characters">Caractères</button>
    <button class="tab-btn" onclick="showView('analyses')" data-i18n="tab_analyses">Analyses</button>
  </div>
  <div class="meta" id="nav-meta">—</div>
  <button class="btn-export-csv" onclick="exportCSV()" title="⬇ CSV">⬇ CSV</button>
  <button class="btn-present" id="btn-present" onclick="togglePresentMode()" data-i18n="btn_present">⊞ Présentation</button>
</nav>

<!-- ── Main ───────────────────────────────────────────────────────── -->
<main>

<!-- ════ Vue 1 : Classement ════════════════════════════════════════ -->
<div id="view-ranking" class="view active">
  <div class="card">
    <h2 data-i18n="h_ranking">Classement des moteurs</h2>
    <div class="stat-row" id="ranking-stats"></div>
    <div class="table-wrap">
      <table id="ranking-table">
        <thead>
          <tr>
            <th data-col="rank" class="sortable sorted" data-dir="asc" data-i18n="col_rank">#<i class="sort-icon">↑</i></th>
            <th data-col="name" class="sortable" data-i18n="col_engine">Concurrent<i class="sort-icon">↕</i></th>
            <th data-col="cer"  class="sortable" data-i18n="col_cer">CER exact<i class="sort-icon">↕</i></th>
            <th data-col="cer_diplomatic" class="sortable" id="th-cer-diplo" data-i18n="col_cer_diplo">CER diplo.<i class="sort-icon">↕</i></th>
            <th data-col="wer"  class="sortable" data-i18n="col_wer">WER<i class="sort-icon">↕</i></th>
            <th data-col="mer"  class="sortable" data-i18n="col_mer">MER<i class="sort-icon">↕</i></th>
            <th data-col="wil"  class="sortable" data-i18n="col_wil">WIL<i class="sort-icon">↕</i></th>
            <th data-col="ligature_score" class="sortable" id="th-ligatures" data-i18n="col_ligatures">Ligatures<i class="sort-icon">↕</i></th>
            <th data-col="diacritic_score" class="sortable" id="th-diacritics" data-i18n="col_diacritics">Diacritiques<i class="sort-icon">↕</i></th>
            <th data-col="gini" class="sortable" id="th-gini" data-i18n="col_gini">Gini<i class="sort-icon">↕</i></th>
            <th data-col="anchor_score" class="sortable" id="th-anchor" data-i18n="col_anchor">Ancrage<i class="sort-icon">↕</i></th>
            <th data-i18n="col_cer_median">CER médian</th>
            <th data-i18n="col_cer_min">CER min</th>
            <th data-i18n="col_cer_max">CER max</th>
            <th id="th-overnorm" data-i18n="col_overnorm">Sur-norm.</th>
            <th data-i18n="col_docs">Docs</th>
          </tr>
        </thead>
        <tbody id="ranking-tbody"></tbody>
      </table>
    </div>
    <div class="stat-row" style="margin-top:.75rem">
      <div class="legend-row">
        <span class="legend-dot" style="background:#16a34a"></span>CER &lt; 5 %
      </div>
      <div class="legend-row">
        <span class="legend-dot" style="background:#ca8a04"></span>5–15 %
      </div>
      <div class="legend-row">
        <span class="legend-dot" style="background:#ea580c"></span>15–30 %
      </div>
      <div class="legend-row">
        <span class="legend-dot" style="background:#dc2626"></span>&gt; 30 %
      </div>
    </div>
  </div>
</div>

<!-- ════ Vue 2 : Galerie ═══════════════════════════════════════════ -->
<div id="view-gallery" class="view">
  <div class="card">
    <h2 data-i18n="h_gallery">Galerie des documents</h2>
    <div class="gallery-controls">
      <label><span data-i18n="gallery_sort_label">Trier par :</span>
        <select id="gallery-sort" onchange="renderGallery()">
          <option value="doc_id" data-i18n-opt="gallery_sort_id">Identifiant</option>
          <option value="mean_cer" data-i18n-opt="gallery_sort_cer">CER moyen</option>
          <option value="difficulty_score" data-i18n-opt="gallery_sort_difficulty">Difficulté</option>
          <option value="best_engine" data-i18n-opt="gallery_sort_best">Meilleur moteur</option>
        </select>
      </label>
      <label><span data-i18n="gallery_filter_cer_label">Filtrer CER &gt;</span>
        <input type="number" id="gallery-filter-cer" min="0" max="100" value="0" step="1"
          style="width:60px" onchange="renderGallery()"> %
      </label>
      <label><span data-i18n="gallery_filter_engine_label">Moteur :</span>
        <select id="gallery-engine-select" onchange="renderGallery()">
          <option value="" data-i18n-opt="gallery_filter_all">Tous</option>
        </select>
      </label>
    </div>
    <div id="gallery-grid" class="gallery-grid"></div>
    <div id="gallery-empty" class="empty-state" style="display:none" data-i18n="gallery_empty">
      Aucun document ne correspond aux filtres.
    </div>
  </div>
</div>

<!-- ════ Vue 3 : Document ══════════════════════════════════════════ -->
<div id="view-document" class="view">
  <div class="doc-layout">
    <!-- Sidebar -->
    <aside class="doc-sidebar">
      <div class="doc-sidebar-header" data-i18n="doc_sidebar_header">Documents</div>
      <div id="doc-list"></div>
    </aside>

    <!-- Contenu principal -->
    <div>
      <div class="card" id="doc-detail-header">
        <div style="display:flex; align-items:baseline; justify-content:space-between; flex-wrap:wrap; gap:.5rem">
          <h2 id="doc-detail-title" data-i18n="doc_title_default">Sélectionner un document</h2>
          <div class="stat-row" id="doc-detail-metrics"></div>
        </div>
      </div>

      <!-- Image zoomable -->
      <div class="card">
        <h3 data-i18n="h_image">Image originale</h3>
        <div class="doc-image-wrap" id="doc-image-wrap"
          onwheel="handleZoom(event)"
          onmousedown="startDrag(event)"
          onmousemove="doDrag(event)"
          onmouseup="endDrag()"
          onmouseleave="endDrag()">
          <div class="doc-image-placeholder" id="doc-image-placeholder">
            <span style="font-size:2rem">🖼</span>
            <span>Sélectionnez un document</span>
          </div>
          <img id="doc-image" src="" alt="Image du document" style="display:none">
          <div class="zoom-controls">
            <button class="zoom-btn" onclick="zoom(1.25)" title="Zoom +">+</button>
            <button class="zoom-btn" onclick="zoom(0.8)"  title="Zoom −">−</button>
            <button class="zoom-btn" onclick="resetZoom()" title="Réinitialiser">↺</button>
          </div>
        </div>
      </div>

      <!-- Diff côte à côte GT / OCR -->
      <div class="card" id="doc-sidebyside-card">
        <div class="sbs-header">
          <h3 data-i18n="h_diff">Comparaison GT / OCR</h3>
          <div class="sbs-engine-select" id="sbs-engine-select" style="display:none">
            <label data-i18n="sbs_engine_label">Concurrent :</label>
            <select id="sbs-engine-dropdown" onchange="renderSideBySide(currentDocId)"></select>
          </div>
        </div>
        <div class="sbs-columns" id="sbs-columns">
          <div class="sbs-col sbs-col-gt">
            <div class="sbs-col-header sbs-gt-header">
              <span>✓ Vérité terrain (GT)</span>
            </div>
            <div class="sbs-col-body" id="sbs-gt-body">—</div>
          </div>
          <div class="sbs-col sbs-col-ocr">
            <div class="sbs-col-header sbs-ocr-header" id="sbs-ocr-header">
              <span id="sbs-ocr-engine-name">OCR</span>
              <span class="cer-badge" id="sbs-ocr-cer" style="display:none"></span>
            </div>
            <div class="sbs-col-body" id="sbs-ocr-body">—</div>
          </div>
        </div>
        <!-- Pipeline triple-diff (affiché en dessous si applicable) -->
        <div id="sbs-triple-diff" style="display:none"></div>
      </div>

      <!-- Sprint 10 — Distribution CER par ligne -->
      <div class="card" id="doc-line-metrics-card" style="display:none">
        <h3 data-i18n="h_line_metrics">Distribution des erreurs par ligne</h3>
        <div id="doc-line-metrics-content"></div>
      </div>

      <!-- Sprint 10 — Hallucinations détectées -->
      <div class="card" id="doc-hallucination-card" style="display:none">
        <h3 data-i18n="h_hallucination">Analyse des hallucinations</h3>
        <div id="doc-hallucination-content"></div>
      </div>
    </div>
  </div>
</div>

<!-- ════ Vue 4 : Analyses ══════════════════════════════════════════ -->
<div id="view-analyses" class="view">
  <div class="charts-grid">

    <div class="chart-card">
      <h3 data-i18n="h_cer_dist">Distribution du CER par moteur</h3>
      <div class="chart-canvas-wrap">
        <canvas id="chart-cer-hist"></canvas>
      </div>
    </div>

    <div class="chart-card">
      <h3 data-i18n="h_radar">Profil des moteurs (radar)</h3>
      <div class="chart-canvas-wrap">
        <canvas id="chart-radar"></canvas>
      </div>
      <div style="font-size:.72rem;color:var(--text-muted);margin-top:.5rem" data-i18n="radar_note">
        Axe radar : CER, WER, MER, WIL — valeurs inversées (plus c'est haut, meilleur est le moteur).
      </div>
    </div>

    <div class="chart-card">
      <h3 data-i18n="h_cer_doc">CER par document (tous moteurs)</h3>
      <div class="chart-canvas-wrap">
        <canvas id="chart-cer-doc"></canvas>
      </div>
    </div>

    <div class="chart-card">
      <h3 data-i18n="h_duration">Temps d'exécution moyen (secondes/document)</h3>
      <div class="chart-canvas-wrap">
        <canvas id="chart-duration"></canvas>
      </div>
    </div>

    <div class="chart-card">
      <h3 data-i18n="h_quality_cer">Qualité image ↔ CER (scatter plot)</h3>
      <div class="chart-canvas-wrap">
        <canvas id="chart-quality-cer"></canvas>
      </div>
      <div style="font-size:.72rem;color:var(--text-muted);margin-top:.4rem" data-i18n="quality_cer_note">
        Chaque point = un document. Axe X = score qualité image [0–1]. Axe Y = CER. Corrélation négative attendue.
      </div>
    </div>

    <div class="chart-card" style="grid-column:1/-1">
      <h3 data-i18n="h_taxonomy">Taxonomie des erreurs par moteur</h3>
      <div class="chart-canvas-wrap" style="max-height:300px">
        <canvas id="chart-taxonomy"></canvas>
      </div>
      <div style="font-size:.72rem;color:var(--text-muted);margin-top:.4rem" data-i18n="taxonomy_note">
        Distribution des classes d'erreurs (classes 1–9 de la taxonomie Picarones).
      </div>
    </div>

    <!-- Sprint 7 — Courbe de fiabilité -->
    <div class="chart-card" style="grid-column:1/-1">
      <h3 data-i18n="h_reliability">Courbes de fiabilité</h3>
      <div class="chart-canvas-wrap" style="max-height:300px">
        <canvas id="chart-reliability"></canvas>
      </div>
      <div style="font-size:.72rem;color:var(--text-muted);margin-top:.4rem" data-i18n="reliability_note">
        Pour les X% documents les plus faciles (triés par CER croissant), quel est le CER moyen cumulé ?
        Une courbe basse = moteur performant même sur les documents faciles.
      </div>
    </div>

    <!-- Sprint 7 — Intervalles de confiance -->
    <div class="chart-card">
      <h3 data-i18n="h_bootstrap">Intervalles de confiance à 95 % (bootstrap)</h3>
      <div class="chart-canvas-wrap">
        <canvas id="chart-bootstrap-ci"></canvas>
      </div>
      <div style="font-size:.72rem;color:var(--text-muted);margin-top:.4rem" data-i18n="bootstrap_note">
        IC à 95% sur le CER moyen par moteur (1000 itérations bootstrap).
      </div>
    </div>

    <!-- Sprint 7 — Diagramme de Venn -->
    <div class="chart-card">
      <h3 data-i18n="h_venn">Erreurs communes / exclusives (Venn)</h3>
      <div id="venn-container" style="min-height:260px;display:flex;align-items:center;justify-content:center"></div>
      <div style="font-size:.72rem;color:var(--text-muted);margin-top:.4rem technical" data-i18n="venn_note">
        Intersection des ensembles d'erreurs entre les 2 ou 3 premiers concurrents.
        Erreurs communes = segments partagés.
      </div>
    </div>

    <!-- Sprint 7 — Tests de Wilcoxon -->
    <div class="chart-card technical">
      <h3 data-i18n="h_pairwise">Tests de Wilcoxon — comparaisons par paires</h3>
      <div id="wilcoxon-table-container" style="overflow-x:auto"></div>
      <div style="font-size:.72rem;color:var(--text-muted);margin-top:.4rem" data-i18n="pairwise_note">
        Test signé-rangé de Wilcoxon (non-paramétrique). Seuil α = 0.05.
      </div>
    </div>

    <!-- Sprint 7 — Clustering des erreurs -->
    <div class="chart-card" style="grid-column:1/-1">
      <h3 data-i18n="h_clusters">Clustering des patterns d'erreurs</h3>
      <div id="error-clusters-container"></div>
    </div>

    <!-- Sprint 10 — Scatter Gini vs CER moyen -->
    <div class="chart-card">
      <h3 data-i18n="h_gini_cer">Gini vs CER moyen <span style="font-size:.72rem;font-weight:400;color:var(--text-muted)" data-i18n="gini_cer_ideal">— idéal : bas-gauche</span></h3>
      <div class="chart-canvas-wrap">
        <canvas id="chart-gini-cer"></canvas>
      </div>
      <div style="font-size:.72rem;color:var(--text-muted);margin-top:.4rem" data-i18n="gini_cer_note">
        Axe X = CER moyen, Axe Y = coefficient de Gini. Un moteur idéal a CER bas ET Gini bas (erreurs rares et uniformes).
      </div>
    </div>

    <!-- Sprint 10 — Scatter ratio longueur vs ancrage -->
    <div class="chart-card">
      <h3 data-i18n="h_ratio_anchor">Ratio longueur vs ancrage <span style="font-size:.72rem;font-weight:400;color:var(--text-muted)" data-i18n="ratio_anchor_subtitle">— hallucinations VLM</span></h3>
      <div class="chart-canvas-wrap">
        <canvas id="chart-ratio-anchor"></canvas>
      </div>
      <div style="font-size:.72rem;color:var(--text-muted);margin-top:.4rem" data-i18n="ratio_anchor_note">
        Axe X = score d'ancrage trigrammes [0–1]. Axe Y = ratio longueur sortie/GT.
        Zone ⚠️ : ancrage &lt; 0.5 ou ratio &gt; 1.2 → hallucinations probables.
      </div>
    </div>

    <!-- Sprint 7 — Matrice de corrélation -->
    <div class="chart-card technical" style="grid-column:1/-1">
      <h3 data-i18n="h_correlation">Matrice de corrélation entre métriques</h3>
      <div style="margin-bottom:.5rem">
        <label style="font-size:.82rem;font-weight:600"><span data-i18n="corr_engine_label">Moteur :</span>
          <select id="corr-engine-select" onchange="renderCorrelationMatrix()"
            style="padding:.25rem .5rem;border-radius:6px;border:1px solid var(--border);margin-left:.25rem"></select>
        </label>
      </div>
      <div id="corr-matrix-container" style="overflow-x:auto"></div>
      <div style="font-size:.72rem;color:var(--text-muted);margin-top:.4rem" data-i18n="corr_note">
        Coefficient de Pearson entre les métriques CER, WER, qualité image, ligatures, diacritiques.
        Vert = corrélation positive, Rouge = corrélation négative.
      </div>
    </div>

  </div>
</div>

<!-- ════ Vue 5 : Caractères ════════════════════════════════════════ -->
<div id="view-characters" class="view">
  <div class="card">
    <h2 data-i18n="h_characters">Analyse des caractères</h2>

    <!-- Sélecteur de moteur -->
    <div class="stat-row" style="margin-bottom:1rem">
      <label for="char-engine-select" style="font-weight:600;margin-right:.5rem" data-i18n="char_engine_label">Moteur :</label>
      <select id="char-engine-select" onchange="renderCharView()"
        style="padding:.35rem .7rem;border-radius:6px;border:1px solid var(--border)"></select>
    </div>

    <!-- Scores ligatures / diacritiques -->
    <div class="stat-row" id="char-scores-row" style="gap:1.5rem;margin-bottom:1.5rem"></div>

    <!-- Matrice de confusion unicode -->
    <h3 style="margin-bottom:.75rem">Matrice de confusion unicode
      <span style="font-size:.75rem;font-weight:400;color:var(--text-muted)">
        — substitutions les plus fréquentes (caractère GT → caractère OCR)
      </span>
    </h3>
    <div id="confusion-heatmap" style="overflow-x:auto;margin-bottom:1.5rem"></div>

    <!-- Détail ligatures par type -->
    <h3 style="margin-bottom:.75rem">Reconnaissance des ligatures</h3>
    <div id="ligature-detail" style="margin-bottom:1.5rem"></div>

    <!-- Taxonomie détaillée -->
    <h3 style="margin-bottom:.75rem">Distribution taxonomique des erreurs</h3>
    <div id="taxonomy-detail"></div>
  </div>
</div>

</main>

<footer>
  <span data-i18n="footer_by">par Picarones</span> v{picarones_version}
  — <span id="footer-date"></span>
</footer>

<!-- ── Données embarquées ──────────────────────────────────────────── -->
<script>
const DATA = {report_data_json};
const I18N = {i18n_json};
</script>

<!-- ── Application ────────────────────────────────────────────────── -->
<script>
'use strict';

// ── Palette couleurs par moteur ──────────────────────────────────
const PALETTE = [
  '#2563eb','#dc2626','#16a34a','#ca8a04','#7c3aed',
  '#0891b2','#c2410c','#0f766e','#9333ea','#b45309',
];
function engineColor(idx) {{ return PALETTE[idx % PALETTE.length]; }}

// ── Navigation ──────────────────────────────────────────────────
let currentView = 'ranking';
function _switchView(name) {{
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('view-' + name).classList.add('active');
  // Activer le bon onglet nav
  const tabMap = {{ranking:'classement',gallery:'galerie',document:'document',characters:'caract',analyses:'analyses'}};
  const prefix = tabMap[name] || name;
  document.querySelectorAll('.tab-btn').forEach(b => {{
    if (b.textContent.toLowerCase().startsWith(prefix.toLowerCase())) b.classList.add('active');
  }});
  currentView = name;
  if (name === 'analyses' && !chartsBuilt) buildCharts();
  if (name === 'characters' && !charViewBuilt) initCharView();
}}
function showView(name) {{
  _switchView(name);
  updateURL(name);
}}

// ── Formatage ───────────────────────────────────────────────────
function pct(v, d=2) {{
  if (v === null || v === undefined) return '—';
  return (v * 100).toFixed(d) + ' %';
}}
function cerColor(v) {{
  if (v < 0.05) return '#16a34a';
  if (v < 0.15) return '#ca8a04';
  if (v < 0.30) return '#ea580c';
  return '#dc2626';
}}
function cerBg(v) {{
  if (v < 0.05) return '#dcfce7';
  if (v < 0.15) return '#fef9c3';
  if (v < 0.30) return '#ffedd5';
  return '#fee2e2';
}}
function esc(s) {{
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}}

// ── Diff renderer ──────────────────────────────────────────────
function renderDiff(ops) {{
  if (!ops || !ops.length) return '<em style="color:var(--text-muted)">— aucune sortie —</em>';
  return ops.map(op => {{
    if (op.op === 'equal')
      return '<span class="d-eq">' + esc(op.text) + '</span>';
    if (op.op === 'insert')
      return '<span class="d-ins" title="Insertion OCR">' + esc(op.text) + '</span>';
    if (op.op === 'delete')
      return '<span class="d-del" title="Suppression (présent GT)">' + esc(op.text) + '</span>';
    if (op.op === 'replace')
      return '<span class="d-rep-old" title="Remplacement">' + esc(op.old) + '</span>'
           + '<span class="d-rep-new">' + esc(op.new) + '</span>';
    return '';
  }}).join(' ');
}}

// ── Rendu côte à côte (char-level) ──────────────────────────────────
function renderSideBySide(docId) {{
  const doc = DATA.documents.find(d => d.doc_id === docId);
  if (!doc) return;

  const sel = document.getElementById('sbs-engine-dropdown');
  const engineIdx = sel && sel.value !== '' ? parseInt(sel.value, 10) : 0;
  const er = doc.engine_results[engineIdx];
  if (!er) return;

  const ops = er.diff || [];

  // Construire le HTML GT (gauche) et OCR (droite) depuis les mêmes ops
  let gtHtml = '', ocrHtml = '';
  ops.forEach(op => {{
    if (op.op === 'equal') {{
      const t = esc(op.text);
      gtHtml  += t;
      ocrHtml += t;
    }} else if (op.op === 'delete') {{
      // Présent dans GT, absent de l'OCR → orange dans GT
      gtHtml += '<span class="d-miss" title="Absent de l\'OCR">' + esc(op.text) + '</span>';
    }} else if (op.op === 'insert') {{
      // Présent dans OCR, absent du GT → vert dans OCR
      ocrHtml += '<span class="d-ins-ocr" title="Insertion OCR">' + esc(op.text) + '</span>';
    }} else if (op.op === 'replace') {{
      // Substitution : orange dans GT, rouge dans OCR
      gtHtml  += '<span class="d-miss" title="Différent dans l\'OCR">' + esc(op.old) + '</span>';
      ocrHtml += '<span class="d-err"  title="Différent du GT">'       + esc(op.new) + '</span>';
    }}
  }});

  document.getElementById('sbs-gt-body').innerHTML  = gtHtml  || '<em style="color:var(--text-muted)">—</em>';
  document.getElementById('sbs-ocr-body').innerHTML = ocrHtml || '<em style="color:var(--text-muted)">Aucune sortie</em>';

  // En-tête OCR : nom moteur + CER
  const c = cerColor(er.cer); const bg = cerBg(er.cer);
  document.getElementById('sbs-ocr-engine-name').textContent = er.engine;
  const cerBadgeEl = document.getElementById('sbs-ocr-cer');
  cerBadgeEl.textContent = pct(er.cer);
  cerBadgeEl.style.cssText = `color:${{c}};background:${{bg}};display:inline-block`;

  // Pipeline triple-diff (si applicable)
  const tripleEl = document.getElementById('sbs-triple-diff');
  if (er.ocr_intermediate) {{
    const ocrDiffHtml = renderDiff(er.ocr_diff);
    const llmDiffHtml = renderDiff(er.llm_correction_diff);
    const isPipeline = er.ocr_intermediate !== undefined;
    const modeLabel = {{text_only:'texte seul', text_and_image:'image+texte', zero_shot:'zero-shot'}}[er.pipeline_mode] || '';
    const pipeTag = `<span class="pipeline-tag">⛓ ${{modeLabel || 'pipeline'}}</span>`;
    let onBadge = '';
    if (er.over_normalization) {{
      const on = er.over_normalization;
      const onPct = (on.score * 100).toFixed(2);
      const cls = on.score > 0.05 ? 'over-norm-badge high' : 'over-norm-badge';
      onBadge = `<span class="${{cls}}" title="Classe 10 — sur-normalisation LLM">Sur-norm. ${{onPct}}%</span>`;
    }}
    let diplomaBadge = '';
    if (er.cer_diplomatic !== null && er.cer_diplomatic !== undefined) {{
      const dipC = cerColor(er.cer_diplomatic); const dipB = cerBg(er.cer_diplomatic);
      const delta = er.cer - er.cer_diplomatic;
      const deltaHint = delta > 0.001 ? ` (−${{(delta*100).toFixed(1)}}% avec normalisation)` : '';
      diplomaBadge = `<span class="cer-badge" style="color:${{dipC}};background:${{dipB}};opacity:.85"
        title="CER diplomatique${{deltaHint}}">diplo. ${{pct(er.cer_diplomatic)}}</span>`;
    }}
    tripleEl.style.display = '';
    tripleEl.innerHTML = `
      <div style="margin-top:.75rem;padding-top:.75rem;border-top:1px solid var(--border)">
        <div style="display:flex;align-items:center;gap:.4rem;margin-bottom:.5rem;font-size:.83rem;font-weight:600">
          ${{pipeTag}} ${{diplomaBadge}} ${{onBadge}}
          <span class="badge" style="background:#f1f5f9">WER ${{pct(er.wer)}}</span>
        </div>
        <div class="triple-diff-wrap">
          <div class="triple-diff-section">
            <h5>GT → OCR brut</h5>
            ${{ocrDiffHtml || '<em style="color:var(--text-muted)">—</em>'}}
          </div>
          <div class="triple-diff-section">
            <h5>OCR brut → Correction LLM</h5>
            ${{llmDiffHtml || '<em style="color:var(--text-muted)">—</em>'}}
          </div>
        </div>
      </div>`;
  }} else {{
    // Afficher WER / CER diplomatique même hors pipeline
    let diplomaBadge = '';
    if (er.cer_diplomatic !== null && er.cer_diplomatic !== undefined) {{
      const dipC = cerColor(er.cer_diplomatic); const dipB = cerBg(er.cer_diplomatic);
      const delta = er.cer - er.cer_diplomatic;
      const deltaHint = delta > 0.001 ? ` (−${{(delta*100).toFixed(1)}}% avec normalisation)` : '';
      diplomaBadge = `<span class="cer-badge" style="color:${{dipC}};background:${{dipB}};opacity:.85"
        title="CER diplomatique${{deltaHint}}">diplo. ${{pct(er.cer_diplomatic)}}</span>`;
    }}
    const errBadge = er.error ? `<span class="badge" style="background:#fee2e2;color:#dc2626">Erreur</span>` : '';
    if (diplomaBadge || errBadge) {{
      tripleEl.style.display = '';
      tripleEl.innerHTML = `<div style="margin-top:.5rem;display:flex;gap:.4rem;flex-wrap:wrap;font-size:.82rem">
        <span class="badge" style="background:#f1f5f9">WER ${{pct(er.wer)}}</span>
        ${{diplomaBadge}} ${{errBadge}}
      </div>`;
    }} else {{
      tripleEl.style.display = 'none';
      tripleEl.innerHTML = '';
    }}
  }}
}}

// ── Score badge (ligatures / diacritiques) ───────────────────────
function _scoreBadge(v, label) {{
  if (v === null || v === undefined) return '<span style="color:var(--text-muted)">—</span>';
  const pctVal = (v * 100).toFixed(1);
  const color = v >= 0.9 ? '#16a34a' : v >= 0.7 ? '#ca8a04' : '#dc2626';
  const bg = v >= 0.9 ? '#f0fdf4' : v >= 0.7 ? '#fefce8' : '#fef2f2';
  return `<span class="cer-badge" style="color:${{color}};background:${{bg}}" title="${{label}} : ${{pctVal}}%">${{pctVal}}%</span>`;
}}

// ── Vue Classement ──────────────────────────────────────────────
let rankingSort = {{ col: 'cer', dir: 'asc' }};

function renderRanking() {{
  const engines = [...DATA.engines];
  // Trier
  engines.sort((a, b) => {{
    let va = a[rankingSort.col], vb = b[rankingSort.col];
    if (typeof va === 'string') va = va.toLowerCase();
    if (typeof vb === 'string') vb = vb.toLowerCase();
    if (va === null) va = Infinity;
    if (vb === null) vb = Infinity;
    return rankingSort.dir === 'asc' ? (va > vb ? 1 : -1) : (va < vb ? 1 : -1);
  }});

  const tbody = document.getElementById('ranking-tbody');
  tbody.innerHTML = engines.map((e, i) => {{
    const rank = i + 1;
    const badgeClass = rank === 1 ? 'rank-badge rank-1' : 'rank-badge';
    const cerC = cerColor(e.cer); const cerB = cerBg(e.cer);
    const barW = Math.min(100, e.cer * 100 * 3);

    // Badge pipeline
    let pipelineBadge = '';
    let pipelineStepsHtml = '';
    if (e.is_pipeline && e.pipeline_info) {{
      const pi = e.pipeline_info;
      const modeLabel = {{text_only:'texte', text_and_image:'image+texte', zero_shot:'zero-shot'}}[pi.pipeline_mode] || pi.pipeline_mode || '';
      pipelineBadge = `<span class="pipeline-tag" title="Pipeline OCR+LLM — mode ${{modeLabel}}">
        ⛓ pipeline<span class="pipe-arrow">·${{modeLabel}}</span></span>`;
      if (pi.pipeline_steps) {{
        pipelineStepsHtml = `<div class="pipeline-steps">` +
          pi.pipeline_steps.map(s => s.type === 'ocr'
            ? `<span class="step-chip ocr">OCR: ${{esc(s.engine)}}</span>`
            : `<span class="step-chip llm">LLM: ${{esc(s.model)}}</span>`
          ).join(`<span class="step-arrow">→</span>`) +
          `</div>`;
      }}
    }}

    // Sur-normalisation (classe 10)
    let overNormCell = '<td style="color:var(--text-muted)">—</td>';
    if (e.is_pipeline && e.pipeline_info && e.pipeline_info.over_normalization) {{
      const on = e.pipeline_info.over_normalization;
      const onPct = (on.score * 100).toFixed(2);
      const cls = on.score > 0.05 ? 'over-norm-badge high' : 'over-norm-badge';
      overNormCell = `<td><span class="${{cls}}" title="Classe 10 — ${{on.over_normalized_count}} mots corrects dégradés sur ${{on.total_correct_ocr_words}}">${{onPct}} %</span></td>`;
    }}

    // CER diplomatique
    let diploCerCell = '<td style="color:var(--text-muted)">—</td>';
    if (e.cer_diplomatic !== null && e.cer_diplomatic !== undefined) {{
      const dipC = cerColor(e.cer_diplomatic); const dipB = cerBg(e.cer_diplomatic);
      const delta = e.cer - e.cer_diplomatic;
      const deltaStr = delta > 0.001 ? ` <span style="font-size:.65rem;color:#059669">-${{(delta*100).toFixed(1)}}%</span>` : '';
      const profileHint = e.cer_diplomatic_profile ? ` title="Profil : ${{esc(e.cer_diplomatic_profile)}}"` : '';
      diploCerCell = `<td${{profileHint}}>
        <span class="cer-badge" style="color:${{dipC}};background:${{dipB}}">${{pct(e.cer_diplomatic)}}</span>${{deltaStr}}
      </td>`;
    }}

    // ── Sprint 10 : Gini + Ancrage ─────────────────────────────────────
    let giniCell = '<td style="color:var(--text-muted)">—</td>';
    if (e.gini !== null && e.gini !== undefined) {{
      const gv = e.gini;
      const gColor = gv < 0.3 ? '#16a34a' : gv < 0.5 ? '#ca8a04' : '#dc2626';
      const gBg = gv < 0.3 ? '#f0fdf4' : gv < 0.5 ? '#fefce8' : '#fef2f2';
      giniCell = `<td><span class="cer-badge" style="color:${{gColor}};background:${{gBg}}"
        title="Gini=${{gv.toFixed(3)}} — 0=uniforme, 1=concentré">${{gv.toFixed(3)}}</span></td>`;
    }}
    let anchorCell = '<td style="color:var(--text-muted)">—</td>';
    if (e.anchor_score !== null && e.anchor_score !== undefined) {{
      const av = e.anchor_score;
      const hallBadge = (e.hallucinating_doc_rate && e.hallucinating_doc_rate > 0.2)
        ? ' <span title="Hallucinations détectées">⚠️</span>' : '';
      anchorCell = `<td>${{_scoreBadge(av, 'Ancrage trigrammes')}}${{hallBadge}}</td>`;
    }}

    return `<tr>
      <td><span class="${{badgeClass}}">${{rank}}</span></td>
      <td>
        <span class="engine-name">${{esc(e.name)}}</span>
        ${{pipelineBadge}}
        ${{e.is_vlm ? '<span class="pipeline-tag" style="background:#fce7f3;color:#9d174d">👁 VLM</span>' : ''}}
        <span class="engine-version">v${{esc(e.version)}}</span>
        ${{pipelineStepsHtml}}
      </td>
      <td>
        <span class="bar" style="width:${{barW}}px;background:${{cerC}}"></span>
        <span class="cer-badge" style="color:${{cerC}};background:${{cerB}}">${{pct(e.cer)}}</span>
      </td>
      ${{diploCerCell}}
      <td>${{pct(e.wer)}}</td>
      <td>${{pct(e.mer)}}</td>
      <td>${{pct(e.wil)}}</td>
      <td>${{_scoreBadge(e.ligature_score, 'Ligatures')}}</td>
      <td>${{_scoreBadge(e.diacritic_score, 'Diacritiques')}}</td>
      ${{giniCell}}
      ${{anchorCell}}
      <td style="color:var(--text-muted)">${{pct(e.cer_median)}}</td>
      <td style="color:var(--text-muted)">${{pct(e.cer_min)}}</td>
      <td style="color:var(--text-muted)">${{pct(e.cer_max)}}</td>
      ${{overNormCell}}
      <td><span class="pill">${{e.doc_count}}</span></td>
    </tr>`;
  }}).join('');

  // Stats globales
  const pipelineCount = DATA.engines.filter(e => e.is_pipeline).length;
  const stats = document.getElementById('ranking-stats');
  stats.innerHTML = `
    <div class="stat">Corpus <b>${{esc(DATA.meta.corpus_name)}}</b></div>
    <div class="stat">Documents <b>${{DATA.meta.document_count}}</b></div>
    <div class="stat">Concurrents <b>${{DATA.engines.length}}</b>
      ${{pipelineCount ? `<span class="pipeline-tag" style="margin-left:.3rem">${{pipelineCount}} pipeline${{pipelineCount>1?'s':''}}</span>` : ''}}
    </div>
  `;
}}

// Tri au clic sur en-tête
document.querySelectorAll('#ranking-table th.sortable').forEach(th => {{
  th.addEventListener('click', () => {{
    const col = th.dataset.col;
    if (rankingSort.col === col) {{
      rankingSort.dir = rankingSort.dir === 'asc' ? 'desc' : 'asc';
    }} else {{
      rankingSort.col = col;
      rankingSort.dir = 'asc';
    }}
    document.querySelectorAll('#ranking-table th').forEach(t => {{
      t.classList.remove('sorted');
      const icon = t.querySelector('.sort-icon');
      if (icon) icon.textContent = '↕';
    }});
    th.classList.add('sorted');
    const icon = th.querySelector('.sort-icon');
    if (icon) icon.textContent = rankingSort.dir === 'asc' ? '↑' : '↓';
    renderRanking();
  }});
}});

// ── Vue Galerie ─────────────────────────────────────────────────
function renderGallery() {{
  const sortKey  = document.getElementById('gallery-sort').value;
  const filterCer = parseFloat(document.getElementById('gallery-filter-cer').value) / 100 || 0;
  const filterEngine = document.getElementById('gallery-engine-select').value;

  let docs = [...DATA.documents];

  // Filtre CER
  if (filterCer > 0) {{
    docs = docs.filter(d => {{
      if (filterEngine) {{
        const er = d.engine_results.find(r => r.engine === filterEngine);
        return er && er.cer >= filterCer;
      }}
      return d.mean_cer >= filterCer;
    }});
  }}

  // Tri
  docs.sort((a, b) => {{
    if (sortKey === 'mean_cer') return a.mean_cer - b.mean_cer;
    if (sortKey === 'difficulty_score') return (b.difficulty_score||0) - (a.difficulty_score||0);
    if (sortKey === 'best_engine') return a.best_engine.localeCompare(b.best_engine);
    return a.doc_id.localeCompare(b.doc_id);
  }});

  const grid = document.getElementById('gallery-grid');
  const empty = document.getElementById('gallery-empty');

  if (!docs.length) {{
    grid.innerHTML = '';
    empty.style.display = '';
    return;
  }}
  empty.style.display = 'none';

  grid.innerHTML = docs.map(doc => {{
    const imgTag = doc.image_b64
      ? `<img src="${{doc.image_b64}}" alt="${{esc(doc.doc_id)}}" loading="lazy">`
      : `<div class="img-placeholder">🖹</div>`;

    const badges = doc.engine_results.map(er => {{
      const c = cerColor(er.cer); const bg = cerBg(er.cer);
      const isPipe = er.ocr_intermediate !== undefined;
      const label = isPipe ? '⛓' + er.engine.slice(0,8) : er.engine.slice(0,8);
      return `<span class="engine-cer-badge" style="color:${{c}};background:${{bg}}"
        title="${{esc(er.engine)}}${{isPipe?' (pipeline)':''}}">${{esc(label)}} ${{pct(er.cer,1)}}</span>`;
    }}).join('');

    // Difficulty badge
    let diffBadge = '';
    if (doc.difficulty_score !== undefined) {{
      const dScore = doc.difficulty_score;
      const dColor = dScore < 0.25 ? '#16a34a' : dScore < 0.5 ? '#ca8a04' : dScore < 0.75 ? '#ea580c' : '#dc2626';
      const dBg    = dScore < 0.25 ? '#f0fdf4' : dScore < 0.5 ? '#fefce8' : dScore < 0.75 ? '#fff7ed' : '#fef2f2';
      diffBadge = `<span class="diff-badge" style="color:${{dColor}};background:${{dBg}};margin-left:.3rem"
        title="Difficulté intrinsèque : ${{doc.difficulty_label}}">⚡ ${{doc.difficulty_label}}</span>`;
    }}

    return `<div class="gallery-card" onclick="openDocument('${{esc(doc.doc_id)}}')">
      ${{imgTag}}
      <div class="gallery-card-body">
        <div class="gallery-card-title">${{esc(doc.doc_id)}}${{diffBadge}}</div>
        <div class="gallery-card-badges">${{badges}}</div>
      </div>
    </div>`;
  }}).join('');
}}

// ── Vue Document ────────────────────────────────────────────────
let currentDocId = null;
let zoomLevel = 1;
let dragStart = null;
let imgOffset = {{ x: 0, y: 0 }};

function openDocument(docId) {{
  _switchView('document');
  updateURL('document', {{ doc: docId }});
  loadDocument(docId);
}}

function loadDocument(docId) {{
  const doc = DATA.documents.find(d => d.doc_id === docId);
  if (!doc) return;
  currentDocId = docId;

  // Sidebar : highlight
  document.querySelectorAll('.doc-list-item').forEach(el => {{
    el.classList.toggle('active', el.dataset.docId === docId);
  }});

  // Titre
  document.getElementById('doc-detail-title').textContent = doc.doc_id;

  // Métriques
  const metricsDiv = document.getElementById('doc-detail-metrics');
  const cer = doc.mean_cer;
  const dScore = doc.difficulty_score;
  const dColor = dScore < 0.25 ? '#16a34a' : dScore < 0.5 ? '#ca8a04' : dScore < 0.75 ? '#ea580c' : '#dc2626';
  const dLabel = doc.difficulty_label || '';
  metricsDiv.innerHTML = `<div class="stat">CER moyen <b style="color:${{cerColor(cer)}}">${{pct(cer)}}</b></div>
    <div class="stat">Meilleur moteur <b>${{esc(doc.best_engine)}}</b></div>
    ${{dScore !== undefined ? `<div class="stat">Difficulté <b style="color:${{dColor}}">${{dLabel}} (${{(dScore*100).toFixed(0)}}%)</b></div>` : ''}}`;

  // Image
  resetZoom();
  const img = document.getElementById('doc-image');
  const placeholder = document.getElementById('doc-image-placeholder');
  if (doc.image_b64) {{
    img.src = doc.image_b64;
    img.style.display = '';
    placeholder.style.display = 'none';
  }} else {{
    img.style.display = 'none';
    placeholder.style.display = '';
    placeholder.innerHTML = `<span style="font-size:2rem">🖹</span><span>${{esc(doc.image_path)}}</span>`;
  }}

  // Side-by-side diff — sélecteur de concurrent
  const selWrap = document.getElementById('sbs-engine-select');
  const sel = document.getElementById('sbs-engine-dropdown');
  if (doc.engine_results.length > 1) {{
    sel.innerHTML = doc.engine_results.map((er, i) =>
      `<option value="${{i}}">${{esc(er.engine)}}</option>`
    ).join('');
    selWrap.style.display = '';
  }} else {{
    sel.innerHTML = '';
    selWrap.style.display = 'none';
  }}
  renderSideBySide(docId);

  // ── Sprint 10 : distribution CER par ligne ──────────────────────────
  const lineCard = document.getElementById('doc-line-metrics-card');
  const lineContent = document.getElementById('doc-line-metrics-content');
  // Prendre le premier moteur ayant des line_metrics
  const erWithLine = doc.engine_results.find(er => er.line_metrics);
  if (erWithLine && erWithLine.line_metrics) {{
    lineCard.style.display = '';
    lineContent.innerHTML = renderLineMetrics(doc.engine_results);
  }} else {{
    lineCard.style.display = 'none';
  }}

  // ── Sprint 10 : hallucinations ──────────────────────────────────────
  const hallCard = document.getElementById('doc-hallucination-card');
  const hallContent = document.getElementById('doc-hallucination-content');
  const erWithHall = doc.engine_results.find(er => er.hallucination_metrics && er.hallucination_metrics.is_hallucinating);
  if (erWithHall || doc.engine_results.some(er => er.hallucination_metrics)) {{
    hallCard.style.display = '';
    hallContent.innerHTML = renderHallucinationPanel(doc.engine_results);
  }} else {{
    hallCard.style.display = 'none';
  }}
}}

// ── Sprint 10 : rendu distribution CER par ligne ────────────────
function renderLineMetrics(engineResults) {{
  const heatmapColors = (v) => {{
    if (v < 0.05) return '#86efac';
    if (v < 0.15) return '#fde68a';
    if (v < 0.30) return '#fb923c';
    return '#f87171';
  }};

  return engineResults.filter(er => er.line_metrics).map(er => {{
    const lm = er.line_metrics;
    const c = cerColor(er.cer); const bg = cerBg(er.cer);

    // Heatmap de position
    const heatmap = lm.heatmap || [];
    const maxHeat = Math.max(...heatmap, 0.01);
    const heatmapHtml = heatmap.length > 0
      ? `<div class="heatmap-wrap">` +
        heatmap.map((v, i) => {{
          const h = Math.max(4, Math.round(60 * v / maxHeat));
          return `<div class="heatmap-bar" style="height:${{h}}px;background:${{heatmapColors(v)}}"
            title="Tranche ${{i+1}}/${{heatmap.length}} — CER=${{(v*100).toFixed(1)}}%"></div>`;
        }}).join('') +
        `</div><div class="heatmap-labels"><span>${{I18N.heatmap_start||'Début'}}</span><span>${{I18N.heatmap_mid||'Milieu'}}</span><span>${{I18N.heatmap_end||'Fin'}}</span></div>`
      : '<em style="color:var(--text-muted)">—</em>';

    // Percentiles
    const p = lm.percentiles || {{}};
    const pctBars = ['p50','p75','p90','p95','p99'].map(k => {{
      const v = p[k] || 0;
      const w = Math.min(100, v * 100 * 2);
      const fillColor = v < 0.15 ? '#86efac' : v < 0.30 ? '#fde68a' : '#f87171';
      return `<div class="pct-bar-row">
        <span class="pct-bar-label">${{k}}</span>
        <div class="pct-bar-track"><div class="pct-bar-fill" style="width:${{w}}%;background:${{fillColor}}"></div></div>
        <span class="pct-bar-val">${{(v*100).toFixed(1)}}%</span>
      </div>`;
    }}).join('');

    // Taux catastrophiques
    const cr = lm.catastrophic_rate || {{}};
    const crRows = Object.entries(cr).map(([t, rate]) => {{
      const tPct = (parseFloat(t)*100).toFixed(0);
      const ratePct = (rate*100).toFixed(1);
      const color = rate < 0.05 ? '#16a34a' : rate < 0.15 ? '#ca8a04' : '#dc2626';
      return `<span class="stat"><b style="color:${{color}}">${{ratePct}}%</b> lignes CER&gt;${{tPct}}%</span>`;
    }}).join('');

    // Gini
    const gini = lm.gini !== undefined ? lm.gini.toFixed(3) : '—';
    const giniColor = lm.gini < 0.3 ? '#16a34a' : lm.gini < 0.5 ? '#ca8a04' : '#dc2626';

    return `<div style="margin-bottom:1.25rem;padding-bottom:1rem;border-bottom:1px solid var(--border)">
      <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.6rem">
        <strong>${{esc(er.engine)}}</strong>
        <span class="cer-badge" style="color:${{c}};background:${{bg}}">${{pct(er.cer)}}</span>
        <span class="stat">Gini <b style="color:${{giniColor}}">${{gini}}</b></span>
        <span class="stat">${{lm.line_count}} ${{I18N.lines||'lignes'}}</span>
        ${{crRows}}
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem">
        <div>
          <div style="font-size:.75rem;font-weight:600;color:var(--text-muted);margin-bottom:.3rem">${{I18N.heatmap_title||'CARTE THERMIQUE (position)'}}</div>
          ${{heatmapHtml}}
        </div>
        <div>
          <div style="font-size:.75rem;font-weight:600;color:var(--text-muted);margin-bottom:.3rem">${{I18N.percentile_title||'PERCENTILES CER'}}</div>
          <div class="pct-bars">${{pctBars}}</div>
        </div>
      </div>
    </div>`;
  }}).join('') || `<em style="color:var(--text-muted)">${{I18N.no_line_metrics||'Aucune métrique de ligne disponible.'}}</em>`;
}}

// ── Sprint 10 : rendu panneau hallucinations ─────────────────────
function renderHallucinationPanel(engineResults) {{
  const withHall = engineResults.filter(er => er.hallucination_metrics);
  if (!withHall.length) return `<em style="color:var(--text-muted)">${{I18N.no_hall_metrics||"Aucune métrique d'hallucination disponible."}}</em>`;

  return withHall.map(er => {{
    const hm = er.hallucination_metrics;
    const isHall = hm.is_hallucinating;
    const badgeClass = isHall ? 'hallucination-badge' : 'hallucination-badge ok';
    const badgeLabel = isHall ? (I18N.hall_detected||'⚠️ Hallucinations détectées') : (I18N.hall_ok||'✓ Ancrage satisfaisant');

    const blocksHtml = hm.hallucinated_blocks && hm.hallucinated_blocks.length > 0
      ? hm.hallucinated_blocks.slice(0, 5).map(b =>
          `<div class="halluc-block">
            <div class="halluc-block-meta">${{I18N.hall_block_label||'Bloc halluciné'}} — ${{b.length}} mots (tokens ${{b.start_token}}–${{b.end_token}})</div>
            ${{esc(b.text)}}
          </div>`
        ).join('') +
        (hm.hallucinated_blocks.length > 5 ? `<div style="font-size:.72rem;color:var(--text-muted);margin-top:.25rem">… ${{hm.hallucinated_blocks.length - 5}} ${{I18N.hall_more_blocks||'bloc(s) supplémentaire(s)'}}</div>` : '')
      : `<em style="color:var(--text-muted);font-size:.8rem">${{I18N.no_hall_blocks||'Aucun bloc halluciné détecté.'}}</em>`;

    return `<div style="margin-bottom:1.25rem;padding-bottom:1rem;border-bottom:1px solid var(--border)">
      <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.6rem;flex-wrap:wrap">
        <strong>${{esc(er.engine)}}</strong>
        <span class="${{badgeClass}}">${{badgeLabel}}</span>
        <span class="stat">Ancrage <b>${{(hm.anchor_score*100).toFixed(1)}}%</b></span>
        <span class="stat">Ratio longueur <b>${{hm.length_ratio.toFixed(2)}}</b></span>
        <span class="stat">Insertion nette <b>${{(hm.net_insertion_rate*100).toFixed(1)}}%</b></span>
        <span class="stat">${{hm.gt_word_count}} mots GT / ${{hm.hyp_word_count}} mots sortie</span>
      </div>
      ${{isHall ? `<div style="margin-bottom:.5rem;font-size:.82rem;font-weight:600;color:#9d174d">${{I18N.hall_blocks_title||'Blocs sans ancrage dans le GT :'}}</div>` : ''}}
      ${{isHall ? blocksHtml : ''}}
    </div>`;
  }}).join('');
}}

// ── Sprint 10 — Scatter Gini vs CER moyen ──────────────────────
function buildGiniCerScatter() {{
  const canvas = document.getElementById('chart-gini-cer');
  if (!canvas) return;
  const pts = DATA.gini_vs_cer || [];
  if (!pts.length) {{
    canvas.parentElement.innerHTML = `<p style="color:var(--text-muted);padding:1rem">${{I18N.no_gini||'Données Gini non disponibles.'}}</p>`;
    return;
  }}
  const datasets = pts.map((p, i) => ({{
    label: p.engine,
    data: [{{ x: p.cer * 100, y: p.gini }}],
    backgroundColor: engineColor(DATA.engines.findIndex(e => e.name === p.engine)) + 'cc',
    borderColor: engineColor(DATA.engines.findIndex(e => e.name === p.engine)),
    borderWidth: p.is_pipeline ? 2 : 1,
    pointRadius: p.is_pipeline ? 9 : 7,
    pointStyle: p.is_pipeline ? 'triangle' : 'circle',
  }}));

  chartInstances['gini-cer'] = new Chart(canvas.getContext('2d'), {{
    type: 'scatter',
    data: {{ datasets }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ position: 'top', labels: {{ font: {{ size: 11 }} }} }},
        tooltip: {{ callbacks: {{
          label: ctx => `${{ctx.dataset.label}}: CER=${{ctx.parsed.x.toFixed(2)}}%, Gini=${{ctx.parsed.y.toFixed(3)}}`,
        }} }},
      }},
      scales: {{
        x: {{ min: 0, title: {{ display: true, text: 'CER moyen (%)', font: {{ size: 11 }} }} }},
        y: {{ min: 0, max: 1, title: {{ display: true, text: 'Coefficient de Gini', font: {{ size: 11 }} }} }},
      }},
    }},
  }});
}}

// ── Sprint 10 — Scatter ratio longueur vs score d'ancrage ────────
function buildRatioAnchorScatter() {{
  const canvas = document.getElementById('chart-ratio-anchor');
  if (!canvas) return;
  const pts = DATA.ratio_vs_anchor || [];
  if (!pts.length) {{
    canvas.parentElement.innerHTML = `<p style="color:var(--text-muted);padding:1rem">Données d'ancrage non disponibles.</p>`;
    return;
  }}

  // Zone de danger (ancrage < 0.5 OU ratio > 1.2) dessinée via plugin
  const datasets = pts.map((p, i) => ({{
    label: p.engine + (p.is_vlm ? ' 👁' : ''),
    data: [{{ x: p.anchor_score, y: p.length_ratio }}],
    backgroundColor: engineColor(DATA.engines.findIndex(e => e.name === p.engine)) + 'cc',
    borderColor: engineColor(DATA.engines.findIndex(e => e.name === p.engine)),
    borderWidth: p.is_vlm ? 3 : 1,
    pointRadius: p.is_vlm ? 10 : 7,
    pointStyle: p.is_vlm ? 'star' : 'circle',
  }}));

  chartInstances['ratio-anchor'] = new Chart(canvas.getContext('2d'), {{
    type: 'scatter',
    data: {{ datasets }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ position: 'top', labels: {{ font: {{ size: 11 }} }} }},
        tooltip: {{ callbacks: {{
          label: ctx => `${{ctx.dataset.label}}: ancrage=${{(ctx.parsed.x*100).toFixed(1)}}%, ratio=${{ctx.parsed.y.toFixed(2)}}`,
        }} }},
      }},
      scales: {{
        x: {{ min: 0, max: 1, title: {{ display: true, text: "Score d'ancrage [0–1]", font: {{ size: 11 }} }} }},
        y: {{ min: 0, title: {{ display: true, text: 'Ratio longueur (sortie/GT)', font: {{ size: 11 }} }} }},
      }},
    }},
    plugins: [{{
      id: 'danger-zones',
      beforeDraw(chart) {{
        const {{ ctx: c, chartArea: {{ left, top, right, bottom }}, scales: {{ x, y }} }} = chart;
        c.save();
        // Ancrage < 0.5 (gauche)
        const xHalf = x.getPixelForValue(0.5);
        c.fillStyle = 'rgba(239,68,68,0.07)';
        c.fillRect(left, top, xHalf - left, bottom - top);
        // Ratio > 1.2 (haut)
        const y12 = y.getPixelForValue(1.2);
        if (y12 > top) {{
          c.fillRect(left, top, right - left, y12 - top);
        }}
        // Lignes de seuil
        c.strokeStyle = 'rgba(239,68,68,0.35)'; c.lineWidth = 1; c.setLineDash([4,4]);
        c.beginPath(); c.moveTo(xHalf, top); c.lineTo(xHalf, bottom); c.stroke();
        if (y12 > top) {{
          c.beginPath(); c.moveTo(left, y12); c.lineTo(right, y12); c.stroke();
        }}
        c.restore();
      }},
    }}],
  }});
}}

function buildDocList() {{
  const list = document.getElementById('doc-list');
  list.innerHTML = DATA.documents.map(doc => {{
    const c = cerColor(doc.mean_cer); const bg = cerBg(doc.mean_cer);
    return `<div class="doc-list-item" data-doc-id="${{esc(doc.doc_id)}}"
        onclick="loadDocument('${{esc(doc.doc_id)}}')">
      <span class="doc-list-label">${{esc(doc.doc_id)}}</span>
      <span class="doc-list-cer" style="color:${{c}};background:${{bg}}">${{pct(doc.mean_cer,1)}}</span>
    </div>`;
  }}).join('');
  if (DATA.documents.length) loadDocument(DATA.documents[0].doc_id);
}}

// Zoom
function handleZoom(e) {{
  e.preventDefault();
  zoom(e.deltaY < 0 ? 1.15 : 0.87);
}}
function zoom(factor) {{
  zoomLevel = Math.max(0.5, Math.min(5, zoomLevel * factor));
  applyZoom();
}}
function resetZoom() {{
  zoomLevel = 1; imgOffset = {{ x: 0, y: 0 }};
  applyZoom();
}}
function applyZoom() {{
  const img = document.getElementById('doc-image');
  img.style.transform = `scale(${{zoomLevel}}) translate(${{imgOffset.x}}px, ${{imgOffset.y}}px)`;
}}
function startDrag(e) {{
  if (zoomLevel <= 1) return;
  dragStart = {{ x: e.clientX - imgOffset.x * zoomLevel, y: e.clientY - imgOffset.y * zoomLevel }};
  document.getElementById('doc-image-wrap').style.cursor = 'grabbing';
}}
function doDrag(e) {{
  if (!dragStart) return;
  imgOffset.x = (e.clientX - dragStart.x) / zoomLevel;
  imgOffset.y = (e.clientY - dragStart.y) / zoomLevel;
  applyZoom();
}}
function endDrag() {{
  dragStart = null;
  document.getElementById('doc-image-wrap').style.cursor = zoomLevel > 1 ? 'grab' : 'zoom-in';
}}

// ── Graphiques ──────────────────────────────────────────────────
let chartsBuilt = false;
let chartInstances = {{}};

function destroyChart(id) {{
  if (chartInstances[id]) {{ chartInstances[id].destroy(); delete chartInstances[id]; }}
}}

function buildCharts() {{
  if (chartsBuilt) return;
  chartsBuilt = true;
  buildCerHistogram();
  buildRadar();
  buildCerPerDoc();
  buildDurationChart();
  buildQualityCerScatter();
  buildTaxonomyChart();
  // Sprint 7
  buildReliabilityCurves();
  buildBootstrapCIChart();
  buildVennDiagram();
  buildWilcoxonTable();
  buildErrorClusters();
  initCorrelationMatrix();
  // Sprint 10
  buildGiniCerScatter();
  buildRatioAnchorScatter();
}}

function buildCerHistogram() {{
  destroyChart('cer-hist');
  const ctx = document.getElementById('chart-cer-hist').getContext('2d');
  // Construire histogramme à bins fixes [0-5, 5-10, 10-20, 20-30, 30-50, 50+]
  const bins    = [0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.01];
  const labels  = ['0–5%', '5–10%', '10–20%', '20–30%', '30–50%', '>50%'];
  const colors  = ['#16a34a','#65a30d','#ca8a04','#ea580c','#dc2626','#9f1239'];

  const datasets = DATA.engines.map((e, ei) => {{
    const counts = new Array(labels.length).fill(0);
    e.cer_values.forEach(v => {{
      for (let i = 0; i < bins.length - 1; i++) {{
        if (v >= bins[i] && v < bins[i+1]) {{ counts[i]++; break; }}
      }}
    }});
    return {{
      label: e.name, data: counts,
      backgroundColor: engineColor(ei) + 'aa',
      borderColor: engineColor(ei),
      borderWidth: 1,
    }};
  }});

  chartInstances['cer-hist'] = new Chart(ctx, {{
    type: 'bar',
    data: {{ labels, datasets }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ position: 'top', labels: {{ font: {{ size: 11 }} }} }} }},
      scales: {{
        x: {{ title: {{ display: true, text: 'Plage CER', font: {{ size: 11 }} }} }},
        y: {{ title: {{ display: true, text: 'Nombre de documents', font: {{ size: 11 }} }},
               ticks: {{ stepSize: 1 }} }},
      }},
    }},
  }});
}}

function buildRadar() {{
  destroyChart('radar');
  const ctx = document.getElementById('chart-radar').getContext('2d');
  // Axes : CER, WER, MER, WIL inversés (1 - valeur → plus c'est élevé, mieux c'est)
  const metrics = ['CER', 'WER', 'MER', 'WIL'];
  const keys    = ['cer', 'wer', 'mer', 'wil'];
  const datasets = DATA.engines.map((e, i) => {{
    const data = keys.map(k => Math.max(0, (1 - (e[k] || 0)) * 100));
    return {{
      label: e.name, data,
      backgroundColor: engineColor(i) + '33',
      borderColor: engineColor(i),
      borderWidth: 2,
      pointRadius: 4,
      pointHoverRadius: 6,
    }};
  }});

  chartInstances['radar'] = new Chart(ctx, {{
    type: 'radar',
    data: {{ labels: metrics, datasets }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ position: 'top', labels: {{ font: {{ size: 11 }} }} }} }},
      scales: {{
        r: {{
          min: 0, max: 100,
          ticks: {{ stepSize: 20, font: {{ size: 10 }} }},
          pointLabels: {{ font: {{ size: 12, weight: 'bold' }} }},
        }},
      }},
    }},
  }});
}}

function buildCerPerDoc() {{
  destroyChart('cer-doc');
  const ctx = document.getElementById('chart-cer-doc').getContext('2d');
  const labels = DATA.documents.map(d => d.doc_id);
  const datasets = DATA.engines.map((e, ei) => {{
    const data = DATA.documents.map(doc => {{
      const er = doc.engine_results.find(r => r.engine === e.name);
      return er ? er.cer * 100 : null;
    }});
    return {{
      label: e.name, data,
      borderColor: engineColor(ei),
      backgroundColor: engineColor(ei) + '22',
      tension: 0.3, fill: false,
      pointRadius: 3, pointHoverRadius: 5,
    }};
  }});

  chartInstances['cer-doc'] = new Chart(ctx, {{
    type: 'line',
    data: {{ labels, datasets }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ position: 'top', labels: {{ font: {{ size: 11 }} }} }} }},
      scales: {{
        x: {{ ticks: {{ maxRotation: 45, font: {{ size: 10 }} }} }},
        y: {{ title: {{ display: true, text: 'CER (%)', font: {{ size: 11 }} }}, min: 0 }},
      }},
    }},
  }});
}}

function buildDurationChart() {{
  destroyChart('duration');
  const ctx = document.getElementById('chart-duration').getContext('2d');

  const labels = DATA.engines.map(e => e.name);
  const data   = DATA.engines.map(e => {{
    const docs = DATA.documents;
    const durs = docs.flatMap(d => d.engine_results
      .filter(r => r.engine === e.name)
      .map(r => r.duration));
    const mean = durs.length ? durs.reduce((a,b) => a+b, 0) / durs.length : 0;
    return parseFloat(mean.toFixed(3));
  }});

  chartInstances['duration'] = new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{
        label: 'Durée moy. (s)',
        data,
        backgroundColor: DATA.engines.map((_, i) => engineColor(i) + 'aa'),
        borderColor:     DATA.engines.map((_, i) => engineColor(i)),
        borderWidth: 1,
      }}],
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        y: {{ title: {{ display: true, text: 'Secondes', font: {{ size: 11 }} }}, min: 0 }},
      }},
    }},
  }});
}}

function buildQualityCerScatter() {{
  const ctx = document.getElementById('chart-quality-cer');
  if (!ctx) return;
  // Construire les points : un par document, un dataset par moteur
  const datasets = DATA.engines.map((e, ei) => {{
    const points = DATA.documents.flatMap(doc => {{
      const er = doc.engine_results.find(r => r.engine === e.name);
      if (!er || er.error || !er.image_quality) return [];
      return [{{ x: er.image_quality.quality_score, y: er.cer * 100 }}];
    }});
    return {{
      label: e.name, data: points,
      backgroundColor: engineColor(ei) + 'bb',
      borderColor: engineColor(ei),
      borderWidth: 1, pointRadius: 5, pointHoverRadius: 7,
    }};
  }}).filter(d => d.data.length > 0);

  if (!datasets.length) {{ ctx.parentElement.innerHTML = '<p style="color:var(--text-muted);padding:1rem">Aucune donnée de qualité image disponible.</p>'; return; }}

  chartInstances['quality-cer'] = new Chart(ctx.getContext('2d'), {{
    type: 'scatter',
    data: {{ datasets }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ position: 'top', labels: {{ font: {{ size: 11 }} }} }},
        tooltip: {{ callbacks: {{
          label: ctx => `${{ctx.dataset.label}}: qualité=${{ctx.parsed.x.toFixed(2)}}, CER=${{ctx.parsed.y.toFixed(1)}}%`,
        }} }},
      }},
      scales: {{
        x: {{ min: 0, max: 1, title: {{ display: true, text: 'Score qualité image [0–1]', font: {{ size: 11 }} }} }},
        y: {{ min: 0, title: {{ display: true, text: 'CER (%)', font: {{ size: 11 }} }} }},
      }},
    }},
  }});
}}

function buildTaxonomyChart() {{
  const ctx = document.getElementById('chart-taxonomy');
  if (!ctx) return;
  const taxLabels = ['Confusion visuelle','Diacritique','Casse','Ligature','Abréviation','Hapax','Segmentation','Hors-vocab.','Lacune'];
  const taxKeys = ['visual_confusion','diacritic_error','case_error','ligature_error','abbreviation_error','hapax','segmentation_error','oov_character','lacuna'];
  const taxColors = ['#6366f1','#f59e0b','#ec4899','#14b8a6','#8b5cf6','#64748b','#f97316','#06b6d4','#ef4444'];

  const datasets = DATA.engines.map((e, ei) => {{
    const tax = e.aggregated_taxonomy;
    const data = taxKeys.map(k => tax && tax.counts ? (tax.counts[k] || 0) : 0);
    return {{
      label: e.name, data,
      backgroundColor: engineColor(ei) + '99',
      borderColor: engineColor(ei),
      borderWidth: 1,
    }};
  }});

  chartInstances['taxonomy'] = new Chart(ctx.getContext('2d'), {{
    type: 'bar',
    data: {{ labels: taxLabels, datasets }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ position: 'top', labels: {{ font: {{ size: 11 }} }} }} }},
      scales: {{
        x: {{ ticks: {{ font: {{ size: 10 }} }} }},
        y: {{ title: {{ display: true, text: "Nb d'erreurs", font: {{ size: 11 }} }}, min: 0, ticks: {{ stepSize: 1 }} }},
      }},
    }},
  }});
}}

// ── Sprint 7 — Courbes de fiabilité ─────────────────────────────
function buildReliabilityCurves() {{
  const ctx = document.getElementById('chart-reliability');
  if (!ctx) return;
  const curves = DATA.reliability_curves || [];
  if (!curves.length) {{ ctx.parentElement.innerHTML = '<p style="color:var(--text-muted);padding:1rem">Données insuffisantes.</p>'; return; }}
  const datasets = curves.map((c, i) => {{
    const points = (c.points || []).map(p => ({{ x: p.pct_docs, y: p.mean_cer * 100 }}));
    return {{
      label: c.engine, data: points,
      borderColor: engineColor(i), backgroundColor: engineColor(i) + '22',
      tension: 0.3, fill: false, pointRadius: 2, pointHoverRadius: 5,
    }};
  }});
  destroyChart('reliability');
  chartInstances['reliability'] = new Chart(ctx.getContext('2d'), {{
    type: 'line',
    data: {{ datasets }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      parsing: {{ xAxisKey: 'x', yAxisKey: 'y' }},
      plugins: {{
        legend: {{ position: 'top', labels: {{ font: {{ size: 11 }} }} }},
        tooltip: {{ callbacks: {{
          title: ([item]) => `${{item.parsed.x.toFixed(0)}}% docs les plus faciles`,
          label: item => `${{item.dataset.label}}: CER moy = ${{item.parsed.y.toFixed(2)}}%`,
        }} }},
      }},
      scales: {{
        x: {{ type:'linear', min:0, max:100,
          title: {{ display:true, text:'% documents (triés par CER croissant)', font:{{ size:11 }} }} }},
        y: {{ min:0, title: {{ display:true, text:'CER moyen (%)', font:{{ size:11 }} }} }},
      }},
    }},
  }});
}}

// ── Sprint 7 — Bootstrap CI ──────────────────────────────────────
function buildBootstrapCIChart() {{
  const ctx = document.getElementById('chart-bootstrap-ci');
  if (!ctx) return;
  const cis = DATA.statistics && DATA.statistics.bootstrap_cis || [];
  if (!cis.length) {{ ctx.parentElement.innerHTML = '<p style="color:var(--text-muted);padding:1rem">Données insuffisantes.</p>'; return; }}

  const labels = cis.map(c => c.engine);
  const means  = cis.map(c => (c.mean * 100));
  const lowers = cis.map(c => (c.mean - c.ci_lower) * 100);
  const uppers = cis.map(c => (c.ci_upper - c.mean) * 100);

  destroyChart('bootstrap-ci');
  chartInstances['bootstrap-ci'] = new Chart(ctx.getContext('2d'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{
        label: 'CER moyen (%)',
        data: means,
        backgroundColor: cis.map((_, i) => engineColor(i) + 'aa'),
        borderColor:     cis.map((_, i) => engineColor(i)),
        borderWidth: 1,
        errorBars: {{ symmetric: false }},
      }}],
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            afterLabel: (ctx) => {{
              const ci = cis[ctx.dataIndex];
              return `IC 95% : [${{(ci.ci_lower*100).toFixed(2)}}%, ${{(ci.ci_upper*100).toFixed(2)}}%]`;
            }},
          }},
        }},
      }},
      scales: {{ y: {{ min: 0, title: {{ display:true, text:'CER (%)', font:{{size:11}} }} }} }},
    }},
    plugins: [{{
      id: 'errorBars',
      afterDatasetsDraw(chart) {{
        const {{ ctx: c, data, scales: {{ x, y }} }} = chart;
        chart.data.datasets[0].data.forEach((val, i) => {{
          const ci = cis[i];
          if (!ci) return;
          const xPos = x.getPixelForValue(i);
          const yTop = y.getPixelForValue(ci.ci_upper * 100);
          const yBot = y.getPixelForValue(ci.ci_lower * 100);
          c.save();
          c.strokeStyle = '#374151'; c.lineWidth = 2;
          c.beginPath(); c.moveTo(xPos, yTop); c.lineTo(xPos, yBot); c.stroke();
          c.beginPath(); c.moveTo(xPos-6, yTop); c.lineTo(xPos+6, yTop); c.stroke();
          c.beginPath(); c.moveTo(xPos-6, yBot); c.lineTo(xPos+6, yBot); c.stroke();
          c.restore();
        }});
      }},
    }}],
  }});
}}

// ── Sprint 7 — Diagramme de Venn ────────────────────────────────
function buildVennDiagram() {{
  const container = document.getElementById('venn-container');
  if (!container) return;
  const venn = DATA.venn_data;
  if (!venn || !venn.type) {{
    container.innerHTML = '<p style="color:var(--text-muted)">Données insuffisantes pour le diagramme de Venn.</p>';
    return;
  }}

  if (venn.type === 'venn2') {{
    const total = (venn.only_a || 0) + (venn.both || 0) + (venn.only_b || 0);
    const maxR = 80;
    const rA = Math.sqrt((venn.only_a + venn.both) / (total || 1)) * maxR + 30;
    const rB = Math.sqrt((venn.only_b + venn.both) / (total || 1)) * maxR + 30;
    const overlap = venn.both > 0 ? Math.min(rA, rB) * 0.6 : 0;
    const cxA = 140, cxB = cxA + rA + rB - overlap, cy = 130;
    const w = cxB + rB + 20, h = 260;
    container.innerHTML = `
      <div style="text-align:center">
        <svg width="${{w}}" height="${{h}}" viewBox="0 0 ${{w}} ${{h}}" style="max-width:100%">
          <circle cx="${{cxA}}" cy="${{cy}}" r="${{rA}}" fill="#2563eb" fill-opacity="0.25" stroke="#2563eb" stroke-width="2"/>
          <circle cx="${{cxB}}" cy="${{cy}}" r="${{rB}}" fill="#dc2626" fill-opacity="0.25" stroke="#dc2626" stroke-width="2"/>
          <text x="${{cxA - rA*0.5}}" y="${{cy}}" text-anchor="middle" font-size="13" font-weight="bold" fill="#1e40af">${{venn.only_a}}</text>
          <text x="${{(cxA + cxB)/2}}" y="${{cy}}" text-anchor="middle" font-size="13" font-weight="bold" fill="#374151">${{venn.both}}</text>
          <text x="${{cxB + rB*0.5}}" y="${{cy}}" text-anchor="middle" font-size="13" font-weight="bold" fill="#b91c1c">${{venn.only_b}}</text>
          <text x="${{cxA - rA*0.5}}" y="${{cy + rA + 14}}" text-anchor="middle" font-size="11" fill="#2563eb">${{esc(venn.label_a)}}</text>
          <text x="${{cxB + rB*0.5}}" y="${{cy + rB + 14}}" text-anchor="middle" font-size="11" fill="#dc2626">${{esc(venn.label_b)}}</text>
          <text x="${{(cxA+cxB)/2}}" y="${{cy + Math.min(rA,rB) + 14}}" text-anchor="middle" font-size="10" fill="#64748b">commun</text>
        </svg>
        <p style="font-size:.75rem;color:var(--text-muted);margin-top:.25rem">
          Erreurs exclusives ${{esc(venn.label_a)}} : ${{venn.only_a}} ·
          Communes : ${{venn.both}} ·
          Exclusives ${{esc(venn.label_b)}} : ${{venn.only_b}}
        </p>
      </div>
    `;
  }} else if (venn.type === 'venn3') {{
    // Venn 3 cercles simplifié
    const total = (venn.only_a||0)+(venn.only_b||0)+(venn.only_c||0)+(venn.ab||0)+(venn.ac||0)+(venn.bc||0)+(venn.abc||0) || 1;
    container.innerHTML = `
      <div style="text-align:center">
        <svg width="300" height="280" viewBox="0 0 300 280" style="max-width:100%">
          <circle cx="130" cy="110" r="80" fill="#2563eb" fill-opacity="0.2" stroke="#2563eb" stroke-width="1.5"/>
          <circle cx="170" cy="110" r="80" fill="#dc2626" fill-opacity="0.2" stroke="#dc2626" stroke-width="1.5"/>
          <circle cx="150" cy="155" r="80" fill="#16a34a" fill-opacity="0.2" stroke="#16a34a" stroke-width="1.5"/>
          <text x="95" y="95" text-anchor="middle" font-size="12" font-weight="bold" fill="#1e40af">${{venn.only_a}}</text>
          <text x="205" y="95" text-anchor="middle" font-size="12" font-weight="bold" fill="#b91c1c">${{venn.only_b}}</text>
          <text x="150" y="230" text-anchor="middle" font-size="12" font-weight="bold" fill="#15803d">${{venn.only_c}}</text>
          <text x="148" y="108" text-anchor="middle" font-size="11" fill="#374151">${{venn.ab}}</text>
          <text x="120" y="160" text-anchor="middle" font-size="11" fill="#374151">${{venn.ac}}</text>
          <text x="180" y="160" text-anchor="middle" font-size="11" fill="#374151">${{venn.bc}}</text>
          <text x="150" y="145" text-anchor="middle" font-size="11" font-weight="bold" fill="#374151">${{venn.abc}}</text>
          <text x="95" y="127" text-anchor="middle" font-size="9" fill="#2563eb">${{esc((venn.label_a||'').slice(0,10))}}</text>
          <text x="205" y="127" text-anchor="middle" font-size="9" fill="#dc2626">${{esc((venn.label_b||'').slice(0,10))}}</text>
          <text x="150" y="248" text-anchor="middle" font-size="9" fill="#16a34a">${{esc((venn.label_c||'').slice(0,10))}}</text>
        </svg>
      </div>
    `;
  }}
}}

// ── Sprint 7 — Table de Wilcoxon ─────────────────────────────────
function buildWilcoxonTable() {{
  const container = document.getElementById('wilcoxon-table-container');
  if (!container) return;
  const stats = DATA.statistics && DATA.statistics.pairwise_wilcoxon || [];
  if (!stats.length) {{
    container.innerHTML = '<p style="color:var(--text-muted)">Pas assez de données pour les tests statistiques (min 2 concurrents).</p>';
    return;
  }}
  const rows = stats.map(s => {{
    const sigClass = s.significant ? 'stat-sig' : 'stat-ns';
    const sigLabel = s.significant ? '✓ Significative' : '○ Non significative';
    return `<tr>
      <td style="padding:.4rem .6rem;font-weight:600">${{esc(s.engine_a)}}</td>
      <td style="padding:.4rem .3rem;color:var(--text-muted)">vs</td>
      <td style="padding:.4rem .6rem;font-weight:600">${{esc(s.engine_b)}}</td>
      <td style="padding:.4rem .6rem;text-align:right;font-variant-numeric:tabular-nums">${{s.n_pairs}}</td>
      <td style="padding:.4rem .6rem;text-align:right;font-variant-numeric:tabular-nums">${{s.statistic}}</td>
      <td style="padding:.4rem .6rem;text-align:right;font-variant-numeric:tabular-nums">${{s.p_value}}</td>
      <td style="padding:.4rem .75rem"><span class="${{sigClass}}">${{sigLabel}}</span></td>
      <td style="padding:.4rem .75rem;font-size:.78rem;color:var(--text-muted);max-width:280px">${{esc(s.interpretation)}}</td>
    </tr>`;
  }}).join('');
  container.innerHTML = `
    <table style="border-collapse:collapse;font-size:.84rem;width:100%">
      <thead><tr style="background:var(--bg)">
        <th style="padding:.4rem .6rem;text-align:left;font-size:.75rem;text-transform:uppercase;letter-spacing:.04em">Concurrent A</th>
        <th></th>
        <th style="padding:.4rem .6rem;text-align:left;font-size:.75rem;text-transform:uppercase;letter-spacing:.04em">Concurrent B</th>
        <th style="padding:.4rem .6rem;text-align:right;font-size:.75rem">N paires</th>
        <th style="padding:.4rem .6rem;text-align:right;font-size:.75rem">W</th>
        <th style="padding:.4rem .6rem;text-align:right;font-size:.75rem">p-value</th>
        <th style="padding:.4rem .75rem;text-align:left;font-size:.75rem">Verdict</th>
        <th style="padding:.4rem .75rem;text-align:left;font-size:.75rem">Interprétation</th>
      </tr></thead>
      <tbody>${{rows}}</tbody>
    </table>
  `;
}}

// ── Sprint 7 — Clustering des erreurs ───────────────────────────
function buildErrorClusters() {{
  const container = document.getElementById('error-clusters-container');
  if (!container) return;
  const clusters = DATA.error_clusters || [];
  if (!clusters.length) {{
    container.innerHTML = `<p style="color:var(--text-muted)">Aucun cluster d'erreur détecté.</p>`;
    return;
  }}
  const cards = clusters.map(cl => {{
    const examplesHtml = (cl.examples || []).slice(0, 3).map(ex => {{
      const oldStr = ex.gt_fragment || '';
      const newStr = ex.ocr_fragment || '';
      return `<div class="cluster-ex">
        <span class="ex-old">${{esc(oldStr || '∅')}}</span>
        <span style="color:var(--text-muted)">→</span>
        <span class="ex-new">${{esc(newStr || '∅')}}</span>
        <span style="color:var(--text-muted);font-size:.72rem">(${{esc(ex.engine || '')}})</span>
      </div>`;
    }}).join('');
    return `<div class="cluster-card">
      <div class="cluster-label">Cluster #${{cl.cluster_id}} : ${{esc(cl.label)}}</div>
      <div class="cluster-count">${{cl.count}} cas détectés</div>
      <div class="cluster-examples">${{examplesHtml}}</div>
    </div>`;
  }}).join('');
  container.innerHTML = `<div class="cluster-grid">${{cards}}</div>`;
}}

// ── Sprint 7 — Matrice de corrélation ───────────────────────────
function initCorrelationMatrix() {{
  const sel = document.getElementById('corr-engine-select');
  if (!sel) return;
  const corrs = DATA.correlation_per_engine || [];
  sel.innerHTML = '';
  corrs.forEach(c => {{
    const opt = document.createElement('option');
    opt.value = c.engine; opt.textContent = c.engine;
    sel.appendChild(opt);
  }});
  renderCorrelationMatrix();
}}

function renderCorrelationMatrix() {{
  const container = document.getElementById('corr-matrix-container');
  if (!container) return;
  const sel = document.getElementById('corr-engine-select');
  const engineName = sel && sel.value;
  const corrs = DATA.correlation_per_engine || [];
  const entry = corrs.find(c => c.engine === engineName) || corrs[0];
  if (!entry || !entry.labels || !entry.matrix) {{
    container.innerHTML = '<p style="color:var(--text-muted)">Données insuffisantes.</p>';
    return;
  }}
  const labels = entry.labels;
  const matrix = entry.matrix;
  const n = labels.length;

  const labelNames = {{
    cer: 'CER', wer: 'WER', mer: 'MER', wil: 'WIL',
    quality_score: 'Qualité img', sharpness: 'Netteté',
    ligature: 'Ligatures', diacritic: 'Diacritiques',
  }};
  function corrColor(r) {{
    if (r >= 0.7)  return 'background:#dcfce7;color:#14532d';
    if (r >= 0.3)  return 'background:#f0fdf4;color:#166534';
    if (r >= -0.3) return 'background:#f8fafc;color:#374151';
    if (r >= -0.7) return 'background:#fef2f2;color:#991b1b';
    return 'background:#fee2e2;color:#7f1d1d';
  }}

  const headerRow = '<tr><th></th>' + labels.map(l =>
    `<th>${{esc(labelNames[l] || l)}}</th>`).join('') + '</tr>';
  const dataRows = matrix.map((row, i) =>
    '<tr><th style="text-align:right">' + esc(labelNames[labels[i]] || labels[i]) + '</th>' +
    row.map((v, j) => {{
      const style = corrColor(v);
      const display = i === j ? '1.00' : v.toFixed(2);
      return `<td style="${{style}}">${{display}}</td>`;
    }}).join('') + '</tr>'
  ).join('');

  container.innerHTML = `<table class="corr-table"><thead>${{headerRow}}</thead><tbody>${{dataRows}}</tbody></table>`;
}}

// ── Sprint 7 — URL stateful ──────────────────────────────────────
function updateURL(view, params) {{
  const hash = '#' + view + (params ? '?' + new URLSearchParams(params).toString() : '');
  history.replaceState(null, '', hash);
}}

function readURLState() {{
  const hash = location.hash.slice(1);
  const [view, query] = hash.split('?');
  const params = query ? Object.fromEntries(new URLSearchParams(query)) : {{}};
  return {{ view: view || 'ranking', params }};
}}

// ── Sprint 7 — Mode présentation ────────────────────────────────
let presentMode = false;
function togglePresentMode() {{
  presentMode = !presentMode;
  document.body.classList.toggle('present-mode', presentMode);
  const btn = document.getElementById('btn-present');
  if (btn) {{
    btn.classList.toggle('active', presentMode);
    btn.textContent = presentMode ? '⊡ Normal' : '⊞ Présentation';
  }}
}}

// ── Sprint 7 — Export CSV ────────────────────────────────────────
function exportCSV() {{
  const rows = [['doc_id','engine','cer','wer','mer','wil','duration','ligature_score','diacritic_score','difficulty_score','gini','anchor_score','length_ratio','is_hallucinating']];
  DATA.documents.forEach(doc => {{
    doc.engine_results.forEach(er => {{
      rows.push([
        doc.doc_id,
        er.engine,
        er.cer !== null ? (er.cer * 100).toFixed(4) : '',
        er.wer !== null ? (er.wer * 100).toFixed(4) : '',
        er.mer !== null ? (er.mer * 100).toFixed(4) : '',
        er.wil !== null ? (er.wil * 100).toFixed(4) : '',
        er.duration !== null ? er.duration : '',
        er.ligature_score !== null ? er.ligature_score : '',
        er.diacritic_score !== null ? er.diacritic_score : '',
        doc.difficulty_score !== undefined ? (doc.difficulty_score * 100).toFixed(2) : '',
        er.line_metrics ? er.line_metrics.gini.toFixed(6) : '',
        er.hallucination_metrics ? er.hallucination_metrics.anchor_score.toFixed(6) : '',
        er.hallucination_metrics ? er.hallucination_metrics.length_ratio.toFixed(4) : '',
        er.hallucination_metrics ? (er.hallucination_metrics.is_hallucinating ? '1' : '0') : '',
      ]);
    }});
  }});
  const csv = rows.map(r => r.map(v => JSON.stringify(String(v ?? ''))).join(',')).join('\\n');
  const blob = new Blob(['\ufeff' + csv], {{ type: 'text/csv;charset=utf-8' }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'picarones_metrics_' + DATA.meta.corpus_name.replace(/\\s+/g,'-') + '.csv';
  document.body.appendChild(a); a.click();
  setTimeout(() => {{ document.body.removeChild(a); URL.revokeObjectURL(url); }}, 100);
}}

// ── Vue Caractères ───────────────────────────────────────────────
let charViewBuilt = false;

function initCharView() {{
  charViewBuilt = true;
  // Remplir le sélecteur de moteur
  const sel = document.getElementById('char-engine-select');
  sel.innerHTML = '';
  DATA.engines.forEach(e => {{
    const opt = document.createElement('option');
    opt.value = e.name; opt.textContent = e.name;
    sel.appendChild(opt);
  }});
  renderCharView();
}}

function renderCharView() {{
  const engineName = document.getElementById('char-engine-select').value;
  const eng = DATA.engines.find(e => e.name === engineName);
  if (!eng) return;

  // Scores ligatures / diacritiques
  const scoresRow = document.getElementById('char-scores-row');
  const ligScore = eng.ligature_score;
  const diacScore = eng.diacritic_score;
  scoresRow.innerHTML = `
    <div class="stat">Ligatures <b>${{_scoreBadge(ligScore, 'Ligatures')}}</b></div>
    <div class="stat">Diacritiques <b>${{_scoreBadge(diacScore, 'Diacritiques')}}</b></div>
    ${{eng.aggregated_structure ? `
    <div class="stat">Précision lignes <b>${{_scoreBadge(eng.aggregated_structure.mean_line_accuracy, 'Précision nb lignes')}}</b></div>
    <div class="stat">Ordre lecture <b>${{_scoreBadge(eng.aggregated_structure.mean_reading_order_score, 'Score ordre de lecture')}}</b></div>
    ` : ''}}
    ${{eng.aggregated_image_quality ? `
    <div class="stat">Qualité image moy. <b>${{_scoreBadge(eng.aggregated_image_quality.mean_quality_score, 'Qualité image moyenne')}}</b></div>
    ` : ''}}
  `;

  // Matrice de confusion heatmap
  renderConfusionHeatmap(eng);

  // Détail ligatures
  renderLigatureDetail(eng);

  // Taxonomie détaillée
  renderTaxonomyDetail(eng);
}}

function renderConfusionHeatmap(eng) {{
  const container = document.getElementById('confusion-heatmap');
  const cm = eng.aggregated_confusion;
  if (!cm || !cm.matrix) {{
    container.innerHTML = '<p style="color:var(--text-muted)">Aucune donnée de confusion disponible.</p>';
    return;
  }}

  // Collecter les top confusions (substitutions uniquement, hors ∅)
  const pairs = [];
  for (const [gt, ocrs] of Object.entries(cm.matrix)) {{
    if (gt === '∅') continue;
    for (const [ocr, cnt] of Object.entries(ocrs)) {{
      if (ocr !== gt && ocr !== '∅' && cnt > 0) {{
        pairs.push({{ gt, ocr, cnt }});
      }}
    }}
  }}
  pairs.sort((a,b) => b.cnt - a.cnt);
  const top = pairs.slice(0, 30);

  if (!top.length) {{
    container.innerHTML = '<p style="color:var(--text-muted)">Aucune substitution détectée.</p>';
    return;
  }}

  // Heatmap sous forme de tableau compact
  const maxCnt = top[0].cnt;
  const rows = top.map(p => {{
    const intensity = Math.round((p.cnt / maxCnt) * 200 + 55);  // 55–255
    const bg = `rgb(${{intensity}},50,50)`;
    const fg = intensity > 150 ? '#fff' : '#222';
    return `<tr onclick="showConfusionExamples('${{esc(p.gt)}}','${{esc(p.ocr)}}')" style="cursor:pointer" title="GT='${{esc(p.gt)}}' → OCR='${{esc(p.ocr)}}' : ${{p.cnt}} fois">
      <td style="font-family:monospace;font-size:1.1rem;padding:.3rem .6rem;text-align:center">${{esc(p.gt)}}</td>
      <td style="padding:.1rem .3rem;color:var(--text-muted)">→</td>
      <td style="font-family:monospace;font-size:1.1rem;padding:.3rem .6rem;text-align:center">${{esc(p.ocr)}}</td>
      <td style="padding:.3rem 1rem">
        <div style="display:flex;align-items:center;gap:.5rem">
          <div style="width:${{Math.round(p.cnt/maxCnt*120)}}px;height:12px;border-radius:3px;background:${{bg}}"></div>
          <span style="font-size:.8rem;color:var(--text-muted)">${{p.cnt}}×</span>
        </div>
      </td>
    </tr>`;
  }}).join('');

  container.innerHTML = `
    <p style="font-size:.75rem;color:var(--text-muted);margin-bottom:.5rem">
      Cliquer sur une ligne pour voir les exemples dans la vue Document.
      Total substitutions : <b>${{cm.total_substitutions}}</b>
      · Insertions : <b>${{cm.total_insertions}}</b>
      · Suppressions : <b>${{cm.total_deletions}}</b>
    </p>
    <table style="border-collapse:collapse;font-size:.85rem">
      <thead><tr>
        <th style="padding:.3rem .6rem;text-align:left">GT</th>
        <th></th>
        <th style="padding:.3rem .6rem;text-align:left">OCR</th>
        <th style="padding:.3rem 1rem;text-align:left">Fréquence</th>
      </tr></thead>
      <tbody>${{rows}}</tbody>
    </table>
  `;
}}

function showConfusionExamples(gtChar, ocrChar) {{
  // Naviguer vers la vue Document en cherchant un exemple de cette confusion
  showView('document');
  const docWithConfusion = DATA.documents.find(doc =>
    doc.engine_results.some(er => {{
      const h = er.hypothesis || '';
      const g = doc.ground_truth || '';
      return g.includes(gtChar) && h.includes(ocrChar);
    }})
  );
  if (docWithConfusion) loadDocument(docWithConfusion.doc_id);
}}

function renderLigatureDetail(eng) {{
  const container = document.getElementById('ligature-detail');
  // Agrégation sur tous les documents pour ce moteur
  const ligData = {{}};
  DATA.documents.forEach(doc => {{
    const er = doc.engine_results.find(r => r.engine === eng.name);
    if (!er || !er.ligature_score) return;
    // On n'a que le score global par doc; pour le détail, utiliser aggregated_char_scores
  }});

  const agg = eng.aggregated_char_scores;
  if (!agg || !agg.ligature || !agg.ligature.per_ligature) {{
    const overallScore = eng.ligature_score;
    if (overallScore !== null && overallScore !== undefined) {{
      container.innerHTML = `<div class="stat">Score global ligatures : ${{_scoreBadge(overallScore, 'Ligatures')}}</div>`;
    }} else {{
      container.innerHTML = '<p style="color:var(--text-muted)">Aucune donnée ligature disponible (pas de ligatures dans le corpus).</p>';
    }}
    return;
  }}

  const perLig = agg.ligature.per_ligature;
  if (!Object.keys(perLig).length) {{
    container.innerHTML = '<p style="color:var(--text-muted)">Aucune ligature trouvée dans le corpus GT.</p>';
    return;
  }}

  const rows = Object.entries(perLig)
    .sort((a,b) => b[1].gt_count - a[1].gt_count)
    .map(([lig, d]) => {{
      const sc = d.score;
      const color = sc >= 0.9 ? '#16a34a' : sc >= 0.7 ? '#ca8a04' : '#dc2626';
      const barW = Math.round(sc * 120);
      return `<tr>
        <td style="font-family:monospace;font-size:1.2rem;padding:.3rem .6rem">${{esc(lig)}}</td>
        <td style="padding:.3rem .6rem;font-size:.8rem;color:var(--text-muted)">${{esc(lig.codePointAt(0).toString(16).toUpperCase().padStart(4,'0'))}}</td>
        <td style="padding:.3rem .6rem">${{d.gt_count}} GT</td>
        <td style="padding:.3rem .6rem">${{d.ocr_correct}} corrects</td>
        <td style="padding:.3rem 1rem">
          <div style="display:flex;align-items:center;gap:.5rem">
            <div style="width:${{barW}}px;height:10px;border-radius:3px;background:${{color}}"></div>
            <span style="color:${{color}};font-weight:600">${{(sc*100).toFixed(0)}}%</span>
          </div>
        </td>
      </tr>`;
    }}).join('');

  container.innerHTML = `
    <table style="border-collapse:collapse;font-size:.85rem">
      <thead><tr>
        <th style="padding:.3rem .6rem;text-align:left">Ligature</th>
        <th style="padding:.3rem .6rem;text-align:left">Unicode</th>
        <th style="padding:.3rem .6rem">GT</th>
        <th style="padding:.3rem .6rem">Corrects</th>
        <th style="padding:.3rem 1rem;text-align:left">Score</th>
      </tr></thead>
      <tbody>${{rows}}</tbody>
    </table>
  `;
}}

function renderTaxonomyDetail(eng) {{
  const container = document.getElementById('taxonomy-detail');
  const tax = eng.aggregated_taxonomy;
  if (!tax || !tax.counts) {{
    container.innerHTML = '<p style="color:var(--text-muted)">Aucune donnée taxonomique disponible.</p>';
    return;
  }}

  const classNames = {{
    visual_confusion: '1 — Confusion visuelle',
    diacritic_error: '2 — Erreur diacritique',
    case_error: '3 — Erreur de casse',
    ligature_error: '4 — Ligature',
    abbreviation_error: '5 — Abréviation',
    hapax: '6 — Hapax',
    segmentation_error: '7 — Segmentation',
    oov_character: '8 — Hors-vocabulaire',
    lacuna: '9 — Lacune',
  }};
  const total = tax.total_errors || 1;
  const maxCnt = Math.max(...Object.values(tax.counts));

  const rows = Object.entries(tax.counts)
    .filter(([, cnt]) => cnt > 0)
    .sort((a,b) => b[1]-a[1])
    .map(([cls, cnt]) => {{
      const pctVal = (cnt / total * 100).toFixed(1);
      const barW = maxCnt > 0 ? Math.round(cnt/maxCnt * 200) : 0;
      return `<tr>
        <td style="padding:.3rem .6rem;font-size:.85rem">${{esc(classNames[cls] || cls)}}</td>
        <td style="padding:.3rem .6rem;text-align:right;font-variant-numeric:tabular-nums">${{cnt}}</td>
        <td style="padding:.3rem 1rem">
          <div style="display:flex;align-items:center;gap:.5rem">
            <div style="width:${{barW}}px;height:10px;border-radius:3px;background:#6366f1"></div>
            <span style="color:var(--text-muted);font-size:.8rem">${{pctVal}}%</span>
          </div>
        </td>
      </tr>`;
    }}).join('');

  container.innerHTML = `
    <p style="font-size:.75rem;color:var(--text-muted);margin-bottom:.5rem">Total : <b>${{tax.total_errors}}</b> erreurs classifiées.</p>
    <table style="border-collapse:collapse;font-size:.85rem;min-width:400px">
      <thead><tr>
        <th style="padding:.3rem .6rem;text-align:left">Classe</th>
        <th style="padding:.3rem .6rem;text-align:right">N</th>
        <th style="padding:.3rem 1rem;text-align:left">Proportion</th>
      </tr></thead>
      <tbody>${{rows}}</tbody>
    </table>
  `;
}}

// ── Init ────────────────────────────────────────────────────────
function applyI18n() {{
  // Applique les traductions aux éléments avec data-i18n (textContent)
  document.querySelectorAll('[data-i18n]').forEach(el => {{
    const key = el.getAttribute('data-i18n');
    if (I18N[key] !== undefined) el.textContent = I18N[key];
  }});
  // Options de select avec data-i18n-opt
  document.querySelectorAll('[data-i18n-opt]').forEach(el => {{
    const key = el.getAttribute('data-i18n-opt');
    if (I18N[key] !== undefined) el.textContent = I18N[key];
  }});
  // Tooltips des th via id
  const thMap = {{
    'th-cer-diplo':  'col_cer_diplo_title',
    'th-ligatures':  'col_ligatures_title',
    'th-diacritics': 'col_diacritics_title',
    'th-gini':       'col_gini_title',
    'th-anchor':     'col_anchor_title',
    'th-overnorm':   'col_overnorm_title',
  }};
  Object.entries(thMap).forEach(([id, key]) => {{
    const el = document.getElementById(id);
    if (el && I18N[key]) el.title = I18N[key];
  }});
}}

function init() {{
  // i18n
  applyI18n();

  // Méta nav
  const d = new Date(DATA.meta.run_date);
  const locale = I18N.date_locale || 'fr-FR';
  const fmt = d.toLocaleDateString(locale, {{ year:'numeric', month:'short', day:'numeric' }});
  document.getElementById('nav-meta').textContent =
    DATA.meta.corpus_name + ' · ' + fmt;
  document.getElementById('footer-date').textContent =
    (I18N.footer_generated || 'Rapport généré le') + ' ' + fmt;

  // Sélecteur moteur galerie
  const sel = document.getElementById('gallery-engine-select');
  DATA.engines.forEach(e => {{
    const opt = document.createElement('option');
    opt.value = e.name; opt.textContent = e.name;
    sel.appendChild(opt);
  }});

  renderRanking();
  renderGallery();
  buildDocList();

  // Restaurer l'état depuis l'URL
  const {{ view, params }} = readURLState();
  if (view && view !== 'ranking') {{
    _switchView(view);  // appel direct pour ne pas écraser l'URL
    if (view === 'document' && params.doc) {{
      loadDocument(params.doc);
    }}
  }}

  // Gérer le bouton retour
  window.addEventListener('popstate', () => {{
    const {{ view: v, params: p }} = readURLState();
    _switchView(v || 'ranking');
    if ((v === 'document') && p.doc) loadDocument(p.doc);
  }});
}}

document.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>
"""


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
        """
        self.benchmark = benchmark
        self.images_b64: dict[str, str] = images_b64 or {}
        self.lang = lang

        # Récupérer les images embarquées dans les metadata (fixtures)
        if not self.images_b64:
            self.images_b64 = benchmark.metadata.get("_images_b64", {})  # type: ignore[assignment]

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
        report_json = json.dumps(report_data, ensure_ascii=False, separators=(",", ":"))
        i18n_json = json.dumps(labels, ensure_ascii=False, separators=(",", ":"))

        html = _HTML_TEMPLATE.format(
            corpus_name=self.benchmark.corpus_name,
            picarones_version=self.benchmark.picarones_version,
            report_data_json=report_json,
            i18n_json=i18n_json,
            html_lang=labels.get("html_lang", "fr"),
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
