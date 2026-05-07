"""Construction de la liste ``documents`` (vue galerie + vue détail).

Pour chaque document du corpus, agrège les hypothèses de tous les
moteurs avec leurs métriques, le diff caractère par caractère, et
les champs spécifiques aux pipelines OCR+LLM (intermédiaire, mode,
sur-normalisation).

:func:`annotate_documents_with_difficulty` enrichit ensuite chaque
document avec son score de difficulté intrinsèque (Sprint 7).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from picarones.evaluation import compute_char_diff, compute_word_diff
from picarones.evaluation.metrics.difficulty import (
    compute_all_difficulties,
    difficulty_label,
)
from picarones.reports_v2.html.data._helpers import safe_round

if TYPE_CHECKING:
    from picarones.evaluation.benchmark_result import BenchmarkResult


def build_documents(
    benchmark: "BenchmarkResult", images_b64: dict[str, str],
) -> list[dict]:
    """Retourne la liste ordonnée des documents prêts pour le template.

    L'ordre des documents préserve l'ordre d'apparition (premier moteur
    d'abord, puis compléments depuis les moteurs suivants si certains
    documents ne sont pas couverts par tous les moteurs).
    """
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

    documents: list[dict] = []
    engine_names = [r.engine_name for r in benchmark.engine_reports]
    for doc_id in doc_ids_ordered:
        engine_results: list[dict] = []
        gt = ""
        image_path = ""
        for engine_name in engine_names:
            dr = doc_engine_map[doc_id].get(engine_name)
            if dr is None:
                continue
            gt = dr.ground_truth
            image_path = dr.image_path
            er_entry = _build_engine_result_entry(engine_name, dr)
            engine_results.append(er_entry)

        # CER moyen sur ce document (pour le badge galerie)
        cer_values = [er["cer"] for er in engine_results if er["error"] is None]
        mean_cer = sum(cer_values) / len(cer_values) if cer_values else 1.0
        best_engine = min(engine_results, key=lambda x: x["cer"], default=None)

        # Script type (depuis metadata par document si disponible)
        script_type = ""
        first_engine = engine_names[0] if engine_names else None
        first_dr = doc_engine_map[doc_id].get(first_engine)
        if first_dr and first_dr.image_quality:
            script_type = first_dr.image_quality.get("script_type", "")

        documents.append({
            "doc_id": doc_id,
            "image_path": image_path,
            "image_b64": images_b64.get(doc_id, ""),
            "ground_truth": gt,
            "mean_cer": safe_round(mean_cer),
            "best_engine": best_engine["engine"] if best_engine else "",
            "engine_results": engine_results,
            "script_type": script_type,
        })
    return documents


def _build_engine_result_entry(engine_name: str, dr) -> dict:
    """Construit une entrée moteur pour un document donné (extrait pour lisibilité)."""
    diff_ops = compute_char_diff(dr.ground_truth, dr.hypothesis)
    er_entry: dict = {
        "engine": engine_name,
        "hypothesis": dr.hypothesis,
        "cer": safe_round(dr.metrics.cer),
        "cer_diplomatic": safe_round(dr.metrics.cer_diplomatic) if dr.metrics.cer_diplomatic is not None else None,
        "wer": safe_round(dr.metrics.wer),
        "mer": safe_round(dr.metrics.mer),
        "wil": safe_round(dr.metrics.wil),
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
        er_entry["ligature_score"] = safe_round(dr.char_scores.get("ligature", {}).get("score"))
        er_entry["diacritic_score"] = safe_round(dr.char_scores.get("diacritic", {}).get("score"))
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
    return er_entry


def annotate_documents_with_difficulty(
    benchmark: "BenchmarkResult", documents: list[dict],
) -> None:
    """Annote chaque document du dict avec son score de difficulté (Sprint 7).

    Modifie ``documents`` en place. Les valeurs par défaut ``0.5`` /
    ``"Modéré"`` sont retournées si la difficulté n'a pas pu être
    calculée (par exemple corpus dégénéré).
    """
    doc_ids_ordered = [d["doc_id"] for d in documents]
    gt_map = {d["doc_id"]: d["ground_truth"] for d in documents}
    cer_map: dict[str, dict[str, float]] = {d["doc_id"]: {} for d in documents}
    iq_map: dict[str, float] = {}
    for report in benchmark.engine_reports:
        for dr in report.document_results:
            cer_map.setdefault(dr.doc_id, {})[report.engine_name] = safe_round(dr.metrics.cer)
            if dr.image_quality and "quality_score" in dr.image_quality:
                iq_map[dr.doc_id] = dr.image_quality["quality_score"]
    difficulty_scores = compute_all_difficulties(
        doc_ids=doc_ids_ordered,
        ground_truths=gt_map,
        cer_map=cer_map,
        image_quality_map=iq_map or None,
    )
    for doc in documents:
        ds = difficulty_scores.get(doc["doc_id"])
        if ds:
            doc["difficulty_score"] = safe_round(ds.score)
            doc["difficulty_label"] = difficulty_label(ds.score)
        else:
            doc["difficulty_score"] = 0.5
            doc["difficulty_label"] = "Modéré"


__all__ = ["build_documents", "annotate_documents_with_difficulty"]
