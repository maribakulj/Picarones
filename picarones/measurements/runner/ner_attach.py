"""Câblage NER au post-process du benchmark (Sprint 40).

Le runner appelle :func:`_attach_ner_metrics` après que tous les
documents ont été calculés, pour les moteurs où la GT possède un
niveau ``ENTITIES`` (Sprint 32 — multi-level GT).

L'extracteur NER est typiquement un wrapper :class:`SpacyEntityExtractor`
construit via :func:`picarones.measurements.ner_backends.get_extractor`.
"""

from __future__ import annotations

import logging

from picarones.evaluation.corpus import Corpus

logger = logging.getLogger(__name__)


def _attach_ner_metrics(
    corpus: Corpus,
    doc_results: list,
    entity_extractor: callable,
) -> None:
    """Calcule et attache ``DocumentResult.ner_metrics`` pour chaque doc
    dont la GT possède un niveau ``ENTITIES`` (Sprint 32).

    L'extracteur est appelé sur l'hypothèse OCR ``dr.hypothesis``.
    Les erreurs sont dégradées en warnings (pas de propagation) afin
    de ne pas casser le benchmark si un document spécifique fait
    crasher le NER.
    """
    try:
        from picarones.evaluation.corpus import GTLevel
        from picarones.measurements.ner import compute_ner_metrics
    except ImportError as exc:
        logger.warning("[ner.attach] imports indisponibles : %s", exc)
        return

    docs_by_id = {d.doc_id: d for d in corpus.documents}
    n_done = 0
    for dr in doc_results:
        if dr.engine_error is not None or not dr.hypothesis:
            continue
        doc = docs_by_id.get(dr.doc_id)
        if doc is None or not doc.has_gt(GTLevel.ENTITIES):
            continue
        try:
            gt_payload = doc.get_gt(GTLevel.ENTITIES)
            gt_entities = list(gt_payload.entities) if gt_payload else []
            hyp_entities = entity_extractor(dr.hypothesis) or []
            dr.ner_metrics = compute_ner_metrics(gt_entities, hyp_entities)
            n_done += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[ner.attach] %s : extraction/comparaison NER dégradée : %s",
                dr.doc_id, exc,
            )

    if n_done > 0:
        logger.info("[ner] %d documents évalués pour NER.", n_done)


def _aggregate_ner(doc_results: list) -> "dict | None":
    """Agrège les métriques NER au niveau du moteur.

    Recalcule precision/recall/F1 *micro* à partir des sommes globales
    de TP/FP/FN, plus le détail par catégorie, plus les compteurs
    totaux d'hallucinations et d'entités manquées.
    """
    relevant = [dr for dr in doc_results if dr.ner_metrics is not None]
    if not relevant:
        return None

    total_tp = 0
    total_fp = 0
    total_fn = 0
    cat_tp: dict[str, int] = {}
    cat_fp: dict[str, int] = {}
    cat_fn: dict[str, int] = {}
    total_hallucinated = 0
    total_missed = 0
    iou_threshold = 0.5

    for dr in relevant:
        m = dr.ner_metrics
        total_tp += int(m.get("true_positives", 0))
        total_fp += int(m.get("false_positives", 0))
        total_fn += int(m.get("false_negatives", 0))
        total_hallucinated += len(m.get("hallucinated_entities", []) or [])
        total_missed += len(m.get("missed_entities", []) or [])
        iou_threshold = float(m.get("iou_threshold", iou_threshold))
        for cat, stats in (m.get("per_category") or {}).items():
            cat_tp[cat] = cat_tp.get(cat, 0)
            cat_fp[cat] = cat_fp.get(cat, 0)
            cat_fn[cat] = cat_fn.get(cat, 0)
            # Reconstitue les sommes par catégorie via support et P/R
            support = int(stats.get("support", 0))
            recall = float(stats.get("recall", 0.0))
            precision = float(stats.get("precision", 0.0))
            tp_cat = round(support * recall) if support > 0 else 0
            fn_cat = max(0, support - tp_cat)
            fp_cat = (
                round(tp_cat * (1 - precision) / precision)
                if precision > 0 else 0
            )
            cat_tp[cat] += tp_cat
            cat_fp[cat] += fp_cat
            cat_fn[cat] += fn_cat

    def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {"precision": p, "recall": r, "f1": f1, "support": tp + fn}

    return {
        "global": _prf(total_tp, total_fp, total_fn),
        "per_category": {
            cat: _prf(cat_tp[cat], cat_fp[cat], cat_fn[cat])
            for cat in sorted(set(cat_tp) | set(cat_fp) | set(cat_fn))
        },
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "hallucinated_total": total_hallucinated,
        "missed_total": total_missed,
        "doc_count": len(relevant),
        "iou_threshold": iou_threshold,
    }


__all__ = ["_aggregate_ner", "_attach_ner_metrics"]
