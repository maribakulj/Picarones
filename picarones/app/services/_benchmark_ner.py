"""NER aggregation post-bench — module extrait du god-module
``benchmark_runner.py`` lors de la Phase 6 de l'audit code-quality
(2026-05).

Avant la Phase 6, ``benchmark_runner.py`` faisait 1 700 LOC avec
budget de complaisance verrouillé à 1 750.  L'extraction des
fonctions NER (~130 LOC) est la première étape du découpage en
sous-modules thématiques :

- ``_benchmark_ner.py`` (ce fichier) — calcul + agrégation NER
- (à venir) ``_benchmark_conversions.py`` — mappings
- (à venir) ``_benchmark_execution.py`` — orchestration
- (à venir) ``_benchmark_helpers.py`` — utilitaires partagés

Les deux fonctions exportées :

- :func:`attach_ner_metrics_to_benchmark` — Sprint D.2.e.  Itère
  les ``DocumentResult`` de chaque ``EngineReport``, extrait les
  entités côté hypothèse via le ``entity_extractor`` injecté,
  compare avec les entités GT et stocke le résultat sur
  ``dr.ner_metrics`` + agrégat corpus-level sur
  ``EngineReport.aggregated_ner``.

- :func:`aggregate_ner_metrics` — Sprint D.2.e.  Agrège les
  ``ner_metrics`` au niveau engine : recalcule precision/recall/F1
  *micro* depuis les sommes TP/FP/FN, plus détail par catégorie.

Naming : préfixe ``_`` retiré sur la version publique (les noms
``_attach_*`` / ``_aggregate_*`` historiques restent disponibles
comme alias dans ``benchmark_runner.py`` pour compat des appels
internes).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from picarones.evaluation.corpus import Corpus

logger = logging.getLogger(__name__)


def attach_ner_metrics_to_benchmark(
    benchmark_result: Any,
    corpus: "Corpus",
    entity_extractor: Callable[[str], list[dict]],
) -> None:
    """Calcule + attache les métriques NER post-bench.

    Parcourt les ``DocumentResult`` de chaque ``EngineReport`` et,
    pour chaque doc dont la GT possède un niveau ``ENTITIES``,
    invoque ``entity_extractor(hypothesis)`` puis
    ``compute_ner_metrics`` contre les entités de la GT.  Le
    résultat est attaché sur ``dr.ner_metrics``.  Les agrégats
    par engine sont calculés via :func:`aggregate_ner_metrics` et
    stockés sur ``EngineReport.aggregated_ner``.

    Tolérance : un échec d'extraction ou de calcul sur un doc
    spécifique est dégradé en warning ; le bench n'est pas
    interrompu.

    Notes (mai 2026, audit B3-final)
    --------------------------------
    Les ``doc_id`` portés par ``DocumentResult`` (issus de
    ``DocumentRef.id`` côté ``CorpusSpec``) sont normalisés via
    ``_safe_doc_id`` (NFD + alphanum + ``_.-/``).  Pour éviter une
    lookup silencieusement vide entre ``benchmark_result`` (clés
    normalisées) et ``corpus`` (clés legacy potentiellement avec
    espaces/accents), on indexe le ``corpus`` avec la même
    normalisation.  Sans ça, un corpus contenant un
    ``Document(doc_id="Image 01")`` voyait sa NER skippée puisque
    ``docs_by_id.get("Image_01")`` retournait ``None``.
    """
    from picarones.app.services._benchmark_conversions import _safe_doc_id
    from picarones.domain.artifacts import ArtifactType
    from picarones.evaluation.metrics.ner import compute_ner_metrics

    docs_by_id = {_safe_doc_id(d.doc_id): d for d in corpus.documents}

    for report in benchmark_result.engine_reports:
        n_done = 0
        for dr in report.document_results:
            if dr.engine_error is not None or not dr.hypothesis:
                continue
            doc = docs_by_id.get(dr.doc_id)
            if doc is None or not doc.has_gt(ArtifactType.ENTITIES):
                continue
            try:
                gt_payload = doc.get_gt(ArtifactType.ENTITIES)
                gt_entities = (
                    list(gt_payload.entities) if gt_payload else []
                )
                hyp_entities = entity_extractor(dr.hypothesis) or []
                dr.ner_metrics = compute_ner_metrics(
                    gt_entities, hyp_entities,
                )
                n_done += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[ner.attach] %s/%s : extraction/comparaison "
                    "NER dégradée : %s",
                    report.engine_name, dr.doc_id, exc,
                )

        if n_done > 0:
            report.aggregated_ner = aggregate_ner_metrics(
                report.document_results,
            )
            logger.info(
                "[ner] %d documents évalués pour engine '%s'.",
                n_done, report.engine_name,
            )


def aggregate_ner_metrics(doc_results: list) -> dict | None:
    """Agrège les ``ner_metrics`` au niveau engine.

    Recalcule precision/recall/F1 *micro* à partir des sommes
    globales TP/FP/FN, plus le détail par catégorie, plus les
    compteurs totaux d'hallucinations et d'entités manquées.
    """
    relevant = [
        dr for dr in doc_results if dr.ner_metrics is not None
    ]
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
            cat_tp.setdefault(cat, 0)
            cat_fp.setdefault(cat, 0)
            cat_fn.setdefault(cat, 0)
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
        return {
            "precision": p, "recall": r, "f1": f1, "support": tp + fn,
        }

    return {
        "global": _prf(total_tp, total_fp, total_fn),
        "per_category": {
            cat: _prf(cat_tp[cat], cat_fp[cat], cat_fn[cat])
            for cat in sorted(cat_tp)
        },
        "n_documents": len(relevant),
        "total_hallucinated": total_hallucinated,
        "total_missed": total_missed,
        "iou_threshold": iou_threshold,
    }


__all__ = [
    "attach_ner_metrics_to_benchmark",
    "aggregate_ner_metrics",
]
