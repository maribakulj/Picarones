"""Enregistrement des hooks de métriques natifs de Picarones.

Chantier 2 du plan d'évolution post-Sprint 97.

Ce module **migre** les 12 hooks document-level et 12 agrégateurs
corpus-level qui étaient codés en dur dans
``picarones.measurements.runner._compute_document_result`` et autour de la
boucle d'agrégation (lignes 794-827 du runner pré-chantier-2).

Approche additive — rétrocompat stricte
---------------------------------------
Tous les hooks sont enregistrés sur les profils ``standard``,
``philological``, ``diagnostics`` et ``full`` (i.e. activés par
défaut quand le runner est appelé sans paramètre ``profile``). Le
profil ``minimal`` n'active aucun hook (pour bench massif où seul
CER/WER comptent). Les profils ``economics`` et ``pipeline`` sont
réservés pour des hooks futurs.

L'import de ce module **suffit** à peupler les registres :
:mod:`picarones.core.metric_hooks` se contente d'exposer les
décorateurs ; le runner ne dépend que d'une seule fonction —
``select_document_hooks(profile)`` — pour découvrir les hooks actifs.

Liste complète des hooks (Sprint d'origine)
-------------------------------------------
**Document-level** (12) :

- ``confusion``           (Sprint 5)  — ``confusion_matrix``
- ``char_scores``         (Sprint 5)  — ``char_scores``
- ``taxonomy``            (Sprint 5)  — ``taxonomy``
- ``structure``           (Sprint 5)  — ``structure``
- ``image_quality``       (Sprint 5)  — ``image_quality``
- ``line_metrics``        (Sprint 10) — ``line_metrics``
- ``hallucination``       (Sprint 10) — ``hallucination_metrics``
- ``calibration``         (Sprint 42) — ``calibration_metrics``
- ``philological``        (Sprint 61) — ``philological_metrics``
- ``searchability``       (Sprint 86) — ``searchability_metrics``
- ``numerical_sequences`` (Sprint 86) — ``numerical_sequence_metrics``
- ``readability``         (Sprint 87) — ``readability_metrics``

**Corpus-level** (12) : un agrégateur par hook documentaire,
remplissant le champ ``aggregated_*`` correspondant du
``EngineReport``.

Le hook ``ner`` (Sprint 40) reste hors de ce mécanisme : il dépend
d'un ``EntityExtractor`` injecté à la main par l'utilisateur, ce
qui n'entre pas dans la sémantique des profils.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

from picarones.core.metric_hooks import (
    PROFILE_DIAGNOSTICS,
    PROFILE_FULL,
    PROFILE_PHILOLOGICAL,
    PROFILE_STANDARD,
    register_corpus_aggregator,
    register_document_metric,
)

logger = logging.getLogger(__name__)


# Profils dans lesquels les 12 hooks "standard" s'activent. Égalent
# par construction le comportement runner pré-chantier-2 ; le profil
# ``minimal`` est volontairement absent.
_STANDARD_PROFILES = (
    PROFILE_STANDARD,
    PROFILE_PHILOLOGICAL,
    PROFILE_DIAGNOSTICS,
    PROFILE_FULL,
)


# ──────────────────────────────────────────────────────────────────────────
# Helper de calibration (déplacé depuis runner.py — chantier 2)
# ──────────────────────────────────────────────────────────────────────────


def calibration_from_engine_result(
    ground_truth: str,
    token_confidences: list,
) -> Optional[dict]:
    """Aligne les ``token_confidences`` du moteur sur la GT (bag-of-words)
    pour produire les listes parallèles ``confidences`` / ``is_correct``,
    puis appelle ``compute_calibration_metrics`` (Sprint 39).

    Convention d'alignement (proxy bag-of-words avec multiplicité, comme
    ``oracle_token_recall`` du Sprint 35) : un token de l'hypothèse est
    "correct" si la GT contient encore une occurrence de ce token.

    Les confidences ``> 1.0`` sont supposées en pourcentage et
    normalisées à ``[0, 1]``. Les confidences négatives (Tesseract met
    -1 pour les non-mots) sont ignorées.
    """
    from picarones.measurements.calibration import compute_calibration_metrics

    if not token_confidences:
        return None

    gt_counter = Counter((ground_truth or "").split())
    confidences: list[float] = []
    is_correct: list[int] = []

    for tc in token_confidences:
        if not isinstance(tc, dict):
            continue
        token = str(tc.get("token", ""))
        if not token:
            continue
        try:
            conf = float(tc.get("confidence"))
        except (TypeError, ValueError):
            continue
        if conf < 0:
            continue
        if conf > 1.0:
            conf = conf / 100.0
        if not 0.0 <= conf <= 1.0:
            continue
        if gt_counter[token] > 0:
            is_correct.append(1)
            gt_counter[token] -= 1
        else:
            is_correct.append(0)
        confidences.append(conf)

    if not confidences:
        return None
    return compute_calibration_metrics(confidences, is_correct)


# ──────────────────────────────────────────────────────────────────────────
# Document-level hooks (12)
# ──────────────────────────────────────────────────────────────────────────


@register_document_metric(
    name="confusion",
    attribute="confusion_matrix",
    profiles=_STANDARD_PROFILES,
    requires_success=True,
)
def _confusion_hook(*, ground_truth, hypothesis, **_):
    from picarones.measurements.confusion import build_confusion_matrix
    return build_confusion_matrix(ground_truth, hypothesis).as_dict()


@register_document_metric(
    name="char_scores",
    attribute="char_scores",
    profiles=_STANDARD_PROFILES,
    requires_success=True,
)
def _char_scores_hook(*, ground_truth, hypothesis, **_):
    from picarones.measurements.char_scores import (
        compute_diacritic_score,
        compute_ligature_score,
    )
    lig = compute_ligature_score(ground_truth, hypothesis)
    diac = compute_diacritic_score(ground_truth, hypothesis)
    return {"ligature": lig.as_dict(), "diacritic": diac.as_dict()}


@register_document_metric(
    name="taxonomy",
    attribute="taxonomy",
    profiles=_STANDARD_PROFILES,
    requires_success=True,
)
def _taxonomy_hook(*, ground_truth, hypothesis, **_):
    from picarones.measurements.taxonomy import classify_errors
    return classify_errors(ground_truth, hypothesis).as_dict()


@register_document_metric(
    name="structure",
    attribute="structure",
    profiles=_STANDARD_PROFILES,
    requires_success=True,
)
def _structure_hook(*, ground_truth, hypothesis, **_):
    from picarones.measurements.structure import analyze_structure
    return analyze_structure(ground_truth, hypothesis).as_dict()


@register_document_metric(
    name="line_metrics",
    attribute="line_metrics",
    profiles=_STANDARD_PROFILES,
    requires_success=True,
)
def _line_metrics_hook(*, ground_truth, hypothesis, **_):
    from picarones.measurements.line_metrics import compute_line_metrics
    return compute_line_metrics(ground_truth, hypothesis).as_dict()


@register_document_metric(
    name="hallucination",
    attribute="hallucination_metrics",
    profiles=_STANDARD_PROFILES,
    requires_success=True,
)
def _hallucination_hook(*, ground_truth, hypothesis, **_):
    from picarones.measurements.hallucination import compute_hallucination_metrics
    return compute_hallucination_metrics(ground_truth, hypothesis).as_dict()


@register_document_metric(
    name="calibration",
    attribute="calibration_metrics",
    profiles=_STANDARD_PROFILES,
    requires_token_confidences=True,
)
def _calibration_hook(*, ground_truth, ocr_result, **_):
    return calibration_from_engine_result(
        ground_truth, ocr_result.token_confidences,
    )


@register_document_metric(
    name="image_quality",
    attribute="image_quality",
    profiles=_STANDARD_PROFILES,
    # Pas de requires_success : on analyse l'image quel que soit le
    # résultat OCR (pour comparer un échec OCR à la qualité image).
)
def _image_quality_hook(*, image_path, **_):
    from picarones.measurements.image_quality import analyze_image_quality
    iq = analyze_image_quality(image_path)
    if iq.error is not None:
        return None
    return iq.as_dict()


@register_document_metric(
    name="philological",
    attribute="philological_metrics",
    profiles=_STANDARD_PROFILES,
    # Pas de requires_success : le runner pré-chantier-2 calculait
    # même sur échec OCR (avec hyp=""). Les modules philologiques
    # retournent ``None`` quand la GT n'a pas de signal exploitable
    # — comportement adaptive intact.
)
def _philological_hook(*, ground_truth, hypothesis, **_):
    from picarones.measurements.philological_hooks import compute_philological_metrics
    return compute_philological_metrics(ground_truth, hypothesis)


@register_document_metric(
    name="searchability",
    attribute="searchability_metrics",
    profiles=_STANDARD_PROFILES,
)
def _searchability_hook(*, ground_truth, hypothesis, **_):
    from picarones.measurements.searchability_hooks import compute_searchability_metrics
    return compute_searchability_metrics(ground_truth, hypothesis)


@register_document_metric(
    name="numerical_sequences",
    attribute="numerical_sequence_metrics",
    profiles=_STANDARD_PROFILES,
)
def _numerical_sequences_hook(*, ground_truth, hypothesis, **_):
    from picarones.measurements.numerical_sequences_hooks import (
        compute_numerical_sequence_metrics_adaptive,
    )
    return compute_numerical_sequence_metrics_adaptive(ground_truth, hypothesis)


@register_document_metric(
    name="readability",
    attribute="readability_metrics",
    profiles=_STANDARD_PROFILES,
)
def _readability_hook(*, ground_truth, hypothesis, corpus_lang, **_):
    from picarones.measurements.readability_hooks import compute_readability_metrics
    return compute_readability_metrics(ground_truth, hypothesis, lang=corpus_lang)


# ──────────────────────────────────────────────────────────────────────────
# Corpus-level aggregators (12)
# ──────────────────────────────────────────────────────────────────────────


@register_corpus_aggregator(
    name="confusion",
    attribute="aggregated_confusion",
    profiles=_STANDARD_PROFILES,
)
def _aggregate_confusion(doc_results: list) -> Optional[dict]:
    from picarones.measurements.confusion import (
        ConfusionMatrix, aggregate_confusion_matrices,
    )
    try:
        matrices = [
            ConfusionMatrix(**dr.confusion_matrix)
            for dr in doc_results
            if dr.confusion_matrix is not None
        ]
        if not matrices:
            return None
        return aggregate_confusion_matrices(matrices).as_compact_dict(min_count=2)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[runner] aggregate_confusion : agrégation indisponible (%s) — "
            "matrice de confusion absente du rapport pour ce moteur",
            exc,
        )
        return None


@register_corpus_aggregator(
    name="char_scores",
    attribute="aggregated_char_scores",
    profiles=_STANDARD_PROFILES,
)
def _aggregate_char_scores(doc_results: list) -> Optional[dict]:
    from picarones.measurements.char_scores import (
        DiacriticScore,
        LigatureScore,
        aggregate_diacritic_scores,
        aggregate_ligature_scores,
    )
    lig_scores = [
        LigatureScore(**dr.char_scores["ligature"])
        for dr in doc_results
        if dr.char_scores is not None
    ]
    diac_scores = [
        DiacriticScore(**dr.char_scores["diacritic"])
        for dr in doc_results
        if dr.char_scores is not None
    ]
    if not lig_scores:
        return None
    return {
        "ligature": aggregate_ligature_scores(lig_scores),
        "diacritic": aggregate_diacritic_scores(diac_scores),
    }


@register_corpus_aggregator(
    name="taxonomy",
    attribute="aggregated_taxonomy",
    profiles=_STANDARD_PROFILES,
)
def _aggregate_taxonomy(doc_results: list) -> Optional[dict]:
    from picarones.measurements.taxonomy import TaxonomyResult, aggregate_taxonomy
    results = [
        TaxonomyResult.from_dict(dr.taxonomy)
        for dr in doc_results
        if dr.taxonomy is not None
    ]
    if not results:
        return None
    return aggregate_taxonomy(results)


@register_corpus_aggregator(
    name="structure",
    attribute="aggregated_structure",
    profiles=_STANDARD_PROFILES,
)
def _aggregate_structure(doc_results: list) -> Optional[dict]:
    from picarones.measurements.structure import StructureResult, aggregate_structure
    results = [
        StructureResult.from_dict(dr.structure)
        for dr in doc_results
        if dr.structure is not None
    ]
    if not results:
        return None
    return aggregate_structure(results)


@register_corpus_aggregator(
    name="image_quality",
    attribute="aggregated_image_quality",
    profiles=_STANDARD_PROFILES,
)
def _aggregate_image_quality(doc_results: list) -> Optional[dict]:
    from picarones.measurements.image_quality import (
        ImageQualityResult, aggregate_image_quality,
    )
    results = [
        ImageQualityResult.from_dict(dr.image_quality)
        for dr in doc_results
        if dr.image_quality is not None
    ]
    if not results:
        return None
    return aggregate_image_quality(results)


@register_corpus_aggregator(
    name="line_metrics",
    attribute="aggregated_line_metrics",
    profiles=_STANDARD_PROFILES,
)
def _aggregate_line_metrics(doc_results: list) -> Optional[dict]:
    from picarones.measurements.line_metrics import (
        LineMetrics, aggregate_line_metrics,
    )
    results = [
        LineMetrics.from_dict(dr.line_metrics)
        for dr in doc_results
        if dr.line_metrics is not None
    ]
    if not results:
        return None
    return aggregate_line_metrics(results)


@register_corpus_aggregator(
    name="hallucination",
    attribute="aggregated_hallucination",
    profiles=_STANDARD_PROFILES,
)
def _aggregate_hallucination(doc_results: list) -> Optional[dict]:
    from picarones.measurements.hallucination import (
        HallucinationMetrics, aggregate_hallucination_metrics,
    )
    results = [
        HallucinationMetrics.from_dict(dr.hallucination_metrics)
        for dr in doc_results
        if dr.hallucination_metrics is not None
    ]
    if not results:
        return None
    return aggregate_hallucination_metrics(results)


@register_corpus_aggregator(
    name="calibration",
    attribute="aggregated_calibration",
    profiles=_STANDARD_PROFILES,
)
def _aggregate_calibration(doc_results: list) -> Optional[dict]:
    """Agrège la calibration micro sur tous les docs.

    Recalcule ECE/MCE à partir de la **somme des bins** de chaque
    document : pour chaque bin, on additionne ``count``, on agrège la
    confiance moyenne pondérée par count, et on agrège l'accuracy
    pondérée par count. L'ECE micro est ensuite la moyenne pondérée
    par bin de ``|conf - acc|``.

    Comportement déplacé verbatim depuis ``runner._aggregate_calibration``
    (chantier 2 — rétrocompat octet par octet du sérialisé).
    """
    relevant = [
        dr for dr in doc_results
        if dr.calibration_metrics is not None
        and (dr.calibration_metrics.get("bins") or [])
    ]
    if not relevant:
        return None

    n_bins = relevant[0].calibration_metrics.get("n_bins", 10)
    sum_conf: list[float] = [0.0] * n_bins
    sum_acc: list[float] = [0.0] * n_bins
    counts: list[int] = [0] * n_bins
    bin_lows: list[float] = [
        b["bin_low"] for b in relevant[0].calibration_metrics["bins"]
    ]
    bin_highs: list[float] = [
        b["bin_high"] for b in relevant[0].calibration_metrics["bins"]
    ]

    for dr in relevant:
        m = dr.calibration_metrics
        if m.get("n_bins") != n_bins:
            logger.warning(
                "[aggregate_calibration] %s : n_bins=%s ≠ %s — ignoré",
                dr.doc_id, m.get("n_bins"), n_bins,
            )
            continue
        for k, b in enumerate(m["bins"]):
            n = int(b.get("count") or 0)
            if n == 0:
                continue
            counts[k] += n
            sum_conf[k] += float(b.get("avg_confidence") or 0.0) * n
            sum_acc[k] += float(b.get("accuracy") or 0.0) * n

    total = sum(counts)
    if total == 0:
        return None

    bins: list[dict] = []
    ece = 0.0
    mce = 0.0
    for k in range(n_bins):
        n = counts[k]
        if n == 0:
            bins.append({
                "bin_low": bin_lows[k] if k < len(bin_lows) else k / n_bins,
                "bin_high": bin_highs[k] if k < len(bin_highs) else (k + 1) / n_bins,
                "avg_confidence": None,
                "accuracy": None,
                "count": 0,
                "gap": None,
            })
            continue
        avg_conf = sum_conf[k] / n
        accuracy = sum_acc[k] / n
        gap = abs(avg_conf - accuracy)
        bins.append({
            "bin_low": bin_lows[k] if k < len(bin_lows) else k / n_bins,
            "bin_high": bin_highs[k] if k < len(bin_highs) else (k + 1) / n_bins,
            "avg_confidence": avg_conf,
            "accuracy": accuracy,
            "count": n,
            "gap": gap,
        })
        ece += (n / total) * gap
        if gap > mce:
            mce = gap

    overall_acc = sum(sum_acc) / total
    overall_conf = sum(sum_conf) / total

    return {
        "ece": ece,
        "mce": mce,
        "n_bins": n_bins,
        "n_predictions": total,
        "overall_accuracy": overall_acc,
        "overall_confidence": overall_conf,
        "bins": bins,
        "doc_count": len(relevant),
    }


@register_corpus_aggregator(
    name="philological",
    attribute="aggregated_philological",
    profiles=_STANDARD_PROFILES,
)
def _aggregate_philological(doc_results: list) -> Optional[dict]:
    from picarones.measurements.philological_hooks import aggregate_philological_metrics
    return aggregate_philological_metrics(
        [dr.philological_metrics for dr in doc_results],
    )


@register_corpus_aggregator(
    name="searchability",
    attribute="aggregated_searchability",
    profiles=_STANDARD_PROFILES,
)
def _aggregate_searchability(doc_results: list) -> Optional[dict]:
    from picarones.measurements.searchability_hooks import aggregate_searchability_metrics
    return aggregate_searchability_metrics(
        [dr.searchability_metrics for dr in doc_results],
    )


@register_corpus_aggregator(
    name="numerical_sequences",
    attribute="aggregated_numerical_sequences",
    profiles=_STANDARD_PROFILES,
)
def _aggregate_numerical_sequences(doc_results: list) -> Optional[dict]:
    from picarones.measurements.numerical_sequences_hooks import (
        aggregate_numerical_sequence_metrics,
    )
    return aggregate_numerical_sequence_metrics(
        [dr.numerical_sequence_metrics for dr in doc_results],
    )


@register_corpus_aggregator(
    name="readability",
    attribute="aggregated_readability",
    profiles=_STANDARD_PROFILES,
)
def _aggregate_readability(doc_results: list) -> Optional[dict]:
    from picarones.measurements.readability_hooks import aggregate_readability_metrics
    return aggregate_readability_metrics(
        [dr.readability_metrics for dr in doc_results],
    )


__all__ = ["calibration_from_engine_result"]
