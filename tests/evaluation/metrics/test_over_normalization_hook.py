"""Phase 3.4 audit code-quality — la sur-normalisation LLM est
désormais agrégée automatiquement via le registre
:mod:`picarones.evaluation.metric_hooks`.

Avant la Phase 3.4, ``aggregate_over_normalization`` existait dans
``picarones/evaluation/metrics/over_normalization.py`` mais :

- n'avait aucun ``@register_corpus_aggregator`` ;
- le module n'était même pas importé par ``evaluation/metrics/__init__.py``
  (mentionné en docstring uniquement) ;
- ``synthetic.py`` réimplémentait l'agrégation manuellement
  (duplication silencieuse).

Le hook ``_aggregate_over_normalization_hook`` (auto-enregistré)
extrait désormais l'info depuis
``DocumentResult.pipeline_metadata["over_normalization"]`` et
alimente ``EngineReport.aggregated_over_normalization`` pour les
profils ``philological``, ``diagnostics`` et ``full``.
"""

from __future__ import annotations

from picarones.evaluation.benchmark_result import DocumentResult, EngineReport
from picarones.evaluation.metric_hooks import (
    PROFILE_DIAGNOSTICS,
    PROFILE_FULL,
    PROFILE_MINIMAL,
    PROFILE_PHILOLOGICAL,
    PROFILE_STANDARD,
    _all_corpus_aggregator_names,
    run_corpus_aggregators,
    select_corpus_aggregators,
)
from picarones.evaluation.metric_result import MetricsResult
from picarones.evaluation.metrics.over_normalization import (
    OverNormalizationResult,
    aggregate_over_normalization,
)


# --------------------------------------------------------------------------
# Auto-enregistrement
# --------------------------------------------------------------------------


def test_over_normalization_aggregator_is_registered() -> None:
    """L'import de ``picarones.evaluation.metrics`` doit déclencher
    l'enregistrement de l'agrégateur ``over_normalization``."""
    import picarones.evaluation.metrics  # noqa: F401 — déclenchement

    assert "over_normalization" in _all_corpus_aggregator_names(), (
        "Le hook ``_aggregate_over_normalization_hook`` n'est pas "
        "enregistré.  Vérifier que ``over_normalization`` est dans "
        "``picarones/evaluation/metrics/__init__.py`` (Phase 3.4)."
    )


def test_aggregator_in_correct_profiles() -> None:
    """L'agrégateur doit être actif pour ``philological``,
    ``diagnostics``, ``full`` — pas pour ``minimal`` ni ``standard``."""
    import picarones.evaluation.metrics  # noqa: F401

    for profile in (PROFILE_PHILOLOGICAL, PROFILE_DIAGNOSTICS, PROFILE_FULL):
        names = [a.name for a in select_corpus_aggregators(profile)]
        assert "over_normalization" in names, (
            f"Profil ``{profile}`` n'inclut pas l'agrégateur over_normalization."
        )

    for profile in (PROFILE_MINIMAL, PROFILE_STANDARD):
        names = [a.name for a in select_corpus_aggregators(profile)]
        assert "over_normalization" not in names, (
            f"Profil ``{profile}`` ne devrait pas inclure over_normalization."
        )


# --------------------------------------------------------------------------
# Fonction pure aggregate_over_normalization (rétrocompat)
# --------------------------------------------------------------------------


def test_pure_aggregate_empty_list_returns_zero() -> None:
    """Pas de docs → score None, compteurs à zéro (rétrocompat de la
    fonction utilitaire pure)."""
    out = aggregate_over_normalization([])
    assert out == {
        "score": None,
        "total_correct_ocr_words": 0,
        "over_normalized_count": 0,
    }


def test_pure_aggregate_sums_counts() -> None:
    """L'agrégation somme les compteurs bruts puis recalcule le score."""
    r1 = OverNormalizationResult(
        total_correct_ocr_words=100,
        over_normalized_count=10,
    )
    r2 = OverNormalizationResult(
        total_correct_ocr_words=50,
        over_normalized_count=5,
    )
    out = aggregate_over_normalization([r1, r2, None])  # None ignoré
    assert out == {
        "score": 0.1,  # 15 / 150
        "total_correct_ocr_words": 150,
        "over_normalized_count": 15,
        "document_count": 2,
    }


# --------------------------------------------------------------------------
# Hook décoré — extraction depuis DocumentResult.pipeline_metadata
# --------------------------------------------------------------------------


def _make_dr(
    doc_id: str,
    over_norm_dict: dict | None,
) -> DocumentResult:
    return DocumentResult(
        doc_id=doc_id,
        image_path=f"/tmp/{doc_id}.png",
        ground_truth="fait",
        hypothesis="fait",
        metrics=MetricsResult(cer=0.0, wer=0.0),
        duration_seconds=1.0,
        ocr_intermediate="faict",
        pipeline_metadata=(
            {"over_normalization": over_norm_dict}
            if over_norm_dict is not None
            else {}
        ),
    )


def test_hook_returns_none_when_no_pipeline_metadata() -> None:
    """Benchmark OCR seul (sans LLM) → aucun ``pipeline_metadata``,
    donc le hook retourne ``None`` et ``aggregated_over_normalization``
    reste à ``None``."""
    import picarones.evaluation.metrics  # noqa: F401

    docs = [_make_dr("d1", None), _make_dr("d2", None)]
    out = run_corpus_aggregators(PROFILE_FULL, docs)
    assert "aggregated_over_normalization" not in out


def test_hook_aggregates_from_pipeline_metadata() -> None:
    """Pipeline OCR+LLM → ``pipeline_metadata["over_normalization"]``
    est extrait et agrégé."""
    import picarones.evaluation.metrics  # noqa: F401

    docs = [
        _make_dr("d1", {
            "score": 0.1,
            "total_correct_ocr_words": 100,
            "over_normalized_count": 10,
            "over_normalized_passages": [],
        }),
        _make_dr("d2", {
            "score": 0.2,
            "total_correct_ocr_words": 50,
            "over_normalized_count": 10,
            "over_normalized_passages": [],
        }),
    ]
    out = run_corpus_aggregators(PROFILE_PHILOLOGICAL, docs)
    assert "aggregated_over_normalization" in out
    result = out["aggregated_over_normalization"]
    # 20 over-normalized / 150 correct OCR = 0.1333
    assert result["over_normalized_count"] == 20
    assert result["total_correct_ocr_words"] == 150
    assert result["document_count"] == 2
    assert 0.13 < result["score"] < 0.14


def test_hook_resilient_to_malformed_dict() -> None:
    """Si un document a un ``pipeline_metadata["over_normalization"]``
    mal formé (manque un champ, valeur non castable), il est skipé
    avec un warning — l'agrégateur n'échoue pas."""
    import picarones.evaluation.metrics  # noqa: F401

    docs = [
        _make_dr("d1", {"total_correct_ocr_words": 100, "over_normalized_count": 5}),
        _make_dr("d2", {"total_correct_ocr_words": "garbage", "over_normalized_count": 0}),
        _make_dr("d3", None),
    ]
    out = run_corpus_aggregators(PROFILE_FULL, docs)
    # d1 est valide → l'agrégateur retourne un dict, même si d2 est ignoré
    assert "aggregated_over_normalization" in out
    assert out["aggregated_over_normalization"]["over_normalized_count"] == 5


# --------------------------------------------------------------------------
# Sérialisation EngineReport
# --------------------------------------------------------------------------


def test_engine_report_round_trip_with_over_normalization() -> None:
    """Le champ ``aggregated_over_normalization`` est préservé par
    ``as_dict`` / ``from_dict``."""
    er = EngineReport(
        engine_name="tesseract+ministral",
        engine_version="5.3.0",
        engine_config={},
        document_results=[],
        aggregated_over_normalization={
            "score": 0.15,
            "total_correct_ocr_words": 200,
            "over_normalized_count": 30,
            "document_count": 5,
        },
    )
    d = er.as_dict()
    assert d["aggregated_over_normalization"]["score"] == 0.15

    rebuilt = EngineReport.from_dict(d)
    assert rebuilt.aggregated_over_normalization == er.aggregated_over_normalization
