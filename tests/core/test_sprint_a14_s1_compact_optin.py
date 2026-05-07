"""Sprint A14-S1 — A.I.0 P0 : ``DocumentResult.compact()`` est opt-in.

Avant ce sprint, le runner appelait ``dr.compact()`` sans argument
avant de sérialiser le JSON, ce qui :

- tronquait ``ground_truth``, ``hypothesis`` et ``ocr_intermediate``
  à 200 caractères ;
- effaçait 13 dicts d'analyse per-document (confusion, taxonomy,
  philological, searchability, etc.).

Le rapport HTML — qui consomme ce JSON — recevait des données déjà
mutilées, contredisant la promesse "self-contained HTML report" du
README.

Désormais, ``compact()`` est no-op par défaut.  Le caller doit
explicitement demander la troncature via ``text_limit`` et/ou la
suppression des analyses via ``drop_analyses=True``.
"""

from __future__ import annotations

from picarones.evaluation.metric_result import MetricsResult
from picarones.evaluation.benchmark_result import DocumentResult


def _make_dr(**kwargs) -> DocumentResult:
    base = dict(
        doc_id="d1",
        image_path="x.png",
        ground_truth="A" * 1000,
        hypothesis="B" * 1000,
        metrics=MetricsResult(cer=0.1, wer=0.1, error=None),
        duration_seconds=0.1,
        confusion_matrix={"k": "v"},
        char_scores={"ligature": {"score": 0.9}},
        taxonomy={"class": "v"},
        structure={"k": "v"},
        image_quality={"k": "v"},
        line_metrics={"k": "v"},
        hallucination_metrics={"k": "v"},
        ner_metrics={"k": "v"},
        calibration_metrics={"k": "v"},
        philological_metrics={"k": "v"},
        searchability_metrics={"k": "v"},
        numerical_sequence_metrics={"k": "v"},
        readability_metrics={"k": "v"},
        ocr_intermediate="C" * 1000,
    )
    base.update(kwargs)
    return DocumentResult(**base)


class TestCompactDefaultIsNoOp:
    def test_default_call_does_not_truncate_text(self) -> None:
        dr = _make_dr()
        before_gt = dr.ground_truth
        before_hyp = dr.hypothesis
        before_ocr = dr.ocr_intermediate
        dr.compact()
        assert dr.ground_truth == before_gt
        assert dr.hypothesis == before_hyp
        assert dr.ocr_intermediate == before_ocr

    def test_default_call_preserves_all_analyses(self) -> None:
        dr = _make_dr()
        dr.compact()
        for field in (
            "confusion_matrix", "char_scores", "taxonomy", "structure",
            "image_quality", "line_metrics", "hallucination_metrics",
            "ner_metrics", "calibration_metrics", "philological_metrics",
            "searchability_metrics", "numerical_sequence_metrics",
            "readability_metrics",
        ):
            assert getattr(dr, field) is not None, (
                f"{field} a été effacé alors que ``compact()`` est "
                "censé être no-op par défaut depuis Sprint A14-S1."
            )


class TestCompactTextLimit:
    def test_text_limit_truncates_ground_truth(self) -> None:
        dr = _make_dr()
        dr.compact(text_limit=200)
        assert len(dr.ground_truth) == 201  # 200 + ellipsis

    def test_text_limit_truncates_hypothesis(self) -> None:
        dr = _make_dr()
        dr.compact(text_limit=50)
        assert len(dr.hypothesis) == 51

    def test_text_limit_truncates_ocr_intermediate(self) -> None:
        dr = _make_dr()
        dr.compact(text_limit=100)
        assert len(dr.ocr_intermediate) == 101

    def test_text_limit_zero_or_none_is_noop(self) -> None:
        dr = _make_dr()
        dr.compact(text_limit=0)
        assert len(dr.ground_truth) == 1000
        dr2 = _make_dr()
        dr2.compact(text_limit=None)
        assert len(dr2.ground_truth) == 1000

    def test_text_limit_does_not_truncate_short_text(self) -> None:
        dr = _make_dr(ground_truth="short", hypothesis="also short")
        dr.compact(text_limit=200)
        assert dr.ground_truth == "short"
        assert dr.hypothesis == "also short"


class TestCompactDropAnalyses:
    def test_drop_analyses_clears_all_thirteen_fields(self) -> None:
        dr = _make_dr()
        dr.compact(drop_analyses=True)
        for field in (
            "confusion_matrix", "char_scores", "taxonomy", "structure",
            "image_quality", "line_metrics", "hallucination_metrics",
            "ner_metrics", "calibration_metrics", "philological_metrics",
            "searchability_metrics", "numerical_sequence_metrics",
            "readability_metrics",
        ):
            assert getattr(dr, field) is None, f"{field} aurait dû être effacé"

    def test_drop_analyses_alone_preserves_text(self) -> None:
        dr = _make_dr()
        dr.compact(drop_analyses=True)  # pas de text_limit
        assert len(dr.ground_truth) == 1000
        assert len(dr.hypothesis) == 1000

    def test_combined_legacy_behavior(self) -> None:
        """``compact(text_limit=200, drop_analyses=True)`` reproduit
        l'ancien comportement par défaut (avant Sprint A14-S1)."""
        dr = _make_dr()
        dr.compact(text_limit=200, drop_analyses=True)
        assert len(dr.ground_truth) == 201
        assert dr.confusion_matrix is None
        assert dr.philological_metrics is None
