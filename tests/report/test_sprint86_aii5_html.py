"""Tests Sprint 86 — A.II.5 bout-en-bout : helpers runner +
rendu HTML.

Couvre :

1. ``compute_searchability_metrics`` adaptive masking.
2. ``aggregate_searchability_metrics`` micro-recall.
3. ``compute_numerical_sequence_metrics_adaptive`` masking.
4. ``aggregate_numerical_sequence_metrics`` somme par catégorie.
5. Champs ``DocumentResult.searchability_metrics`` et
   ``EngineReport.aggregated_searchability``.
6. Rendu HTML adaptive + anti-injection.
7. Complétude i18n FR/EN.
"""

from __future__ import annotations

import json
from pathlib import Path

from picarones.evaluation.metrics.numerical_sequences_hooks import (
    aggregate_numerical_sequence_metrics,
    compute_numerical_sequence_metrics_adaptive,
)
from picarones.evaluation.metric_result import MetricsResult
from picarones.evaluation.benchmark_result import DocumentResult, EngineReport


def _stub_metrics() -> MetricsResult:
    return MetricsResult(
        cer=0.0, cer_nfc=0.0, cer_caseless=0.0,
        wer=0.0, wer_normalized=0.0, mer=0.0, wil=0.0,
        reference_length=0, hypothesis_length=0,
    )
from picarones.evaluation.metrics.searchability_hooks import (
    aggregate_searchability_metrics,
    compute_searchability_metrics,
)
from picarones.reports_v2.html.renderers.numerical_sequences import (
    build_numerical_sequences_html,
)
from picarones.reports_v2.html.renderers.searchability import (
    build_searchability_summary_html,
)


def _load_labels(lang: str) -> dict:
    p = (
        Path(__file__).parent.parent.parent
        / "picarones" / "reports_v2" / "i18n" / f"{lang}.json"
    )
    return json.loads(p.read_text(encoding="utf-8"))


# ──────────────────────────────────────────────────────────────────────────
# 1. Helpers searchability
# ──────────────────────────────────────────────────────────────────────────


class TestSearchabilityRunner:
    def test_empty_gt_returns_none(self) -> None:
        assert compute_searchability_metrics("", "anything") is None

    def test_normal(self) -> None:
        r = compute_searchability_metrics("le roi", "le roy")
        assert r is not None
        assert r["recall"] == 1.0
        assert r["n_gt_tokens"] == 2

    def test_aggregate_micro_recall(self) -> None:
        d1 = {"n_gt_tokens": 10, "n_searchable": 9, "missed_tokens": ["x"]}
        d2 = {"n_gt_tokens": 20, "n_searchable": 15, "missed_tokens": ["y"]}
        agg = aggregate_searchability_metrics([d1, d2])
        assert agg is not None
        assert agg["n_gt_tokens"] == 30
        assert agg["n_searchable"] == 24
        assert agg["recall"] == 24 / 30
        assert agg["n_docs"] == 2

    def test_aggregate_empty(self) -> None:
        assert aggregate_searchability_metrics([None, None]) is None
        assert aggregate_searchability_metrics([]) is None


# ──────────────────────────────────────────────────────────────────────────
# 2. Helpers numerical sequences
# ──────────────────────────────────────────────────────────────────────────


class TestNumericalSequencesRunner:
    def test_no_signal_returns_none(self) -> None:
        # GT sans aucune séquence numérique
        assert compute_numerical_sequence_metrics_adaptive(
            "lorem ipsum dolor", "sit amet",
        ) is None

    def test_signal_present(self) -> None:
        r = compute_numerical_sequence_metrics_adaptive(
            "an III, 1789", "an III, 1789",
        )
        assert r is not None
        assert r["n_total"] >= 1

    def test_aggregate_sums_per_category(self) -> None:
        d1 = {
            "n_total": 3,
            "global_strict_score": 1.0,
            "global_value_score": 1.0,
            "per_category": {
                "year": {"n_total": 2, "strict": 2, "value": 2,
                         "strict_score": 1.0, "value_score": 1.0,
                         "lost_items": []},
                "roman": {"n_total": 1, "strict": 1, "value": 1,
                          "strict_score": 1.0, "value_score": 1.0,
                          "lost_items": []},
                "foliation": {"n_total": 0, "strict": 0, "value": 0,
                              "strict_score": 0.0, "value_score": 0.0,
                              "lost_items": []},
                "currency": {"n_total": 0, "strict": 0, "value": 0,
                             "strict_score": 0.0, "value_score": 0.0,
                             "lost_items": []},
                "regnal": {"n_total": 0, "strict": 0, "value": 0,
                           "strict_score": 0.0, "value_score": 0.0,
                           "lost_items": []},
            },
        }
        d2 = {
            "n_total": 4,
            "global_strict_score": 0.5,
            "global_value_score": 0.5,
            "per_category": {
                "year": {"n_total": 4, "strict": 2, "value": 2,
                         "strict_score": 0.5, "value_score": 0.5,
                         "lost_items": ["1500", "1600"]},
                "roman": {"n_total": 0, "strict": 0, "value": 0,
                          "strict_score": 0.0, "value_score": 0.0,
                          "lost_items": []},
                "foliation": {"n_total": 0, "strict": 0, "value": 0,
                              "strict_score": 0.0, "value_score": 0.0,
                              "lost_items": []},
                "currency": {"n_total": 0, "strict": 0, "value": 0,
                             "strict_score": 0.0, "value_score": 0.0,
                             "lost_items": []},
                "regnal": {"n_total": 0, "strict": 0, "value": 0,
                           "strict_score": 0.0, "value_score": 0.0,
                           "lost_items": []},
            },
        }
        agg = aggregate_numerical_sequence_metrics([d1, d2])
        assert agg["n_total"] == 7
        assert agg["per_category"]["year"]["n_total"] == 6
        assert agg["per_category"]["year"]["strict"] == 4
        assert agg["per_category"]["year"]["strict_score"] == 4 / 6
        # global = (2+1 + 2) / 7 = 5/7
        assert agg["global_strict_score"] == 5 / 7

    def test_aggregate_empty(self) -> None:
        assert aggregate_numerical_sequence_metrics([None]) is None


# ──────────────────────────────────────────────────────────────────────────
# 3. Champs results.py
# ──────────────────────────────────────────────────────────────────────────


class TestResultsFields:
    def test_document_result_serializes_searchability(self) -> None:
        dr = DocumentResult(
            doc_id="doc1", image_path="x.png",
            ground_truth="hello", hypothesis="helo",
            metrics=_stub_metrics(), duration_seconds=1.0,
            searchability_metrics={"recall": 0.9},
            numerical_sequence_metrics={"n_total": 1},
        )
        d = dr.as_dict()
        assert d["searchability_metrics"] == {"recall": 0.9}
        assert d["numerical_sequence_metrics"] == {"n_total": 1}

    def test_document_result_omits_when_none(self) -> None:
        dr = DocumentResult(
            doc_id="doc1", image_path="x.png",
            ground_truth="hello", hypothesis="helo",
            metrics=_stub_metrics(), duration_seconds=1.0,
        )
        d = dr.as_dict()
        assert "searchability_metrics" not in d
        assert "numerical_sequence_metrics" not in d

    def test_compact_clears_fields(self) -> None:
        dr = DocumentResult(
            doc_id="doc1", image_path="x.png",
            ground_truth="hello", hypothesis="helo",
            metrics=_stub_metrics(), duration_seconds=1.0,
            searchability_metrics={"recall": 0.9},
            numerical_sequence_metrics={"n_total": 1},
        )
        # Sprint A14-S1 — opt-in via drop_analyses=True.
        dr.compact(drop_analyses=True)
        assert dr.searchability_metrics is None
        assert dr.numerical_sequence_metrics is None

    def test_engine_report_serializes_aggregates(self) -> None:
        er = EngineReport(
            engine_name="t", engine_version="0",
            engine_config={},
            document_results=[],
            pipeline_info=None,
            aggregated_searchability={"recall": 0.85},
            aggregated_numerical_sequences={"global_strict_score": 0.9},
        )
        d = er.as_dict()
        assert d["aggregated_searchability"]["recall"] == 0.85
        assert d["aggregated_numerical_sequences"]["global_strict_score"] == 0.9

    def test_engine_report_omits_when_none(self) -> None:
        er = EngineReport(
            engine_name="t", engine_version="0",
            engine_config={},
            document_results=[],
            pipeline_info=None,
        )
        d = er.as_dict()
        assert "aggregated_searchability" not in d
        assert "aggregated_numerical_sequences" not in d


# ──────────────────────────────────────────────────────────────────────────
# 4. Rendu HTML
# ──────────────────────────────────────────────────────────────────────────


class TestSearchabilityHtml:
    def test_empty_returns_empty(self) -> None:
        assert build_searchability_summary_html([]) == ""

    def test_no_signal_returns_empty(self) -> None:
        engines = [{"name": "t"}]  # pas de aggregated_searchability
        assert build_searchability_summary_html(engines) == ""

    def test_renders_table_with_recall(self) -> None:
        engines = [{
            "name": "tess",
            "aggregated_searchability": {
                "recall": 0.92, "n_searchable": 92,
                "n_gt_tokens": 100, "n_docs": 5,
            },
        }]
        html = build_searchability_summary_html(
            engines, _load_labels("fr"),
        )
        assert "<table" in html
        assert "92.0%" in html
        assert "92 / 100" in html
        assert "tess" in html

    def test_anti_injection(self) -> None:
        engines = [{
            "name": "<script>alert(1)</script>",
            "aggregated_searchability": {
                "recall": 0.5, "n_searchable": 5, "n_gt_tokens": 10,
                "n_docs": 1,
            },
        }]
        html = build_searchability_summary_html(
            engines, _load_labels("fr"),
        )
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html

    def test_renders_in_english(self) -> None:
        engines = [{
            "name": "tess",
            "aggregated_searchability": {
                "recall": 0.95, "n_searchable": 95,
                "n_gt_tokens": 100, "n_docs": 5,
            },
        }]
        html = build_searchability_summary_html(
            engines, _load_labels("en"),
        )
        assert "Fuzzy searchability" in html


class TestNumericalSequencesHtml:
    def _engine(self, name="tess", **kwargs) -> dict:
        per_cat_default = {
            cat: {"n_total": 0, "strict": 0, "value": 0,
                  "strict_score": 0.0, "value_score": 0.0,
                  "lost_items": []}
            for cat in ("year", "roman", "foliation", "currency", "regnal")
        }
        per_cat_default.update(kwargs.get("per_cat_overrides", {}))
        return {
            "name": name,
            "aggregated_numerical_sequences": {
                "global_strict_score": kwargs.get("strict", 0.5),
                "global_value_score": kwargs.get("value", 0.5),
                "n_total": kwargs.get("n_total", 1),
                "n_docs": 1,
                "per_category": per_cat_default,
            },
        }

    def test_empty_returns_empty(self) -> None:
        assert build_numerical_sequences_html([]) == ""

    def test_no_signal_returns_empty(self) -> None:
        engines = [{"name": "t"}]
        assert build_numerical_sequences_html(engines) == ""

    def test_omits_categories_without_signal(self) -> None:
        # Seul 'year' a du signal
        e = self._engine(per_cat_overrides={
            "year": {"n_total": 5, "strict": 5, "value": 5,
                     "strict_score": 1.0, "value_score": 1.0,
                     "lost_items": []},
        })
        html = build_numerical_sequences_html([e], _load_labels("fr"))
        assert "Année" in html
        # Romain absent puisqu'aucun n_total > 0
        assert "Romain" not in html

    def test_renders_per_category_score(self) -> None:
        e = self._engine(strict=0.8, value=0.9, n_total=20,
                         per_cat_overrides={
            "year": {"n_total": 10, "strict": 8, "value": 9,
                     "strict_score": 0.8, "value_score": 0.9,
                     "lost_items": []},
        })
        html = build_numerical_sequences_html([e], _load_labels("fr"))
        assert "80%" in html  # year strict score
        assert "n=20" in html or "n=10" in html

    def test_anti_injection(self) -> None:
        e = self._engine(name="<img/>", per_cat_overrides={
            "year": {"n_total": 1, "strict": 1, "value": 1,
                     "strict_score": 1.0, "value_score": 1.0,
                     "lost_items": []},
        })
        html = build_numerical_sequences_html([e], _load_labels("fr"))
        assert "<img/>" not in html
        assert "&lt;img" in html


# ──────────────────────────────────────────────────────────────────────────
# 5. Complétude i18n
# ──────────────────────────────────────────────────────────────────────────


_KEYS = {
    "search_title", "search_note", "search_engine", "search_recall",
    "search_count", "search_docs",
    "numseq_title", "numseq_note", "numseq_engine", "numseq_global",
    "numseq_cat_year", "numseq_cat_roman", "numseq_cat_foliation",
    "numseq_cat_currency", "numseq_cat_regnal",
}


class TestI18nCompleteness:
    def test_fr_has_all(self) -> None:
        d = _load_labels("fr")
        missing = _KEYS - d.keys()
        assert not missing, f"manque FR : {missing}"

    def test_en_has_all(self) -> None:
        d = _load_labels("en")
        missing = _KEYS - d.keys()
        assert not missing, f"manque EN : {missing}"
