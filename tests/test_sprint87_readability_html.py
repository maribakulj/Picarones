"""Tests Sprint 87 — A.II.2 (delta Flesch) bout-en-bout : helper
runner + rendu HTML.

Couvre :

1. ``compute_readability_metrics`` adaptive masking.
2. ``aggregate_readability_metrics`` moyenne + over-norm rate.
3. Champs ``DocumentResult.readability_metrics`` et
   ``EngineReport.aggregated_readability``.
4. Rendu HTML adaptive + anti-injection.
5. Complétude i18n FR/EN.
"""

from __future__ import annotations

import json
from pathlib import Path

from picarones.measurements.metrics import MetricsResult
from picarones.measurements.readability_runner import (
    aggregate_readability_metrics,
    compute_readability_metrics,
)
from picarones.core.results import DocumentResult, EngineReport
from picarones.report.readability_render import (
    build_readability_summary_html,
)


def _stub_metrics() -> MetricsResult:
    return MetricsResult(
        cer=0.0, cer_nfc=0.0, cer_caseless=0.0,
        wer=0.0, wer_normalized=0.0, mer=0.0, wil=0.0,
        reference_length=0, hypothesis_length=0,
    )


def _load_labels(lang: str) -> dict:
    p = (
        Path(__file__).parent.parent
        / "picarones" / "report" / "i18n" / f"{lang}.json"
    )
    return json.loads(p.read_text(encoding="utf-8"))


# ──────────────────────────────────────────────────────────────────────────
# 1. compute_readability_metrics
# ──────────────────────────────────────────────────────────────────────────


class TestComputeReadability:
    def test_short_gt_returns_none(self) -> None:
        # < 5 mots → masquage adaptatif
        assert compute_readability_metrics("a b c", "a b c d") is None

    def test_normal(self) -> None:
        gt = "Le roi a signé la charte ce matin avec ses ministres."
        hyp = "Le roi a signé la charte ce matin avec ses ministres."
        r = compute_readability_metrics(gt, hyp)
        assert r is not None
        assert r["lang"] == "fr"
        assert r["flesch_delta"] == 0.0
        assert r["n_words_reference"] >= 10

    def test_returns_lang(self) -> None:
        gt = "The king signed the charter today with his ministers."
        r = compute_readability_metrics(gt, gt, lang="en")
        assert r is not None
        assert r["lang"] == "en"

    def test_empty_hypothesis(self) -> None:
        # GT exploitable, hypothèse vide → None pour delta
        gt = "Le roi a signé la charte ce matin avec ses ministres."
        r = compute_readability_metrics(gt, "")
        assert r is not None
        assert r["flesch_delta"] is None
        assert r["flesch_hypothesis"] is None


# ──────────────────────────────────────────────────────────────────────────
# 2. aggregate_readability_metrics
# ──────────────────────────────────────────────────────────────────────────


class TestAggregate:
    def test_averages_deltas(self) -> None:
        per_doc = [
            {"flesch_delta": 5.0, "lang": "fr"},
            {"flesch_delta": 10.0, "lang": "fr"},
            {"flesch_delta": -2.0, "lang": "fr"},
        ]
        agg = aggregate_readability_metrics(per_doc)
        assert agg["delta_mean"] == (5.0 + 10.0 - 2.0) / 3
        # 1 over-normalisé (+10), 0 under (-5)
        assert agg["n_over_normalized"] == 1
        assert agg["n_under_normalized"] == 0
        assert agg["n_docs"] == 3

    def test_over_norm_rate(self) -> None:
        per_doc = [{"flesch_delta": d, "lang": "fr"} for d in
                   [10.0, 8.0, 6.0, 0.0, -1.0]]
        agg = aggregate_readability_metrics(per_doc)
        assert agg["n_over_normalized"] == 3
        assert agg["over_normalized_rate"] == 3 / 5

    def test_empty(self) -> None:
        assert aggregate_readability_metrics([]) is None
        assert aggregate_readability_metrics([None]) is None

    def test_no_delta_returns_none(self) -> None:
        # Tous les deltas None → None
        per_doc = [{"flesch_delta": None, "lang": "fr"}]
        assert aggregate_readability_metrics(per_doc) is None


# ──────────────────────────────────────────────────────────────────────────
# 3. Champs results.py
# ──────────────────────────────────────────────────────────────────────────


class TestResultsFields:
    def test_document_result_serializes(self) -> None:
        dr = DocumentResult(
            doc_id="d1", image_path="x.png",
            ground_truth="Le roi a signé la charte ce matin.",
            hypothesis="Le roi signa la charte.",
            metrics=_stub_metrics(), duration_seconds=1.0,
            readability_metrics={"flesch_delta": 5.0, "lang": "fr"},
        )
        d = dr.as_dict()
        assert d["readability_metrics"]["flesch_delta"] == 5.0

    def test_omits_when_none(self) -> None:
        dr = DocumentResult(
            doc_id="d1", image_path="x.png",
            ground_truth="x", hypothesis="x",
            metrics=_stub_metrics(), duration_seconds=1.0,
        )
        d = dr.as_dict()
        assert "readability_metrics" not in d

    def test_compact_clears(self) -> None:
        dr = DocumentResult(
            doc_id="d1", image_path="x.png",
            ground_truth="x", hypothesis="x",
            metrics=_stub_metrics(), duration_seconds=1.0,
            readability_metrics={"flesch_delta": 5.0},
        )
        dr.compact()
        assert dr.readability_metrics is None

    def test_engine_report_serializes(self) -> None:
        er = EngineReport(
            engine_name="t", engine_version="0",
            engine_config={},
            document_results=[],
            pipeline_info=None,
            aggregated_readability={"delta_mean": 3.5, "lang": "fr"},
        )
        d = er.as_dict()
        assert d["aggregated_readability"]["delta_mean"] == 3.5

    def test_engine_report_omits_when_none(self) -> None:
        er = EngineReport(
            engine_name="t", engine_version="0",
            engine_config={},
            document_results=[],
            pipeline_info=None,
        )
        d = er.as_dict()
        assert "aggregated_readability" not in d


# ──────────────────────────────────────────────────────────────────────────
# 4. Rendu HTML
# ──────────────────────────────────────────────────────────────────────────


class TestRender:
    def test_empty_returns_empty(self) -> None:
        assert build_readability_summary_html([]) == ""

    def test_no_signal_returns_empty(self) -> None:
        engines = [{"name": "t"}]
        assert build_readability_summary_html(engines) == ""

    def test_renders_table(self) -> None:
        engines = [{
            "name": "tess",
            "aggregated_readability": {
                "lang": "fr", "delta_mean": 11.5, "delta_median": 12.0,
                "delta_min": 5.0, "delta_max": 20.0,
                "over_normalized_rate": 0.85,
                "n_over_normalized": 17, "n_under_normalized": 0,
                "n_docs": 20,
            },
        }]
        html = build_readability_summary_html(
            engines, _load_labels("fr"),
        )
        assert "<table" in html
        # Δ > 5 → over-norm
        assert "+11.50" in html
        assert "85%" in html

    def test_anti_injection(self) -> None:
        engines = [{
            "name": "<script>alert(1)</script>",
            "aggregated_readability": {
                "lang": "fr", "delta_mean": 0.0, "delta_median": 0.0,
                "over_normalized_rate": 0.0,
                "n_over_normalized": 0, "n_under_normalized": 0,
                "n_docs": 1,
            },
        }]
        html = build_readability_summary_html(
            engines, _load_labels("fr"),
        )
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html

    def test_renders_in_english(self) -> None:
        engines = [{
            "name": "tess",
            "aggregated_readability": {
                "lang": "en", "delta_mean": 0.0, "delta_median": 0.0,
                "over_normalized_rate": 0.0,
                "n_over_normalized": 0, "n_under_normalized": 0,
                "n_docs": 1,
            },
        }]
        html = build_readability_summary_html(
            engines, _load_labels("en"),
        )
        assert "Readability" in html


# ──────────────────────────────────────────────────────────────────────────
# 5. Complétude i18n
# ──────────────────────────────────────────────────────────────────────────


_KEYS = {
    "readability_title", "readability_note", "readability_engine",
    "readability_delta_mean", "readability_delta_median",
    "readability_over_norm_rate", "readability_under_norm_count",
    "readability_docs",
}


class TestI18n:
    def test_fr(self) -> None:
        d = _load_labels("fr")
        assert not _KEYS - d.keys()

    def test_en(self) -> None:
        d = _load_labels("en")
        assert not _KEYS - d.keys()
