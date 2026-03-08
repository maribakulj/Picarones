"""Tests Sprint 10 — Distribution des erreurs par ligne et détection des hallucinations VLM.

Classes de tests
----------------
TestLineMetrics          (12 tests) — compute_line_metrics + aggregate_line_metrics
TestHallucinationMetrics (12 tests) — compute_hallucination_metrics + aggregate_hallucination_metrics
TestLineMetricsInResults  (4 tests) — intégration dans DocumentResult / EngineReport
TestFixturesVLM           (6 tests) — moteur VLM fictif et génération de données
TestReportSprint10        (6 tests) — rapport HTML contient les nouvelles métriques
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers communs
# ---------------------------------------------------------------------------

GT_SIMPLE = "Le renard brun saute par-dessus le chien paresseux."
HYP_PERFECT = "Le renard brun saute par-dessus le chien paresseux."
HYP_ERRORS = "Le renrd brin soute par-desous le chen paressux."
HYP_MISSING = "Le renard brun saute."

GT_MULTILINE = "Icy commence le prologue\nde maiſtre Jehan Froiſſart\nſus les croniques de France."
HYP_MULTILINE_PERFECT = "Icy commence le prologue\nde maiſtre Jehan Froiſſart\nſus les croniques de France."
HYP_MULTILINE_ERRORS = "Icy commence le prologue\nde maistre Jehan Froissart\nsus les croniques de France."

GT_MEDIEVAL = "Icy commence le prologue de maiſtre Jehan Froiſſart ſus les croniques de France & d'Angleterre."
HYP_HALLUCINATED = (
    "Icy commence le prologue de maistre Jehan Froissart sus les croniques de France et d'Angleterre. "
    "Ledit document fut enregistré au greffe le lendemain. "
    "Signé et paraphé par le notaire royal en présence de témoins. "
    "Archives nationales, cote F/7/1234, pièce n° 42."
)


# ===========================================================================
# TestLineMetrics
# ===========================================================================

class TestLineMetrics:
    """Tests pour picarones.core.line_metrics.compute_line_metrics."""

    def test_import(self):
        from picarones.core.line_metrics import compute_line_metrics, LineMetrics
        assert callable(compute_line_metrics)
        assert LineMetrics is not None

    def test_perfect_match_cer_zero(self):
        from picarones.core.line_metrics import compute_line_metrics
        result = compute_line_metrics(GT_MULTILINE, HYP_MULTILINE_PERFECT)
        assert result.mean_cer == pytest.approx(0.0, abs=1e-9)
        assert all(v == pytest.approx(0.0, abs=1e-9) for v in result.cer_per_line)

    def test_line_count(self):
        from picarones.core.line_metrics import compute_line_metrics
        result = compute_line_metrics(GT_MULTILINE, HYP_MULTILINE_ERRORS)
        assert result.line_count == 3

    def test_cer_per_line_length(self):
        from picarones.core.line_metrics import compute_line_metrics
        result = compute_line_metrics(GT_MULTILINE, HYP_MULTILINE_ERRORS)
        assert len(result.cer_per_line) == 3

    def test_percentiles_keys(self):
        from picarones.core.line_metrics import compute_line_metrics
        result = compute_line_metrics(GT_MULTILINE, HYP_MULTILINE_ERRORS)
        for key in ("p50", "p75", "p90", "p95", "p99"):
            assert key in result.percentiles
            assert 0.0 <= result.percentiles[key] <= 1.0

    def test_percentile_ordering(self):
        """p50 ≤ p75 ≤ p90 ≤ p95 ≤ p99."""
        from picarones.core.line_metrics import compute_line_metrics
        result = compute_line_metrics(GT_MULTILINE, HYP_MULTILINE_ERRORS)
        p = result.percentiles
        assert p["p50"] <= p["p75"] <= p["p90"] <= p["p95"] <= p["p99"]

    def test_gini_zero_for_perfect(self):
        from picarones.core.line_metrics import compute_line_metrics
        result = compute_line_metrics(GT_MULTILINE, HYP_MULTILINE_PERFECT)
        assert result.gini == pytest.approx(0.0, abs=1e-9)

    def test_gini_range(self):
        from picarones.core.line_metrics import compute_line_metrics
        result = compute_line_metrics(GT_MULTILINE, HYP_MULTILINE_ERRORS)
        assert 0.0 <= result.gini <= 1.0

    def test_catastrophic_rate_keys(self):
        from picarones.core.line_metrics import compute_line_metrics
        result = compute_line_metrics(GT_MULTILINE, HYP_MULTILINE_ERRORS,
                                      thresholds=[0.30, 0.50, 1.00])
        for t in (0.30, 0.50, 1.00):
            assert t in result.catastrophic_rate
            assert 0.0 <= result.catastrophic_rate[t] <= 1.0

    def test_heatmap_length(self):
        from picarones.core.line_metrics import compute_line_metrics
        result = compute_line_metrics(GT_MULTILINE, HYP_MULTILINE_ERRORS, heatmap_bins=5)
        assert len(result.heatmap) == 5

    def test_as_dict_and_from_dict_roundtrip(self):
        from picarones.core.line_metrics import compute_line_metrics, LineMetrics
        result = compute_line_metrics(GT_MULTILINE, HYP_MULTILINE_ERRORS)
        d = result.as_dict()
        restored = LineMetrics.from_dict(d)
        assert restored.gini == pytest.approx(result.gini, abs=1e-5)
        assert restored.line_count == result.line_count
        assert len(restored.cer_per_line) == len(result.cer_per_line)

    def test_aggregate_line_metrics(self):
        from picarones.core.line_metrics import compute_line_metrics, aggregate_line_metrics, LineMetrics
        r1 = compute_line_metrics(GT_MULTILINE, HYP_MULTILINE_PERFECT)
        r2 = compute_line_metrics(GT_MULTILINE, HYP_MULTILINE_ERRORS)
        agg = aggregate_line_metrics([r1, r2])
        assert "gini_mean" in agg
        assert "percentiles" in agg
        assert "catastrophic_rate" in agg
        assert "document_count" in agg
        assert agg["document_count"] == 2
        assert agg["gini_mean"] >= 0.0


# ===========================================================================
# TestHallucinationMetrics
# ===========================================================================

class TestHallucinationMetrics:
    """Tests pour picarones.core.hallucination.compute_hallucination_metrics."""

    def test_import(self):
        from picarones.core.hallucination import compute_hallucination_metrics, HallucinationMetrics
        assert callable(compute_hallucination_metrics)
        assert HallucinationMetrics is not None

    def test_perfect_match_anchor_one(self):
        from picarones.core.hallucination import compute_hallucination_metrics
        result = compute_hallucination_metrics(GT_SIMPLE, HYP_PERFECT)
        # Ancrage parfait → score proche de 1.0
        assert result.anchor_score == pytest.approx(1.0, abs=0.05)
        assert result.is_hallucinating is False

    def test_length_ratio_perfect(self):
        from picarones.core.hallucination import compute_hallucination_metrics
        result = compute_hallucination_metrics(GT_SIMPLE, HYP_PERFECT)
        assert result.length_ratio == pytest.approx(1.0, abs=0.05)

    def test_hallucination_detected(self):
        from picarones.core.hallucination import compute_hallucination_metrics
        result = compute_hallucination_metrics(GT_MEDIEVAL, HYP_HALLUCINATED)
        # L'hypothèse est beaucoup plus longue
        assert result.length_ratio > 1.0
        assert result.is_hallucinating is True

    def test_hallucinated_blocks_detected(self):
        from picarones.core.hallucination import compute_hallucination_metrics
        result = compute_hallucination_metrics(GT_MEDIEVAL, HYP_HALLUCINATED,
                                               anchor_threshold=0.5, min_block_length=3)
        # Des blocs hallucinés doivent être détectés
        assert len(result.hallucinated_blocks) > 0

    def test_net_insertion_rate_range(self):
        from picarones.core.hallucination import compute_hallucination_metrics
        result = compute_hallucination_metrics(GT_MEDIEVAL, HYP_HALLUCINATED)
        assert 0.0 <= result.net_insertion_rate <= 1.0

    def test_word_counts(self):
        from picarones.core.hallucination import compute_hallucination_metrics
        result = compute_hallucination_metrics(GT_SIMPLE, HYP_PERFECT)
        assert result.gt_word_count > 0
        assert result.hyp_word_count > 0

    def test_empty_reference(self):
        from picarones.core.hallucination import compute_hallucination_metrics
        result = compute_hallucination_metrics("", "some text here added by model")
        # Référence vide : insertion nette maximale
        assert result.net_insertion_rate == pytest.approx(1.0, abs=0.05)

    def test_empty_hypothesis(self):
        from picarones.core.hallucination import compute_hallucination_metrics
        result = compute_hallucination_metrics(GT_SIMPLE, "")
        assert result.hyp_word_count == 0
        assert result.net_insertion_rate == pytest.approx(0.0)

    def test_as_dict_and_from_dict_roundtrip(self):
        from picarones.core.hallucination import compute_hallucination_metrics, HallucinationMetrics
        result = compute_hallucination_metrics(GT_MEDIEVAL, HYP_HALLUCINATED)
        d = result.as_dict()
        restored = HallucinationMetrics.from_dict(d)
        assert restored.anchor_score == pytest.approx(result.anchor_score, abs=1e-5)
        assert restored.is_hallucinating == result.is_hallucinating
        assert len(restored.hallucinated_blocks) == len(result.hallucinated_blocks)

    def test_aggregate_hallucination_metrics(self):
        from picarones.core.hallucination import compute_hallucination_metrics, aggregate_hallucination_metrics
        r1 = compute_hallucination_metrics(GT_SIMPLE, HYP_PERFECT)
        r2 = compute_hallucination_metrics(GT_MEDIEVAL, HYP_HALLUCINATED)
        agg = aggregate_hallucination_metrics([r1, r2])
        assert "anchor_score_mean" in agg
        assert "length_ratio_mean" in agg
        assert "hallucinating_doc_count" in agg
        assert "document_count" in agg
        assert agg["document_count"] == 2
        assert agg["hallucinating_doc_count"] >= 1

    def test_anchor_threshold_respected(self):
        """Un ancrage très bas déclenche le badge hallucination."""
        from picarones.core.hallucination import compute_hallucination_metrics
        result = compute_hallucination_metrics(
            "abc def ghi", "xyz uvw rst opq lmn",
            anchor_threshold=0.5
        )
        assert result.anchor_score < 0.5
        assert result.is_hallucinating is True


# ===========================================================================
# TestLineMetricsInResults
# ===========================================================================

class TestLineMetricsInResults:
    """Tests pour l'intégration des métriques Sprint 10 dans les modèles de données."""

    def test_document_result_has_line_metrics_field(self):
        from picarones.core.results import DocumentResult
        from picarones.core.metrics import MetricsResult
        dr = DocumentResult(
            doc_id="test_001",
            image_path="/test/img.jpg",
            ground_truth=GT_SIMPLE,
            hypothesis=HYP_ERRORS,
            metrics=MetricsResult(
                cer=0.1, cer_nfc=0.1, cer_caseless=0.09,
                wer=0.2, wer_normalized=0.2,
                mer=0.15, wil=0.18,
                reference_length=50, hypothesis_length=48,
            ),
            duration_seconds=1.0,
            line_metrics={"gini": 0.3, "line_count": 3},
        )
        assert dr.line_metrics is not None
        assert dr.line_metrics["gini"] == pytest.approx(0.3)

    def test_document_result_has_hallucination_metrics_field(self):
        from picarones.core.results import DocumentResult
        from picarones.core.metrics import MetricsResult
        dr = DocumentResult(
            doc_id="test_002",
            image_path="/test/img.jpg",
            ground_truth=GT_SIMPLE,
            hypothesis=HYP_HALLUCINATED,
            metrics=MetricsResult(
                cer=0.5, cer_nfc=0.5, cer_caseless=0.5,
                wer=0.6, wer_normalized=0.6,
                mer=0.55, wil=0.65,
                reference_length=50, hypothesis_length=100,
            ),
            duration_seconds=2.0,
            hallucination_metrics={"anchor_score": 0.3, "is_hallucinating": True},
        )
        assert dr.hallucination_metrics is not None
        assert dr.hallucination_metrics["is_hallucinating"] is True

    def test_document_result_as_dict_includes_sprint10_fields(self):
        from picarones.core.results import DocumentResult
        from picarones.core.metrics import MetricsResult
        dr = DocumentResult(
            doc_id="test_003",
            image_path="/test/img.jpg",
            ground_truth=GT_SIMPLE,
            hypothesis=HYP_PERFECT,
            metrics=MetricsResult(
                cer=0.0, cer_nfc=0.0, cer_caseless=0.0,
                wer=0.0, wer_normalized=0.0,
                mer=0.0, wil=0.0,
                reference_length=50, hypothesis_length=50,
            ),
            duration_seconds=0.5,
            line_metrics={"gini": 0.0, "line_count": 1},
            hallucination_metrics={"anchor_score": 1.0, "is_hallucinating": False},
        )
        d = dr.as_dict()
        assert "line_metrics" in d
        assert "hallucination_metrics" in d

    def test_engine_report_has_aggregated_sprint10_fields(self):
        from picarones.core.results import EngineReport, DocumentResult
        from picarones.core.metrics import MetricsResult
        dr = DocumentResult(
            doc_id="test_004",
            image_path="/test/img.jpg",
            ground_truth=GT_SIMPLE,
            hypothesis=HYP_PERFECT,
            metrics=MetricsResult(
                cer=0.0, cer_nfc=0.0, cer_caseless=0.0,
                wer=0.0, wer_normalized=0.0,
                mer=0.0, wil=0.0,
                reference_length=50, hypothesis_length=50,
            ),
            duration_seconds=0.5,
        )
        report = EngineReport(
            engine_name="test_engine",
            engine_version="1.0",
            engine_config={},
            document_results=[dr],
            aggregated_line_metrics={"gini_mean": 0.1, "document_count": 1},
            aggregated_hallucination={"anchor_score_mean": 0.95, "document_count": 1},
        )
        assert report.aggregated_line_metrics is not None
        assert report.aggregated_hallucination is not None
        d = report.as_dict()
        assert "aggregated_line_metrics" in d
        assert "aggregated_hallucination" in d


# ===========================================================================
# TestFixturesVLM
# ===========================================================================

class TestFixturesVLM:
    """Tests pour le moteur VLM fictif dans picarones.fixtures."""

    def test_generate_sample_benchmark_has_vlm_engine(self):
        from picarones.fixtures import generate_sample_benchmark
        bm = generate_sample_benchmark(n_docs=3, seed=42)
        engine_names = [r.engine_name for r in bm.engine_reports]
        assert any("vision" in name.lower() or "vlm" in name.lower() or "zero-shot" in name.lower()
                   for name in engine_names)

    def test_vlm_engine_has_hallucination_metrics(self):
        from picarones.fixtures import generate_sample_benchmark
        bm = generate_sample_benchmark(n_docs=3, seed=42)
        vlm_report = next(
            (r for r in bm.engine_reports
             if r.pipeline_info.get("is_vlm")),
            None
        )
        assert vlm_report is not None, "Moteur VLM non trouvé"
        assert vlm_report.aggregated_hallucination is not None
        assert "anchor_score_mean" in vlm_report.aggregated_hallucination

    def test_all_engines_have_line_metrics(self):
        from picarones.fixtures import generate_sample_benchmark
        bm = generate_sample_benchmark(n_docs=3, seed=42)
        for report in bm.engine_reports:
            assert report.aggregated_line_metrics is not None, \
                f"Pas de line_metrics pour {report.engine_name}"
            assert "gini_mean" in report.aggregated_line_metrics

    def test_all_documents_have_line_metrics(self):
        from picarones.fixtures import generate_sample_benchmark
        bm = generate_sample_benchmark(n_docs=3, seed=42)
        for report in bm.engine_reports:
            for dr in report.document_results:
                assert dr.line_metrics is not None, \
                    f"{report.engine_name}/{dr.doc_id}: line_metrics manquant"
                assert "gini" in dr.line_metrics

    def test_all_documents_have_hallucination_metrics(self):
        from picarones.fixtures import generate_sample_benchmark
        bm = generate_sample_benchmark(n_docs=3, seed=42)
        for report in bm.engine_reports:
            for dr in report.document_results:
                assert dr.hallucination_metrics is not None, \
                    f"{report.engine_name}/{dr.doc_id}: hallucination_metrics manquant"
                assert "anchor_score" in dr.hallucination_metrics

    def test_vlm_engine_has_valid_hallucination_aggregation(self):
        """Le moteur VLM doit avoir des métriques d'hallucination agrégées valides."""
        from picarones.fixtures import generate_sample_benchmark
        bm = generate_sample_benchmark(n_docs=6, seed=42)
        vlm_report = next(
            (r for r in bm.engine_reports if r.pipeline_info.get("is_vlm")),
            None
        )
        if vlm_report is None:
            pytest.skip("Moteur VLM non trouvé")

        agg = vlm_report.aggregated_hallucination
        assert agg is not None
        assert 0.0 <= agg.get("anchor_score_mean", -1) <= 1.0
        assert agg.get("length_ratio_mean", 0) >= 0.0
        assert agg.get("document_count", 0) == 6


# ===========================================================================
# TestReportSprint10
# ===========================================================================

class TestReportSprint10:
    """Tests pour le rapport HTML — nouvelles métriques Sprint 10."""

    @pytest.fixture(scope="class")
    def html_report(self, tmp_path_factory):
        """Génère un rapport HTML de démonstration."""
        from picarones.fixtures import generate_sample_benchmark
        from picarones.report.generator import ReportGenerator
        bm = generate_sample_benchmark(n_docs=3, seed=42)
        tmp = tmp_path_factory.mktemp("report")
        out = tmp / "sprint10_test.html"
        ReportGenerator(bm).generate(str(out))
        return out.read_text(encoding="utf-8")

    def test_report_generated_not_empty(self, html_report):
        assert len(html_report) > 50_000

    def test_report_has_gini_column_header(self, html_report):
        assert "Gini" in html_report

    def test_report_has_ancrage_column_header(self, html_report):
        assert "Ancrage" in html_report

    def test_report_has_gini_cer_scatter_canvas(self, html_report):
        assert "chart-gini-cer" in html_report

    def test_report_has_ratio_anchor_scatter_canvas(self, html_report):
        assert "chart-ratio-anchor" in html_report

    def test_report_has_vlm_badge(self, html_report):
        """Le badge VLM doit apparaître pour le moteur zero-shot."""
        assert "VLM" in html_report or "zero-shot" in html_report.lower() or "zero_shot" in html_report
