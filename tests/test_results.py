"""Tests unitaires pour picarones.core.results."""

import json
import pytest
from pathlib import Path

from picarones.core.metrics import MetricsResult
from picarones.core.results import BenchmarkResult, DocumentResult, EngineReport


def _make_metrics(cer: float = 0.05) -> MetricsResult:
    return MetricsResult(
        cer=cer, cer_nfc=cer, cer_caseless=cer,
        wer=cer * 2, wer_normalized=cer * 2, mer=cer, wil=cer,
        reference_length=200, hypothesis_length=195,
    )


def _make_document_result(doc_id: str = "doc1", cer: float = 0.05) -> DocumentResult:
    return DocumentResult(
        doc_id=doc_id,
        image_path=f"/corpus/{doc_id}.png",
        ground_truth="Texte de référence médiéval.",
        hypothesis="Texte de référence médiéval.",
        metrics=_make_metrics(cer),
        duration_seconds=1.23,
    )


def _make_engine_report(name: str = "tesseract", n_docs: int = 3) -> EngineReport:
    docs = [_make_document_result(f"doc{i}", cer=0.03 * i) for i in range(1, n_docs + 1)]
    return EngineReport(
        engine_name=name,
        engine_version="5.3.0",
        engine_config={"lang": "fra"},
        document_results=docs,
    )


class TestDocumentResult:
    def test_as_dict_keys(self):
        dr = _make_document_result()
        d = dr.as_dict()
        for key in ["doc_id", "image_path", "ground_truth", "hypothesis", "metrics", "duration_seconds"]:
            assert key in d

    def test_metrics_serialized(self):
        dr = _make_document_result(cer=0.1)
        d = dr.as_dict()
        assert d["metrics"]["cer"] == pytest.approx(0.1)


class TestEngineReport:
    def test_aggregation_computed(self):
        report = _make_engine_report(n_docs=3)
        assert report.aggregated_metrics != {}
        assert "cer" in report.aggregated_metrics

    def test_mean_cer(self):
        report = _make_engine_report(n_docs=3)
        # docs avec cer=0.03, 0.06, 0.09 → mean=0.06
        assert report.mean_cer == pytest.approx(0.06, rel=1e-2)

    def test_as_dict_structure(self):
        report = _make_engine_report()
        d = report.as_dict()
        assert d["engine_name"] == "tesseract"
        assert len(d["document_results"]) == 3


class TestBenchmarkResult:
    def _make_benchmark(self) -> BenchmarkResult:
        return BenchmarkResult(
            corpus_name="Test corpus",
            corpus_source="/corpus/",
            document_count=3,
            engine_reports=[
                _make_engine_report("tesseract"),
                _make_engine_report("pero_ocr"),
            ],
        )

    def test_ranking_sorted_by_cer(self):
        bm = self._make_benchmark()
        ranking = bm.ranking()
        assert len(ranking) == 2
        cers = [e["mean_cer"] for e in ranking]
        assert cers == sorted(cers)

    def test_to_json_writes_file(self, tmp_path):
        bm = self._make_benchmark()
        out = tmp_path / "results.json"
        bm.to_json(out)
        assert out.exists()
        with out.open() as f:
            data = json.load(f)
        assert data["corpus"]["name"] == "Test corpus"

    def test_to_json_creates_parent_dirs(self, tmp_path):
        bm = self._make_benchmark()
        out = tmp_path / "deep" / "nested" / "results.json"
        bm.to_json(out)
        assert out.exists()

    def test_from_json_round_trip(self, tmp_path):
        bm = self._make_benchmark()
        out = tmp_path / "results.json"
        bm.to_json(out)
        loaded = BenchmarkResult.from_json(out)
        assert loaded["corpus"]["name"] == "Test corpus"
        assert len(loaded["engine_reports"]) == 2

    def test_as_dict_has_version(self):
        bm = self._make_benchmark()
        d = bm.as_dict()
        assert "picarones_version" in d
        assert "run_date" in d

    def test_ranking_has_required_fields(self):
        bm = self._make_benchmark()
        for entry in bm.ranking():
            assert "engine" in entry
            assert "mean_cer" in entry
            assert "mean_wer" in entry
