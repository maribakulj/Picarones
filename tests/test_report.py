"""Tests pour picarones.report (generator + fixtures)."""

import json
import pytest
from pathlib import Path

from picarones.fixtures import generate_sample_benchmark
from picarones.report.generator import ReportGenerator, _build_report_data, _cer_color, _cer_bg


# ---------------------------------------------------------------------------
# Fixtures Python (données de test)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_benchmark():
    return generate_sample_benchmark(n_docs=3, seed=0, include_images=True)


@pytest.fixture
def sample_generator(sample_benchmark):
    return ReportGenerator(sample_benchmark)


# ---------------------------------------------------------------------------
# Tests generate_sample_benchmark
# ---------------------------------------------------------------------------

class TestGenerateSampleBenchmark:
    def test_returns_benchmark_result(self, sample_benchmark):
        from picarones.core.results import BenchmarkResult
        assert isinstance(sample_benchmark, BenchmarkResult)

    def test_correct_engine_count(self, sample_benchmark):
        # 3 moteurs OCR + 1 pipeline tesseract → gpt-4o
        assert len(sample_benchmark.engine_reports) == 4

    def test_correct_doc_count(self, sample_benchmark):
        assert sample_benchmark.document_count == 3
        for report in sample_benchmark.engine_reports:
            assert len(report.document_results) == 3

    def test_engine_names(self, sample_benchmark):
        names = {r.engine_name for r in sample_benchmark.engine_reports}
        assert "tesseract" in names
        assert "pero_ocr" in names

    def test_images_in_metadata(self, sample_benchmark):
        images = sample_benchmark.metadata.get("_images_b64", {})
        assert len(images) == 3
        for v in images.values():
            assert v.startswith("data:image/png;base64,")

    def test_reproducible_with_seed(self):
        bm1 = generate_sample_benchmark(n_docs=3, seed=42)
        bm2 = generate_sample_benchmark(n_docs=3, seed=42)
        # Même CER pour le même seed
        cer1 = bm1.engine_reports[0].document_results[0].metrics.cer
        cer2 = bm2.engine_reports[0].document_results[0].metrics.cer
        assert cer1 == pytest.approx(cer2)

    def test_without_images(self):
        bm = generate_sample_benchmark(n_docs=2, include_images=False)
        assert bm.metadata.get("_images_b64", {}) == {}

    def test_metrics_computed(self, sample_benchmark):
        for report in sample_benchmark.engine_reports:
            for dr in report.document_results:
                assert dr.metrics.cer >= 0.0
                assert dr.metrics.wer >= 0.0

    def test_aggregated_metrics(self, sample_benchmark):
        for report in sample_benchmark.engine_reports:
            assert "cer" in report.aggregated_metrics
            assert "mean" in report.aggregated_metrics["cer"]


# ---------------------------------------------------------------------------
# Tests _build_report_data
# ---------------------------------------------------------------------------

class TestBuildReportData:
    def test_structure(self, sample_benchmark):
        data = _build_report_data(sample_benchmark, {})
        assert "meta" in data
        assert "ranking" in data
        assert "engines" in data
        assert "documents" in data

    def test_engines_count(self, sample_benchmark):
        data = _build_report_data(sample_benchmark, {})
        # 3 moteurs OCR + 1 pipeline tesseract → gpt-4o
        assert len(data["engines"]) == 4

    def test_engine_fields(self, sample_benchmark):
        data = _build_report_data(sample_benchmark, {})
        for e in data["engines"]:
            for field in ["name", "version", "cer", "wer", "mer", "wil", "cer_values"]:
                assert field in e

    def test_documents_count(self, sample_benchmark):
        data = _build_report_data(sample_benchmark, {})
        assert len(data["documents"]) == 3

    def test_document_fields(self, sample_benchmark):
        data = _build_report_data(sample_benchmark, {})
        for doc in data["documents"]:
            for field in ["doc_id", "image_path", "ground_truth", "mean_cer",
                          "best_engine", "engine_results"]:
                assert field in doc

    def test_diff_computed(self, sample_benchmark):
        data = _build_report_data(sample_benchmark, {})
        for doc in data["documents"]:
            for er in doc["engine_results"]:
                assert "diff" in er
                assert isinstance(er["diff"], list)

    def test_images_attached(self, sample_benchmark):
        images = sample_benchmark.metadata.get("_images_b64", {})
        data = _build_report_data(sample_benchmark, images)
        # Au moins un document doit avoir une image b64
        has_image = any(doc["image_b64"] for doc in data["documents"])
        assert has_image

    def test_cer_values_list(self, sample_benchmark):
        data = _build_report_data(sample_benchmark, {})
        for engine in data["engines"]:
            assert len(engine["cer_values"]) == 3
            assert all(isinstance(v, float) for v in engine["cer_values"])


# ---------------------------------------------------------------------------
# Tests ReportGenerator.generate
# ---------------------------------------------------------------------------

class TestReportGenerator:
    def test_generates_file(self, sample_generator, tmp_path):
        out = tmp_path / "rapport.html"
        path = sample_generator.generate(out)
        assert path.exists()
        assert path.suffix == ".html"

    def test_file_not_empty(self, sample_generator, tmp_path):
        out = tmp_path / "rapport.html"
        sample_generator.generate(out)
        content = out.read_text(encoding="utf-8")
        assert len(content) > 5000  # fichier substantiel

    def test_html_structure(self, sample_generator, tmp_path):
        out = tmp_path / "rapport.html"
        sample_generator.generate(out)
        html = out.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html

    def test_contains_chart_js(self, sample_generator, tmp_path):
        out = tmp_path / "rapport.html"
        sample_generator.generate(out)
        html = out.read_text(encoding="utf-8")
        assert "chart.js" in html.lower() or "Chart.js" in html

    def test_contains_diff2html(self, sample_generator, tmp_path):
        out = tmp_path / "rapport.html"
        sample_generator.generate(out)
        html = out.read_text(encoding="utf-8")
        assert "diff2html" in html.lower()

    def test_data_embedded(self, sample_generator, tmp_path):
        out = tmp_path / "rapport.html"
        sample_generator.generate(out)
        html = out.read_text(encoding="utf-8")
        assert "const DATA" in html

    def test_engine_names_in_html(self, sample_generator, tmp_path):
        out = tmp_path / "rapport.html"
        sample_generator.generate(out)
        html = out.read_text(encoding="utf-8")
        assert "tesseract" in html
        assert "pero_ocr" in html

    def test_corpus_name_in_html(self, sample_generator, tmp_path):
        out = tmp_path / "rapport.html"
        sample_generator.generate(out)
        html = out.read_text(encoding="utf-8")
        assert "chroniques" in html.lower()

    def test_creates_parent_dirs(self, sample_generator, tmp_path):
        out = tmp_path / "deep" / "nested" / "rapport.html"
        sample_generator.generate(out)
        assert out.exists()

    def test_returns_absolute_path(self, sample_generator, tmp_path):
        out = tmp_path / "rapport.html"
        result = sample_generator.generate(out)
        assert result.is_absolute()

    def test_json_roundtrip(self, sample_benchmark, tmp_path):
        """Vérifie que le rapport peut être généré depuis un JSON sauvegardé."""
        json_path = tmp_path / "results.json"
        sample_benchmark.to_json(json_path)

        gen = ReportGenerator.from_json(json_path)
        html_path = tmp_path / "rapport.html"
        gen.generate(html_path)
        assert html_path.exists()
        html = html_path.read_text(encoding="utf-8")
        assert "const DATA" in html

    def test_embedded_json_valid(self, sample_generator, tmp_path):
        """Extrait et parse le JSON embarqué dans le HTML pour vérifier sa validité."""
        out = tmp_path / "rapport.html"
        sample_generator.generate(out)
        html = out.read_text(encoding="utf-8")

        # Extraire le JSON entre "const DATA = " et le ";" de fin de ligne
        import re
        match = re.search(r"const DATA = (\{.*?\});", html, re.DOTALL)
        assert match, "Bloc const DATA non trouvé dans le HTML"
        data = json.loads(match.group(1))
        assert "engines" in data
        assert "documents" in data
        assert len(data["engines"]) == 4  # 3 OCR + 1 pipeline


# ---------------------------------------------------------------------------
# Tests helpers de couleur
# ---------------------------------------------------------------------------

class TestCerColor:
    def test_green_below_5pct(self):
        assert _cer_color(0.04) == "#16a34a"

    def test_yellow_5_to_15pct(self):
        assert _cer_color(0.10) == "#ca8a04"

    def test_orange_15_to_30pct(self):
        assert _cer_color(0.20) == "#ea580c"

    def test_red_above_30pct(self):
        assert _cer_color(0.50) == "#dc2626"

    def test_boundary_exactly_5pct(self):
        # 0.05 est dans la zone jaune (>= 0.05)
        assert _cer_color(0.05) == "#ca8a04"
