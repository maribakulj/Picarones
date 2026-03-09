"""Tests pour les nouvelles fonctionnalités du sprint 12 :
1. Filtrage des fichiers cachés macOS (._*) dans corpus et ZIP
2. Profils de normalisation avec exclusion de caractères
3. Vue Analyses — Chart.js inline (plus de CDN)
4. Métriques robustes dans le rapport HTML
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# 1. Filtrage des fichiers cachés macOS
# ---------------------------------------------------------------------------

FAKE_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
    b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
    b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


class TestMacOSHiddenFilesFiltering:
    def test_hidden_images_ignored_in_corpus(self, tmp_path):
        """Les fichiers ._* ne doivent pas être comptés comme images valides."""
        from picarones.core.corpus import load_corpus_from_directory

        # Image réelle avec GT
        (tmp_path / "page_001.png").write_bytes(FAKE_PNG)
        (tmp_path / "page_001.gt.txt").write_text("Texte réel", encoding="utf-8")

        # Fichiers AppleDouble macOS (sans GT associé)
        (tmp_path / "._page_001.png").write_bytes(b"\x00\x05\x16\x07")
        (tmp_path / ".DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1")

        corpus = load_corpus_from_directory(tmp_path)
        assert len(corpus) == 1
        assert corpus.documents[0].doc_id == "page_001"

    def test_hidden_files_not_extracted_from_zip(self, tmp_path):
        """_flatten_zip_to_dir doit ignorer les entrées ._* dans le ZIP."""
        from picarones.web.app import _flatten_zip_to_dir

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("page_001.png", FAKE_PNG)
            zf.writestr("page_001.gt.txt", "Texte réel")
            zf.writestr("._page_001.png", b"\x00\x05\x16\x07")
            zf.writestr("__MACOSX/._page_001.png", b"\x00\x05\x16\x07")

        buf.seek(0)
        dest = tmp_path / "corpus"
        dest.mkdir()
        with zipfile.ZipFile(buf) as zf:
            _flatten_zip_to_dir(zf, dest)

        files = {f.name for f in dest.iterdir()}
        assert "._page_001.png" not in files
        assert "page_001.png" in files
        assert "page_001.gt.txt" in files


# ---------------------------------------------------------------------------
# 2. Profils de normalisation avec exclusion de caractères
# ---------------------------------------------------------------------------

class TestExcludeCharsNormalization:
    def test_parse_exclude_chars_from_comma_string(self):
        from picarones.core.normalization import _parse_exclude_chars

        result = _parse_exclude_chars("', -, –")
        assert "'" in result
        assert "-" in result
        assert "–" in result

    def test_parse_exclude_chars_from_plain_string(self):
        from picarones.core.normalization import _parse_exclude_chars

        result = _parse_exclude_chars(".,;:!?")
        assert "." in result
        assert "," in result
        assert "?" in result

    def test_parse_exclude_chars_empty(self):
        from picarones.core.normalization import _parse_exclude_chars

        assert _parse_exclude_chars("") == frozenset()
        assert _parse_exclude_chars(None) == frozenset()

    def test_normalize_strips_excluded_chars(self):
        from picarones.core.normalization import NormalizationProfile

        profile = NormalizationProfile(
            name="test",
            exclude_chars=frozenset([".", ","]),
        )
        assert profile.normalize("Bonjour, monde.") == "Bonjour monde"

    def test_sans_ponctuation_profile_exists(self):
        from picarones.core.normalization import NORMALIZATION_PROFILES

        assert "sans_ponctuation" in NORMALIZATION_PROFILES
        p = NORMALIZATION_PROFILES["sans_ponctuation"]
        assert "." in p.exclude_chars
        assert "," in p.exclude_chars
        assert "?" in p.exclude_chars

    def test_sans_apostrophes_profile_exists(self):
        from picarones.core.normalization import NORMALIZATION_PROFILES

        assert "sans_apostrophes" in NORMALIZATION_PROFILES
        p = NORMALIZATION_PROFILES["sans_apostrophes"]
        assert "'" in p.exclude_chars
        assert "\u2019" in p.exclude_chars  # apostrophe typographique

    def test_compute_metrics_with_char_exclude(self):
        from picarones.core.metrics import compute_metrics

        ref = "Bonjour, monde!"
        hyp = "Bonjour monde"
        # Sans exclusion, CER > 0 (virgule et ! manquants)
        metrics_raw = compute_metrics(ref, hyp)
        assert metrics_raw.cer > 0

        # Avec exclusion de la ponctuation, les deux textes deviennent identiques
        metrics_excl = compute_metrics(ref, hyp, char_exclude=frozenset([",", "!", " "]))
        # CER devrait être 0 ou très faible maintenant (Bonjourmonde == Bonjourmonde)
        assert metrics_excl.cer == 0.0

    def test_char_exclude_propagated_in_run_benchmark(self, tmp_path):
        """char_exclude doit être transmis à run_benchmark et réduire le CER."""
        from picarones.core.corpus import Corpus, Document
        from picarones.core.runner import run_benchmark
        from picarones.engines.base import BaseOCREngine, EngineResult

        class MockEngine(BaseOCREngine):
            name = "mock"
            version = "0.0"

            def _run_ocr(self, image_path):
                return EngineResult(text="Bonjour monde", success=True)

        doc = Document(image_path=tmp_path / "page.png", ground_truth="Bonjour, monde!")
        (tmp_path / "page.png").write_bytes(FAKE_PNG)
        corpus = Corpus(name="test", documents=[doc])

        result_raw = run_benchmark(corpus, [MockEngine()])
        cer_raw = result_raw.engine_reports[0].document_results[0].metrics.cer

        result_excl = run_benchmark(corpus, [MockEngine()], char_exclude=frozenset([",", "!"]))
        cer_excl = result_excl.engine_reports[0].document_results[0].metrics.cer

        assert cer_excl <= cer_raw


# ---------------------------------------------------------------------------
# 3. Vue Analyses — Chart.js inline
# ---------------------------------------------------------------------------

class TestChartJsInline:
    def test_chartjs_embedded_inline(self, sample_generator, tmp_path):
        """Le rapport HTML doit embarquer Chart.js inline (pas de CDN)."""
        out = tmp_path / "rapport.html"
        sample_generator.generate(out)
        html = out.read_text(encoding="utf-8")

        assert "cdnjs.cloudflare.com/ajax/libs/Chart.js" not in html
        assert "Chart.js v" in html or "new Chart(" in html

    def test_no_diff2html_cdn(self, sample_generator, tmp_path):
        """Le rapport ne doit plus référencer diff2html (CDN supprimé)."""
        out = tmp_path / "rapport.html"
        sample_generator.generate(out)
        html = out.read_text(encoding="utf-8")

        assert "diff2html" not in html

    def test_build_charts_function_present(self, sample_generator, tmp_path):
        out = tmp_path / "rapport.html"
        sample_generator.generate(out)
        html = out.read_text(encoding="utf-8")

        assert "function buildCharts()" in html
        assert "buildCerHistogram" in html
        assert "buildRadar" in html


@pytest.fixture
def sample_generator():
    """Fixture partagée : crée un ReportGenerator avec des données fictives."""
    from picarones.report.generator import ReportGenerator
    from picarones.core.results import BenchmarkResult, DocumentResult, EngineReport
    from picarones.core.metrics import MetricsResult

    def _make_metric(cer=0.1):
        return MetricsResult(
            cer=cer, cer_nfc=cer, cer_caseless=cer,
            wer=cer, wer_normalized=cer, mer=cer, wil=cer,
            reference_length=100, hypothesis_length=100,
        )

    docs = [
        DocumentResult(
            doc_id=f"doc_{i}", image_path="", ground_truth="GT text",
            hypothesis="Hyp text", metrics=_make_metric(0.1 + i * 0.01),
            duration_seconds=0.1,
        )
        for i in range(3)
    ]
    report = EngineReport(engine_name="tesseract", engine_version="5.0", engine_config={}, document_results=docs)
    bm = BenchmarkResult(
        corpus_name="TestCorpus", corpus_source=None, document_count=3,
        engine_reports=[report],
    )
    return ReportGenerator(bm)


# ---------------------------------------------------------------------------
# 4. Métriques robustes — présence dans le rapport HTML
# ---------------------------------------------------------------------------

class TestRobustMetrics:
    def test_robust_metrics_card_present(self, sample_generator, tmp_path):
        """La carte Métriques robustes doit être présente dans le rapport."""
        out = tmp_path / "rapport.html"
        sample_generator.generate(out)
        html = out.read_text(encoding="utf-8")

        assert "robust-metrics-card" in html
        assert "robust-anchor" in html
        assert "robust-ratio" in html
        assert "renderRobustMetrics" in html

    def test_robust_metrics_js_syntax_valid(self, sample_generator, tmp_path):
        """La fonction renderRobustMetrics ne doit pas introduire de SyntaxError JS."""
        import re
        import subprocess

        out = tmp_path / "rapport.html"
        sample_generator.generate(out)
        html = out.read_text(encoding="utf-8")

        scripts = re.findall(r"<script>(.*?)</script>", html, re.DOTALL)
        # Le bloc applicatif est le dernier script
        app_js = tmp_path / "app.js"
        app_js.write_text(scripts[-1], encoding="utf-8")

        result = subprocess.run(
            ["node", "--check", str(app_js)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"Erreur JS : {result.stderr}"
