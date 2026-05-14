"""Phase B6 — rendu HTML des ``BenchmarkResult.view_results``.

Vérifie que le renderer ``build_view_results_html`` :

1. Retourne ``""`` quand ``view_results`` est vide ou ``None`` (compat
   ascendante : un BenchmarkResult issu de
   ``run_benchmark_via_service`` sans RunOrchestrator n'a pas de
   ``view_results``).
2. Génère une section par vue présente, avec titre + note
   méthodologique + tableau engine × moyenne_par_metric.
3. Liste explicitement les pipelines OMIS de chaque vue (= ceux qui
   n'ont pas produit d'artefact éligible).
4. Échappe le HTML correctement (résistance XSS via noms d'engine
   custom).
5. S'intègre proprement dans le rapport HTML complet (test bout-en-bout
   via ``ReportGenerator``).
"""

from __future__ import annotations

from picarones.evaluation.benchmark_result import BenchmarkResult, EngineReport
from picarones.evaluation.metric_result import MetricsResult
from picarones.reports.html.renderers.view_results import (
    build_view_results_html,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _make_engine_report(name: str) -> EngineReport:
    return EngineReport(
        engine_name=name,
        engine_version="test",
        engine_config={},
        document_results=[],
        aggregated_metrics={},
    )


# ──────────────────────────────────────────────────────────────────────
# Renderer adaptatif (cas vides)
# ──────────────────────────────────────────────────────────────────────


class TestEmptyViewResults:
    def test_none_returns_empty_string(self) -> None:
        assert build_view_results_html(None, all_engine_names=["t"]) == ""

    def test_empty_dict_returns_empty_string(self) -> None:
        assert build_view_results_html({}, all_engine_names=["t"]) == ""


# ──────────────────────────────────────────────────────────────────────
# Rendu d'une vue avec données
# ──────────────────────────────────────────────────────────────────────


class TestSingleViewRendering:
    def _sample_view_results(
        self,
    ) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
        return {
            "text_final": {
                "tesseract": {
                    "doc1": {"cer": 0.05, "wer": 0.10},
                    "doc2": {"cer": 0.03, "wer": 0.08},
                },
            },
        }

    def test_section_contains_view_title(self) -> None:
        html = build_view_results_html(
            self._sample_view_results(), all_engine_names=["tesseract"],
        )
        assert "TextView" in html
        # Note méthodologique présente.
        assert "projetées" in html.lower() or "projeté" in html.lower()

    def test_section_contains_engine_metrics_table(self) -> None:
        html = build_view_results_html(
            self._sample_view_results(), all_engine_names=["tesseract"],
        )
        # Header + métrique + valeur.
        assert "tesseract" in html
        assert "cer" in html
        assert "wer" in html
        # Moyenne CER : (0.05 + 0.03) / 2 = 0.04 → 4.00%.
        assert "4.00%" in html
        # Moyenne WER : (0.10 + 0.08) / 2 = 0.09 → 9.00%.
        assert "9.00%" in html

    def test_no_omitted_when_all_eligible(self) -> None:
        html = build_view_results_html(
            self._sample_view_results(), all_engine_names=["tesseract"],
        )
        # "Tous les pipelines éligibles" affiché car aucun n'est omis.
        assert "éligibles" in html or "eligible" in html.lower()


# ──────────────────────────────────────────────────────────────────────
# Pipelines omis (cas AltoView avec engine OCR pur)
# ──────────────────────────────────────────────────────────────────────


class TestOmittedPipelines:
    def test_alto_view_omits_text_only_engine(self) -> None:
        """Cas typique : AltoView ne reçoit que des résultats du
        pipeline qui produit ALTO.  Un pipeline OCR seul est omis."""
        view_results = {
            "alto_documentary": {
                "tesseract_alto": {
                    "doc1": {"alto_validity": 1.0},
                },
                # Pas de "tesseract_text_only" → omis de cette vue
            },
        }
        html = build_view_results_html(
            view_results,
            all_engine_names=["tesseract_alto", "tesseract_text_only"],
        )
        assert "tesseract_alto" in html
        # tesseract_text_only listé dans Pipelines omis.
        assert "tesseract_text_only" in html
        # Le label "Pipelines omis" est présent.
        assert "omis" in html.lower() or "omitted" in html.lower()


# ──────────────────────────────────────────────────────────────────────
# Multi-vues (le cas typique de production)
# ──────────────────────────────────────────────────────────────────────


class TestMultipleViews:
    def test_renders_three_canonical_views(self) -> None:
        view_results = {
            "text_final": {
                "tesseract": {"doc1": {"cer": 0.1}},
            },
            "alto_documentary": {
                "tesseract": {"doc1": {"alto_validity": 1.0}},
            },
            "searchability": {
                "tesseract": {"doc1": {"searchability_recall": 0.95}},
            },
        }
        html = build_view_results_html(
            view_results, all_engine_names=["tesseract"],
        )
        assert "TextView" in html
        assert "AltoView" in html
        assert "SearchView" in html


# ──────────────────────────────────────────────────────────────────────
# Sécurité — XSS via noms d'engine custom
# ──────────────────────────────────────────────────────────────────────


class TestXssEscaping:
    def test_engine_name_with_html_chars_is_escaped(self) -> None:
        view_results = {
            "text_final": {
                "<script>alert(1)</script>": {"doc1": {"cer": 0.1}},
            },
        }
        html = build_view_results_html(
            view_results, all_engine_names=["<script>alert(1)</script>"],
        )
        # Le HTML brut ne doit pas apparaître non échappé.
        assert "<script>" not in html
        # L'entité échappée est présente.
        assert "&lt;script&gt;" in html

    def test_metric_name_with_html_chars_is_escaped(self) -> None:
        view_results = {
            "text_final": {
                "tesseract": {"doc1": {"<weird>": 0.1}},
            },
        }
        html = build_view_results_html(
            view_results, all_engine_names=["tesseract"],
        )
        assert "<weird>" not in html
        assert "&lt;weird&gt;" in html


# ──────────────────────────────────────────────────────────────────────
# Internationalization
# ──────────────────────────────────────────────────────────────────────


class TestI18n:
    def _sample(self) -> dict:
        return {
            "alto_documentary": {
                "tess": {"doc1": {"alto_validity": 1.0}},
            },
        }

    def test_french_default_labels(self) -> None:
        html = build_view_results_html(
            self._sample(), all_engine_names=["tess", "other"], lang="fr",
        )
        assert "documentaire" in html.lower()
        assert "pipelines omis" in html.lower()

    def test_english_labels(self) -> None:
        html = build_view_results_html(
            self._sample(), all_engine_names=["tess", "other"], lang="en",
        )
        assert "documentary" in html.lower()
        assert "omitted pipelines" in html.lower()


# ──────────────────────────────────────────────────────────────────────
# Intégration avec ReportGenerator
# ──────────────────────────────────────────────────────────────────────


class TestReportGeneratorIntegration:
    def _make_benchmark(
        self, with_view_results: bool,
    ) -> BenchmarkResult:
        # Document minimal.  Les hooks et agrégats sont vides — on
        # teste juste la présence/absence de la section view_results.
        from picarones.evaluation.benchmark_result import DocumentResult

        engine = EngineReport(
            engine_name="tesseract",
            engine_version="5.x",
            engine_config={},
            document_results=[
                DocumentResult(
                    doc_id="doc1",
                    image_path="/tmp/doc1.png",
                    ground_truth="Bonjour",
                    hypothesis="Bonjour",
                    metrics=MetricsResult(
                        cer=0.0, cer_nfc=0.0, cer_caseless=0.0,
                        wer=0.0, wer_normalized=0.0, mer=0.0, wil=0.0,
                        reference_length=7, hypothesis_length=7,
                    ),
                    duration_seconds=0.1,
                ),
            ],
            aggregated_metrics={},
        )
        view_results: dict = {}
        if with_view_results:
            view_results = {
                "text_final": {
                    "tesseract": {"doc1": {"cer": 0.0, "wer": 0.0}},
                },
                "alto_documentary": {
                    # Aucun engine n'a produit d'ALTO ici → vue vide
                    # mais tesseract est listé comme omis.
                },
            }
        return BenchmarkResult(
            corpus_name="test_corpus",
            corpus_source=None,
            document_count=1,
            engine_reports=[engine],
            view_results=view_results,
        )

    def test_report_includes_view_section_when_present(self, tmp_path) -> None:
        from picarones.reports.html.generator import ReportGenerator

        bm = self._make_benchmark(with_view_results=True)
        out = tmp_path / "report.html"
        ReportGenerator(bm, lang="fr").generate(out)

        html = out.read_text(encoding="utf-8")
        assert "TextView" in html
        assert "AltoView" in html

    def test_report_omits_view_section_when_absent(self, tmp_path) -> None:
        """Compat ascendante : sans view_results, le rapport HTML
        legacy est intact (aucune section `view-results-section`)."""
        from picarones.reports.html.generator import ReportGenerator

        bm = self._make_benchmark(with_view_results=False)
        out = tmp_path / "report.html"
        ReportGenerator(bm, lang="fr").generate(out)

        html = out.read_text(encoding="utf-8")
        # Le marker CSS du renderer view_results doit être absent.
        assert "view-results-section" not in html
