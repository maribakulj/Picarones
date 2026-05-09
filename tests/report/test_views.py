"""Tests des 5 vues HTML thématiques (chantier 3 post-Sprint 97).

Couvre :

- Importation et signature des 5 vues.
- Adaptive masking : ``""`` quand aucune sous-section n'a de signal.
- Rendu HTML cohérent quand les données sont fournies.
- Anti-injection HTML sur les noms de moteurs et libellés.
- Composition correcte du shell ``<details>`` (premier ouvert,
  autres fermés).
- Câblage générator → vues (les variables sont passées au template).
"""

from __future__ import annotations


import pytest


# ──────────────────────────────────────────────────────────────────────────
# 1. Imports + signatures
# ──────────────────────────────────────────────────────────────────────────


class TestViewsImport:
    def test_all_views_import(self):
        from picarones.reports_v2.html.views import (
            build_advanced_taxonomy_view_html,
            build_diagnostics_view_html,
            build_economics_view_html,
            build_pipeline_view_html,
            build_robustness_view_html,
        )
        assert callable(build_advanced_taxonomy_view_html)
        assert callable(build_diagnostics_view_html)
        assert callable(build_economics_view_html)
        assert callable(build_pipeline_view_html)
        assert callable(build_robustness_view_html)


# ──────────────────────────────────────────────────────────────────────────
# 2. Adaptive masking — vues vides retournent ""
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def empty_report_data() -> dict:
    return {"engines": []}


class TestAdaptiveMasking:
    def test_economics_empty_returns_empty(self, empty_report_data):
        from picarones.reports_v2.html.views import build_economics_view_html

        assert build_economics_view_html(empty_report_data, {}) == ""

    def test_advanced_taxonomy_empty_returns_empty(self, empty_report_data):
        from picarones.reports_v2.html.views import build_advanced_taxonomy_view_html

        assert build_advanced_taxonomy_view_html(empty_report_data, {}) == ""

    def test_diagnostics_empty_returns_empty(self, empty_report_data):
        from picarones.reports_v2.html.views import build_diagnostics_view_html

        assert build_diagnostics_view_html(empty_report_data, {}) == ""

    def test_pipeline_empty_returns_empty(self, empty_report_data):
        from picarones.reports_v2.html.views import build_pipeline_view_html

        assert build_pipeline_view_html(empty_report_data, {}) == ""

    def test_robustness_empty_returns_empty(self, empty_report_data):
        from picarones.reports_v2.html.views import build_robustness_view_html

        assert build_robustness_view_html(empty_report_data, {}) == ""

    def test_advanced_taxonomy_single_engine_returns_empty(self):
        """La comparaison nécessite ≥ 2 moteurs."""
        from picarones.reports_v2.html.views import build_advanced_taxonomy_view_html

        single = {"engines": [{
            "name": "tess",
            "aggregated_taxonomy": {"class_distribution": {"x": 10}},
        }]}
        # Pas de comparison possible → vue masquée
        assert build_advanced_taxonomy_view_html(single, {}) == ""


# ──────────────────────────────────────────────────────────────────────────
# 3. Rendu HTML quand données fournies
# ──────────────────────────────────────────────────────────────────────────


class _MockMetrics:
    def __init__(self, *, cer=0.05, wer=0.1, reference_length=500):
        self.cer = cer
        self.wer = wer
        self.reference_length = reference_length
        self.error = None


class _MockDocResult:
    def __init__(self, duration=1.0):
        self.engine_error = None
        self.duration_seconds = duration
        self.metrics = _MockMetrics()


class _MockEngineReport:
    def __init__(self, name, n_docs=10):
        self.engine_name = name
        self.document_results = [_MockDocResult() for _ in range(n_docs)]


class TestEconomicsView:
    def test_throughput_with_realistic_engines(self):
        from picarones.reports_v2.html.views import build_economics_view_html

        reports = [
            _MockEngineReport("tesseract"),
            _MockEngineReport("pero_ocr"),
        ]
        html = build_economics_view_html(
            {"engines": []}, {},
            engine_reports=reports,
        )
        assert html != ""
        # Les deux moteurs doivent apparaître dans le HTML
        assert "tesseract" in html
        assert "pero" in html

    def test_extra_html_blocks_appended(self):
        from picarones.reports_v2.html.views import build_economics_view_html

        extra = ['<div class="custom">CUSTOM_BLOCK</div>']
        html = build_economics_view_html(
            {"engines": []},
            {"economics_extra_title": "Coût projeté"},
            engine_reports=[_MockEngineReport("tess")],
            extra_html_blocks=extra,
        )
        assert "CUSTOM_BLOCK" in html

    def test_zero_duration_excludes_engine(self):
        """Bench depuis cache (durations=0) ne génère pas de throughput."""
        from picarones.reports_v2.html.views import build_economics_view_html

        report = _MockEngineReport("cached")
        for dr in report.document_results:
            dr.duration_seconds = 0.0
        html = build_economics_view_html(
            {"engines": []}, {}, engine_reports=[report],
        )
        # Aucun moteur n'a de durée → vue masquée
        assert html == ""


class TestAdvancedTaxonomyView:
    def test_two_engines_taxonomy_compared(self):
        from picarones.reports_v2.html.views import build_advanced_taxonomy_view_html

        report_data = {
            "engines": [
                {
                    "name": "tess", "cer": 0.05,
                    "aggregated_taxonomy": {
                        "class_distribution": {
                            "case_error": 100, "ligature_error": 50,
                            "lacuna": 30,
                        },
                    },
                },
                {
                    "name": "pero", "cer": 0.07,
                    "aggregated_taxonomy": {
                        "class_distribution": {
                            "case_error": 30, "lacuna": 80,
                            "diacritic_error": 60,
                        },
                    },
                },
            ],
        }
        html = build_advanced_taxonomy_view_html(report_data, {})
        assert html != ""
        # Le diagramme miroir doit nommer les 2 moteurs
        assert "tess" in html
        assert "pero" in html

    def test_anti_injection_engine_name(self):
        """Un nom de moteur avec balises HTML doit être échappé."""
        from picarones.reports_v2.html.views import build_advanced_taxonomy_view_html

        report_data = {
            "engines": [
                {
                    "name": "<script>alert(1)</script>",
                    "cer": 0.05,
                    "aggregated_taxonomy": {
                        "class_distribution": {"case_error": 10},
                    },
                },
                {
                    "name": "pero",
                    "cer": 0.07,
                    "aggregated_taxonomy": {
                        "class_distribution": {"lacuna": 10},
                    },
                },
            ],
        }
        html = build_advanced_taxonomy_view_html(report_data, {})
        # Pas de balise script non échappée
        assert "<script>alert" not in html
        # Mais le contenu doit être présent sous forme échappée
        assert "&lt;script" in html or "alert" not in html.lower()

    def test_lexical_modernization_optional(self):
        from picarones.reports_v2.html.views import build_advanced_taxonomy_view_html

        report_data = {
            "engines": [
                {
                    "name": "tess", "cer": 0.05,
                    "aggregated_taxonomy": {
                        "class_distribution": {"case_error": 10},
                    },
                },
                {
                    "name": "pero", "cer": 0.07,
                    "aggregated_taxonomy": {
                        "class_distribution": {"case_error": 5},
                    },
                },
            ],
        }
        # Sans lexical_modernization, la sous-section n'apparaît pas
        html_no = build_advanced_taxonomy_view_html(report_data, {})
        # Avec, elle apparaît
        # Le format attendu par ``top_modernized_tokens`` est
        # ``{"tokens": {gt_token: {n_total, n_modernized, rate_modernized,
        # variants}}}`` (cf. ``aggregate_lexical_modernization``).
        lex_data = {
            "tokens": {
                "maistre": {
                    "n_total": 10, "n_modernized": 8,
                    "rate_modernized": 0.8,
                    "variants": {"maître": 8},
                },
            },
        }
        html_yes = build_advanced_taxonomy_view_html(
            report_data, {}, lexical_modernization=lex_data,
        )
        # Au moins une section de plus
        assert len(html_yes) > len(html_no)


class TestDiagnosticsView:
    def test_levers_only_when_signal(self):
        """detect_levers doit être appelé. Si rien ne déclenche, vue masquée."""
        from picarones.reports_v2.html.views import build_diagnostics_view_html

        # report_data minimal — aucun levier ne devrait se déclencher
        empty = {"engines": []}
        assert build_diagnostics_view_html(empty, {}) == ""

    def test_image_predictive_with_qualities(self):
        from picarones.reports_v2.html.views import build_diagnostics_view_html

        # Liste d'image_qualities synthétiques (>= 1 doc)
        qualities = [
            {
                "contrast": 0.8, "noise_level": 0.2,
                "blur_score": 0.1, "estimated_dpi": 300,
                "rotation_estimate": 0.5, "low_contrast_pct": 0.05,
            },
            {
                "contrast": 0.6, "noise_level": 0.4,
                "blur_score": 0.3, "estimated_dpi": 250,
                "rotation_estimate": 1.0, "low_contrast_pct": 0.10,
            },
        ]
        html = build_diagnostics_view_html(
            {"engines": []}, {}, image_qualities=qualities,
        )
        # La section image_predictive doit s'afficher
        assert html != ""


# ──────────────────────────────────────────────────────────────────────────
# 4. Composition du shell <details>
# ──────────────────────────────────────────────────────────────────────────


class TestDetailsShell:
    def test_first_block_open_others_closed(self):
        from picarones.reports_v2.html.views.economics import _render_view_shell

        html = _render_view_shell(
            view_title="Test",
            view_note="Note",
            blocks=[("A", "<p>aaa</p>"), ("B", "<p>bbb</p>"), ("C", "<p>ccc</p>")],
        )
        # Le premier <details> doit être ouvert
        details = html.split("<details")
        assert "open" in details[1].split(">")[0]
        # Les suivants ne doivent pas l'être
        assert "open" not in details[2].split(">")[0]
        assert "open" not in details[3].split(">")[0]
        # Tous les contenus présents
        assert "aaa" in html and "bbb" in html and "ccc" in html

    def test_xml_chars_in_titles_escaped(self):
        from picarones.reports_v2.html.views.economics import _render_view_shell

        html = _render_view_shell(
            view_title="<script>alert(1)</script>",
            view_note="Note <b>bold</b>",
            blocks=[("Block <X>", "<p>content</p>")],
        )
        # Pas d'injection
        assert "<script>alert(1)</script>" not in html
        # Mais visible sous forme échappée
        assert "&lt;script" in html


# ──────────────────────────────────────────────────────────────────────────
# 5. Câblage générator → vues
# ──────────────────────────────────────────────────────────────────────────


class TestGeneratorWiring:
    def test_generator_imports_three_views(self):
        """Test runtime du câblage des 3 vues automatiques (economics,
        advanced_taxonomy, diagnostics).

        Vérifie que la méthode ``ReportGenerator._build_section_html``
        retourne un dict contenant les 3 clés attendues, ce qui
        garantit qu'elles seront splatées vers le template Jinja.

        Cette version remplace l'ancien test qui scannait textuellement
        ``generator.py`` à la recherche de ``var=`` ou ``"var"`` —
        approche fragile (passait sur n'importe quelle occurrence dans
        une docstring) et trop liée à la forme du code.
        """
        from picarones.evaluation.synthetic import generate_sample_benchmark
        from picarones.reports_v2.html.generator import ReportGenerator

        bench = generate_sample_benchmark()
        gen = ReportGenerator(bench, lang="fr")
        from picarones.reports_v2.i18n import get_labels

        report_data = {
            "engines": [],
            "inter_engine_analysis": None,
            "stratified_ranking": None,
            "available_strata": [],
            "corpus_homogeneity": None,
        }
        section_html = gen._build_section_html(report_data, get_labels("fr"))
        for name in (
            "economics_view_html",
            "advanced_taxonomy_view_html",
            "diagnostics_view_html",
        ):
            assert name in section_html, (
                f"clé {name!r} absente du dict retourné par "
                "ReportGenerator._build_section_html — la vue ne sera "
                "pas câblée vers le template."
            )
            # Adaptive : avec un report_data vide, chaque vue retourne ""
            # (rapport adaptatif). On vérifie le type, pas le contenu.
            assert isinstance(section_html[name], str), (
                f"section {name!r} doit être une chaîne, "
                f"pas {type(section_html[name]).__name__}"
            )

    def test_template_uses_three_views(self):
        from pathlib import Path

        tpl_src = (
            Path(__file__).parent.parent.parent
            / "picarones" / "reports_v2" / "html" / "templates" / "view_analyses.html"
        ).read_text(encoding="utf-8")
        assert "{% if economics_view_html %}" in tpl_src
        assert "{% if advanced_taxonomy_view_html %}" in tpl_src
        assert "{% if diagnostics_view_html %}" in tpl_src
