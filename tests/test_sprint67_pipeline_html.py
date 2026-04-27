"""Tests Sprint 67 — vue HTML d'un benchmark de pipeline composée.

Couvre :

1. ``build_pipeline_summary_html`` : affiche pipeline, corpus,
   n_docs, succeeded/failed, durée totale.
2. ``build_pipeline_steps_table_html`` : tableau par étape avec
   colonnes attendues, métriques aux jonctions formatées,
   error_breakdown affiché, vide si aucune étape.
3. ``build_pipeline_report_html`` : document HTML autonome
   (``<!doctype html>``, head, body, styles inline).
4. Anti-injection HTML : noms de pipeline / corpus / step
   contenant ``<script>`` correctement échappés.
5. Adaptive masking : pas d'étape → tableau vide.
6. Complétude i18n : toutes les clés ``pipeline_*`` présentes en
   FR et EN.
"""

from __future__ import annotations

import json
from pathlib import Path

from picarones.core.pipeline_benchmark import (
    PipelineBenchmarkResult,
    StepAggregate,
)
from picarones.report.pipeline_render import (
    build_pipeline_report_html,
    build_pipeline_steps_table_html,
    build_pipeline_summary_html,
)


def _make_bench(
    name: str = "ocr_then_fix",
    corpus: str = "demo",
) -> PipelineBenchmarkResult:
    bench = PipelineBenchmarkResult(
        pipeline_name=name, corpus_name=corpus,
        n_docs=10, total_duration_seconds=12.345,
    )
    bench.per_step_aggregates = [
        StepAggregate(
            step_name="ocr", n_docs=10, n_succeeded=10, n_failed=0,
            duration_seconds_total=2.5, duration_seconds_mean=0.25,
            duration_seconds_median=0.24,
            junction_metrics={
                "text": {"cer": {"mean": 0.182, "median": 0.18, "n": 10}},
            },
        ),
        StepAggregate(
            step_name="rewrite", n_docs=10, n_succeeded=8, n_failed=2,
            duration_seconds_total=1.2, duration_seconds_mean=0.15,
            duration_seconds_median=0.14, failing_doc_ids=["d3", "d7"],
            junction_metrics={
                "text": {"cer": {"mean": 0.05, "median": 0.04, "n": 8}},
            },
            error_breakdown={"raised_exception": 2},
        ),
    ]

    class _FakePR:
        def __init__(self, ok): self._ok = ok
        @property
        def succeeded(self): return self._ok
    bench.per_doc_results = [_FakePR(True)] * 8 + [_FakePR(False)] * 2
    return bench


# ──────────────────────────────────────────────────────────────────────────
# 1. Summary
# ──────────────────────────────────────────────────────────────────────────


class TestSummary:
    def test_renders_pipeline_and_corpus_name(self) -> None:
        bench = _make_bench()
        html = build_pipeline_summary_html(bench)
        assert "ocr_then_fix" in html
        assert "demo" in html

    def test_includes_succeeded_and_failed(self) -> None:
        bench = _make_bench()
        html = build_pipeline_summary_html(bench)
        # 8 réussies sur 10 (les fakes per_doc_results)
        assert "8 / 10" in html
        # 2 échouées
        assert ">2<" in html

    def test_duration_formatted(self) -> None:
        bench = _make_bench()
        html = build_pipeline_summary_html(bench)
        # 12.345s → "12.35 s" (formatage en s pour > 1s < 60s)
        assert "12.35 s" in html


# ──────────────────────────────────────────────────────────────────────────
# 2. Steps table
# ──────────────────────────────────────────────────────────────────────────


class TestStepsTable:
    def test_renders_step_names(self) -> None:
        bench = _make_bench()
        html = build_pipeline_steps_table_html(bench)
        assert "ocr" in html
        assert "rewrite" in html

    def test_columns_present(self) -> None:
        bench = _make_bench()
        html = build_pipeline_steps_table_html(bench)
        # Vérifie les en-têtes par défaut FR
        for col in (
            "Étape", "Réussies", "Échouées", "Taux succès",
            "Durée moyenne", "Durée médiane",
            "Métriques aux jonctions", "Erreurs",
        ):
            assert col in html

    def test_metrics_displayed(self) -> None:
        bench = _make_bench()
        html = build_pipeline_steps_table_html(bench)
        # Métriques formatées : type.metric : mean (n=N)
        assert "text.cer" in html
        assert "0.182" in html
        assert "0.050" in html
        assert "n=10" in html
        assert "n=8" in html

    def test_error_breakdown_displayed(self) -> None:
        bench = _make_bench()
        html = build_pipeline_steps_table_html(bench)
        assert "raised_exception" in html

    def test_empty_when_no_aggregates(self) -> None:
        bench = PipelineBenchmarkResult(
            pipeline_name="x", corpus_name="y",
        )
        assert build_pipeline_steps_table_html(bench) == ""

    def test_success_rate_cell_colored(self) -> None:
        bench = _make_bench()
        html = build_pipeline_steps_table_html(bench)
        # Le gradient utilise des couleurs hex
        assert "background:#" in html


# ──────────────────────────────────────────────────────────────────────────
# 3. Document autonome
# ──────────────────────────────────────────────────────────────────────────


class TestStandaloneDocument:
    def test_doctype_present(self) -> None:
        bench = _make_bench()
        html = build_pipeline_report_html(bench)
        assert html.startswith("<!doctype html>")

    def test_html_head_body_structure(self) -> None:
        bench = _make_bench()
        html = build_pipeline_report_html(bench)
        assert "<html" in html
        assert "<head>" in html
        assert "<body>" in html
        assert "</html>" in html

    def test_styles_inline(self) -> None:
        bench = _make_bench()
        html = build_pipeline_report_html(bench)
        assert "<style>" in html
        # Au moins une règle CSS
        assert "body" in html
        assert "font-family" in html

    def test_title_includes_pipeline_name(self) -> None:
        bench = _make_bench()
        html = build_pipeline_report_html(bench)
        assert "<title>" in html
        assert "ocr_then_fix" in html

    def test_lang_attribute(self) -> None:
        bench = _make_bench()
        html_fr = build_pipeline_report_html(bench, lang="fr")
        html_en = build_pipeline_report_html(bench, lang="en")
        assert 'lang="fr"' in html_fr
        assert 'lang="en"' in html_en

    def test_summary_and_steps_included(self) -> None:
        bench = _make_bench()
        html = build_pipeline_report_html(bench)
        # Le document contient les deux blocs
        assert "ocr_then_fix" in html
        assert "ocr" in html
        assert "rewrite" in html


# ──────────────────────────────────────────────────────────────────────────
# 4. Anti-injection HTML
# ──────────────────────────────────────────────────────────────────────────


class TestAntiInjection:
    def test_pipeline_name_escaped(self) -> None:
        bench = PipelineBenchmarkResult(
            pipeline_name="<script>alert(1)</script>",
            corpus_name="demo",
        )
        html = build_pipeline_summary_html(bench)
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_corpus_name_escaped(self) -> None:
        bench = PipelineBenchmarkResult(
            pipeline_name="p",
            corpus_name="<img src=x onerror=alert(1)>",
        )
        html = build_pipeline_report_html(bench)
        assert "<img src=x" not in html
        assert "&lt;img" in html

    def test_step_name_escaped(self) -> None:
        bench = PipelineBenchmarkResult(
            pipeline_name="p", corpus_name="c",
        )
        bench.per_step_aggregates = [
            StepAggregate(
                step_name="<script>", n_docs=1, n_succeeded=1,
                duration_seconds_mean=0.1, duration_seconds_median=0.1,
            ),
        ]
        html = build_pipeline_steps_table_html(bench)
        assert "<script>" not in html.replace(
            "<script>alert", "@@@",  # ne devrait pas être présent de toute façon
        )
        assert "&lt;script&gt;" in html

    def test_label_escaped_via_i18n(self) -> None:
        bench = _make_bench()
        labels = {"pipeline_summary_title": "<b>X</b>"}
        html = build_pipeline_summary_html(bench, labels=labels)
        assert "<b>X</b>" not in html
        assert "&lt;b&gt;X&lt;/b&gt;" in html


# ──────────────────────────────────────────────────────────────────────────
# 5. Complétude i18n
# ──────────────────────────────────────────────────────────────────────────


class TestI18nCompleteness:
    def _load(self, lang: str) -> dict:
        path = (
            Path(__file__).parent.parent
            / "picarones" / "report" / "i18n" / f"{lang}.json"
        )
        return json.loads(path.read_text(encoding="utf-8"))

    def test_all_pipeline_keys_present_fr(self) -> None:
        d = self._load("fr")
        required = (
            "pipeline_report_title", "pipeline_report_note",
            "pipeline_summary_title", "pipeline_name_label",
            "pipeline_corpus_label", "pipeline_n_docs_label",
            "pipeline_succeeded_label", "pipeline_failed_label",
            "pipeline_duration_label", "pipeline_steps_title",
            "pipeline_step_name_label", "pipeline_success_rate_label",
            "pipeline_duration_mean_label",
            "pipeline_duration_median_label",
            "pipeline_junction_metrics_label",
            "pipeline_error_breakdown_label",
            "pipeline_docs_short",
        )
        for key in required:
            assert key in d, f"manque clé FR : {key}"

    def test_all_pipeline_keys_present_en(self) -> None:
        d_fr = self._load("fr")
        d_en = self._load("en")
        for key in d_fr:
            if key.startswith("pipeline_"):
                assert key in d_en, f"manque clé EN : {key}"
