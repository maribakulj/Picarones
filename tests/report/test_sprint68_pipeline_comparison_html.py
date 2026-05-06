"""Tests Sprint 68 — vue HTML de comparaison de N pipelines.

Couvre :

1. ``RankingSpec`` : ``display_label`` auto / explicite.
2. ``build_pipeline_ranking_table_html`` :
   - tableau rang / pipeline / valeur, ordre cohérent
   - pipelines sans valeur en queue avec tirets
   - cellule de rang colorée (gradient vert→rouge)
   - vide si la comparaison ne contient aucune pipeline
3. ``build_pipeline_gain_table_html`` :
   - tableau pipeline / valeur / absolute / relative
   - baseline marquée explicitement
   - couleur cellule favorable / défavorable selon
     ``higher_is_better``
   - baseline inconnue → chaîne vide
4. ``build_pipeline_comparison_summary_html`` : corpus, n_docs,
   n_pipelines, durée, mini-résumé par pipeline.
5. ``build_pipeline_comparison_report_html`` :
   - document HTML autonome (doctype, head, body, styles)
   - titre, lang attribute FR/EN
   - rankings affichés si ranking_specs fourni
   - gain table affiché uniquement si baseline_pipeline fourni
6. Anti-injection : pipeline name, corpus, labels.
7. Complétude i18n : nouvelles clés ``pipeline_*`` présentes
   en FR et EN.
"""

from __future__ import annotations

import json
from pathlib import Path

from picarones.core.modules import ArtifactType
from picarones.measurements.pipeline_benchmark import (
    PipelineBenchmarkResult,
    StepAggregate,
)
from picarones.measurements.pipeline_comparison import PipelineComparisonResult
from picarones.report.pipeline_render import (
    RankingSpec,
    build_pipeline_comparison_report_html,
    build_pipeline_comparison_summary_html,
    build_pipeline_gain_table_html,
    build_pipeline_ranking_table_html,
)


def _make_bench(name: str, cer_mean: float, n: int = 10) -> PipelineBenchmarkResult:
    bench = PipelineBenchmarkResult(
        pipeline_name=name, corpus_name="demo",
        n_docs=n, total_duration_seconds=1.0,
    )

    class _PR:
        def __init__(self, ok): self._ok = ok
        @property
        def succeeded(self): return self._ok
    bench.per_doc_results = [_PR(True)] * n
    bench.per_step_aggregates = [
        StepAggregate(
            step_name="ocr", n_docs=n, n_succeeded=n, n_failed=0,
            duration_seconds_total=0.5, duration_seconds_mean=0.05,
            duration_seconds_median=0.05,
            junction_metrics={
                "text": {
                    "cer": {"mean": cer_mean, "median": cer_mean, "n": n},
                },
            },
        ),
    ]
    return bench


def _make_comparison(
    pipelines: list[tuple[str, float]],
) -> PipelineComparisonResult:
    """Crée une comparaison avec pipelines = [(name, cer_mean), ...]."""
    comparison = PipelineComparisonResult(
        corpus_name="demo",
        n_docs=10,
        total_duration_seconds=3.0,
    )
    for name, cer in pipelines:
        comparison.per_pipeline[name] = _make_bench(name, cer)
    return comparison


# ──────────────────────────────────────────────────────────────────────────
# 1. RankingSpec
# ──────────────────────────────────────────────────────────────────────────


class TestRankingSpec:
    def test_display_label_default(self) -> None:
        spec = RankingSpec(ArtifactType.TEXT, "cer")
        # Phase 4-bis : ``ArtifactType.TEXT.value`` est désormais
        # ``"raw_text"`` (alias canonique vers ``RAW_TEXT``).
        assert spec.display_label == "raw_text.cer"

    def test_display_label_explicit(self) -> None:
        spec = RankingSpec(ArtifactType.TEXT, "cer", label="CER global")
        assert spec.display_label == "CER global"

    def test_higher_is_better_default_false(self) -> None:
        spec = RankingSpec(ArtifactType.TEXT, "cer")
        assert spec.higher_is_better is False


# ──────────────────────────────────────────────────────────────────────────
# 2. ranking table
# ──────────────────────────────────────────────────────────────────────────


class TestRankingTable:
    def test_orders_by_metric_ascending(self) -> None:
        comparison = _make_comparison([
            ("alpha", 0.20),
            ("beta", 0.05),
            ("gamma", 0.10),
        ])
        spec = RankingSpec(ArtifactType.TEXT, "cer")
        html = build_pipeline_ranking_table_html(comparison, spec)
        # beta (CER 0.05) doit apparaître avant alpha (CER 0.20)
        idx_beta = html.find("beta")
        idx_alpha = html.find("alpha")
        idx_gamma = html.find("gamma")
        assert 0 < idx_beta < idx_gamma < idx_alpha

    def test_higher_is_better_reverses(self) -> None:
        comparison = _make_comparison([
            ("alpha", 0.20),
            ("beta", 0.80),
        ])
        spec = RankingSpec(
            ArtifactType.TEXT, "cer", higher_is_better=True,
        )
        html = build_pipeline_ranking_table_html(comparison, spec)
        # beta (0.80) en premier puisqu'on inverse
        idx_beta = html.find("beta")
        idx_alpha = html.find("alpha")
        assert idx_beta < idx_alpha

    def test_pipelines_without_metric_in_queue(self) -> None:
        # Pipeline "bad" sans CER (aucun step n'a tourné)
        comparison = _make_comparison([("alpha", 0.10)])
        comparison.per_pipeline["bad"] = PipelineBenchmarkResult(
            pipeline_name="bad", corpus_name="demo",
        )
        spec = RankingSpec(ArtifactType.TEXT, "cer")
        html = build_pipeline_ranking_table_html(comparison, spec)
        idx_bad = html.find("bad")
        idx_alpha = html.find("alpha")
        assert 0 < idx_alpha < idx_bad
        # Le pipeline sans valeur affiche un tiret
        assert "—" in html

    def test_rank_cell_colored(self) -> None:
        comparison = _make_comparison([
            ("a", 0.1), ("b", 0.2), ("c", 0.3),
        ])
        spec = RankingSpec(ArtifactType.TEXT, "cer")
        html = build_pipeline_ranking_table_html(comparison, spec)
        assert "background:#" in html

    def test_empty_comparison_returns_empty(self) -> None:
        comparison = PipelineComparisonResult(corpus_name="empty")
        spec = RankingSpec(ArtifactType.TEXT, "cer")
        assert build_pipeline_ranking_table_html(comparison, spec) == ""

    def test_uses_display_label_in_title(self) -> None:
        comparison = _make_comparison([("alpha", 0.1)])
        spec = RankingSpec(
            ArtifactType.TEXT, "cer", label="Mon Label",
        )
        html = build_pipeline_ranking_table_html(comparison, spec)
        assert "Mon Label" in html


# ──────────────────────────────────────────────────────────────────────────
# 3. gain table
# ──────────────────────────────────────────────────────────────────────────


class TestGainTable:
    def test_baseline_marked(self) -> None:
        comparison = _make_comparison([
            ("baseline", 0.20), ("better", 0.10),
        ])
        spec = RankingSpec(ArtifactType.TEXT, "cer")
        html = build_pipeline_gain_table_html(
            comparison, spec, baseline_pipeline="baseline",
        )
        assert "(référence)" in html
        # Les deux pipelines apparaissent
        assert "baseline" in html
        assert "better" in html

    def test_gain_absolute_and_relative(self) -> None:
        comparison = _make_comparison([
            ("baseline", 0.20), ("better", 0.10),
        ])
        spec = RankingSpec(ArtifactType.TEXT, "cer")
        html = build_pipeline_gain_table_html(
            comparison, spec, baseline_pipeline="baseline",
        )
        # better : -0.1000 absolute, -50% relative
        assert "-0.1000" in html
        assert "-50.0%" in html

    def test_color_favorable_when_lower_better(self) -> None:
        # CER baisse → favorable → cellule verte (#cfe8cf)
        comparison = _make_comparison([
            ("baseline", 0.20), ("better", 0.05),
        ])
        spec = RankingSpec(ArtifactType.TEXT, "cer")
        html = build_pipeline_gain_table_html(
            comparison, spec, baseline_pipeline="baseline",
        )
        assert "#cfe8cf" in html

    def test_color_unfavorable_when_lower_better(self) -> None:
        # CER monte → défavorable → cellule rouge (#f4cfcf)
        comparison = _make_comparison([
            ("baseline", 0.10), ("worse", 0.30),
        ])
        spec = RankingSpec(ArtifactType.TEXT, "cer")
        html = build_pipeline_gain_table_html(
            comparison, spec, baseline_pipeline="baseline",
        )
        assert "#f4cfcf" in html

    def test_unknown_baseline_returns_empty(self) -> None:
        comparison = _make_comparison([("alpha", 0.1)])
        spec = RankingSpec(ArtifactType.TEXT, "cer")
        html = build_pipeline_gain_table_html(
            comparison, spec, baseline_pipeline="nonexistent",
        )
        assert html == ""


# ──────────────────────────────────────────────────────────────────────────
# 4. comparison summary
# ──────────────────────────────────────────────────────────────────────────


class TestComparisonSummary:
    def test_renders_corpus_and_counts(self) -> None:
        comparison = _make_comparison([
            ("a", 0.1), ("b", 0.2),
        ])
        html = build_pipeline_comparison_summary_html(comparison)
        assert "demo" in html
        assert "10" in html  # n_docs
        # 2 pipelines
        assert ">2<" in html

    def test_per_pipeline_mini_summary(self) -> None:
        comparison = _make_comparison([
            ("a", 0.1), ("b", 0.2),
        ])
        html = build_pipeline_comparison_summary_html(comparison)
        # Mini-résumé : nom (n_succeeded/n_docs)
        assert "a" in html
        assert "b" in html
        assert "10/10" in html


# ──────────────────────────────────────────────────────────────────────────
# 5. document autonome
# ──────────────────────────────────────────────────────────────────────────


class TestComparisonReport:
    def test_doctype_and_structure(self) -> None:
        comparison = _make_comparison([("a", 0.1)])
        html = build_pipeline_comparison_report_html(comparison)
        assert html.startswith("<!doctype html>")
        assert "<html" in html
        assert "<head>" in html
        assert "<body>" in html
        assert "</html>" in html

    def test_lang_attribute(self) -> None:
        comparison = _make_comparison([("a", 0.1)])
        html_fr = build_pipeline_comparison_report_html(
            comparison, lang="fr",
        )
        html_en = build_pipeline_comparison_report_html(
            comparison, lang="en",
        )
        assert 'lang="fr"' in html_fr
        assert 'lang="en"' in html_en

    def test_rankings_displayed_when_specs_provided(self) -> None:
        comparison = _make_comparison([
            ("a", 0.20), ("b", 0.05),
        ])
        specs = [RankingSpec(ArtifactType.TEXT, "cer", label="CER")]
        html = build_pipeline_comparison_report_html(
            comparison, ranking_specs=specs,
        )
        assert "Classement par CER" in html

    def test_no_rankings_without_specs(self) -> None:
        comparison = _make_comparison([("a", 0.1)])
        html = build_pipeline_comparison_report_html(comparison)
        # Pas de tableau de classement sans ranking_specs
        assert "Classement par" not in html

    def test_gain_table_only_with_baseline(self) -> None:
        comparison = _make_comparison([
            ("baseline", 0.20), ("better", 0.10),
        ])
        specs = [RankingSpec(ArtifactType.TEXT, "cer")]
        # Sans baseline : pas de gain table
        html_no_baseline = build_pipeline_comparison_report_html(
            comparison, ranking_specs=specs,
        )
        assert "Gain vs" not in html_no_baseline
        # Avec baseline : gain table présent
        html_with_baseline = build_pipeline_comparison_report_html(
            comparison, ranking_specs=specs,
            baseline_pipeline="baseline",
        )
        assert "Gain vs" in html_with_baseline


# ──────────────────────────────────────────────────────────────────────────
# 6. Anti-injection
# ──────────────────────────────────────────────────────────────────────────


class TestAntiInjection:
    def test_pipeline_name_escaped_in_ranking(self) -> None:
        comparison = _make_comparison([
            ("<script>alert(1)</script>", 0.1),
        ])
        spec = RankingSpec(ArtifactType.TEXT, "cer")
        html = build_pipeline_ranking_table_html(comparison, spec)
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html

    def test_corpus_name_escaped_in_summary(self) -> None:
        comparison = PipelineComparisonResult(
            corpus_name="<img src=x onerror=alert(1)>",
        )
        html = build_pipeline_comparison_summary_html(comparison)
        assert "<img src=x" not in html
        assert "&lt;img" in html

    def test_label_via_i18n_escaped(self) -> None:
        comparison = _make_comparison([("a", 0.1)])
        spec = RankingSpec(ArtifactType.TEXT, "cer")
        labels = {"pipeline_ranking_title": "<b>Hack</b>"}
        html = build_pipeline_ranking_table_html(
            comparison, spec, labels=labels,
        )
        assert "<b>Hack</b>" not in html
        assert "&lt;b&gt;Hack&lt;/b&gt;" in html


# ──────────────────────────────────────────────────────────────────────────
# 7. Complétude i18n
# ──────────────────────────────────────────────────────────────────────────


class TestI18nCompleteness:
    def _load(self, lang: str) -> dict:
        path = (
            Path(__file__).parent.parent.parent
            / "picarones" / "report" / "i18n" / f"{lang}.json"
        )
        return json.loads(path.read_text(encoding="utf-8"))

    def test_new_keys_present_fr(self) -> None:
        d = self._load("fr")
        required = (
            "pipeline_comparison_report_title",
            "pipeline_comparison_report_note",
            "pipeline_comparison_summary_title",
            "pipeline_n_pipelines_label",
            "pipeline_n_pipelines_short",
            "pipeline_per_pipeline_label",
            "pipeline_ranking_title", "pipeline_rank_label",
            "pipeline_value_label",
            "pipeline_gain_title",
            "pipeline_gain_absolute_label",
            "pipeline_gain_relative_label",
            "pipeline_baseline_marker",
        )
        for key in required:
            assert key in d, f"manque clé FR : {key}"

    def test_new_keys_present_en(self) -> None:
        d_fr = self._load("fr")
        d_en = self._load("en")
        for key in d_fr:
            if key.startswith("pipeline_"):
                assert key in d_en, f"manque clé EN : {key}"
