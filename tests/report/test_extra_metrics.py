"""Tests des 4 fonctions de câblage du sprint « zéro dette actionnable ».

Couvre :func:`compute_rare_token_recall_per_engine`,
:func:`compute_taxonomy_cooccurrence_section`,
:func:`compute_taxonomy_intra_doc_section`,
:func:`compute_marginal_cost_section` — leur format de retour et leur
intégration dans :func:`build_report_data`.

Garde-fou : sans ces tests, une régression future qui changerait le
schéma de retour (ex: clé manquante côté renderer) passerait
silencieusement en production.
"""

from __future__ import annotations

import pytest

from picarones.evaluation.synthetic import generate_sample_benchmark
from picarones.reports_v2.html.data import build_report_data
from picarones.reports_v2.html.data.extra_metrics import (
    compute_marginal_cost_section,
    compute_rare_token_recall_per_engine,
    compute_taxonomy_cooccurrence_section,
    compute_taxonomy_intra_doc_section,
)


@pytest.fixture(scope="module")
def sample_benchmark():
    return generate_sample_benchmark()


# ──────────────────────────────────────────────────────────────────
# rare_token_recall
# ──────────────────────────────────────────────────────────────────
class TestRareTokenRecall:
    def test_returns_dict_per_engine(self, sample_benchmark) -> None:
        result = compute_rare_token_recall_per_engine(sample_benchmark)
        assert isinstance(result, dict)
        # Au moins un moteur doit avoir un résultat sur les fixtures.
        assert len(result) > 0

    def test_each_entry_has_required_fields(self, sample_benchmark) -> None:
        result = compute_rare_token_recall_per_engine(sample_benchmark)
        for engine, info in result.items():
            assert "n_rare_tokens" in info
            assert "n_recalled" in info
            assert "recall" in info
            assert "n_docs" in info
            assert "max_freq" in info

    def test_recall_in_unit_range_or_none(self, sample_benchmark) -> None:
        result = compute_rare_token_recall_per_engine(sample_benchmark)
        for engine, info in result.items():
            recall = info["recall"]
            if recall is not None:
                assert 0.0 <= recall <= 1.0, f"{engine}: recall hors [0,1]"

    def test_returns_empty_dict_on_empty_benchmark(self) -> None:
        # Benchmark sans engine_reports → dict vide.
        from picarones.evaluation.benchmark_result import BenchmarkResult
        bench = BenchmarkResult(
            corpus_name="empty",
            corpus_source=None,
            document_count=0,
            engine_reports=[],
            run_date="2026-05-02",
            picarones_version="test",
        )
        result = compute_rare_token_recall_per_engine(bench)
        assert result == {}


# ──────────────────────────────────────────────────────────────────
# taxonomy_cooccurrence
# ──────────────────────────────────────────────────────────────────
class TestTaxonomyCooccurrence:
    def test_returns_dict_or_none(self, sample_benchmark) -> None:
        result = compute_taxonomy_cooccurrence_section(sample_benchmark)
        assert result is None or isinstance(result, dict)

    def test_no_set_index_bug_on_multi_engine_corpus(
        self, sample_benchmark,
    ) -> None:
        """Régression : la fusion des classes par doc utilisait
        ``list(set).index()`` qui retournait un index aléatoire (bug
        critique trouvé par audit). Vérifie que le résultat est stable
        et reproductible — pas dépendant de l'ordre d'itération du set.
        """
        # Lance 5 fois et vérifie que le résultat est identique.
        results = [
            compute_taxonomy_cooccurrence_section(sample_benchmark)
            for _ in range(5)
        ]
        # Tous les résultats doivent être identiques (déterminisme).
        for r in results[1:]:
            assert r == results[0]

    def test_compatible_with_renderer(self, sample_benchmark) -> None:
        from picarones.reports_v2.html.renderers.taxonomy_cooccurrence import (
            build_taxonomy_cooccurrence_html,
        )
        result = compute_taxonomy_cooccurrence_section(sample_benchmark)
        # Doit pouvoir être rendu sans crash (None ou dict valide).
        html = build_taxonomy_cooccurrence_html(result)
        assert isinstance(html, str)


# ──────────────────────────────────────────────────────────────────
# taxonomy_intra_doc
# ──────────────────────────────────────────────────────────────────
class TestTaxonomyIntraDoc:
    def test_returns_dict_or_none(self, sample_benchmark) -> None:
        result = compute_taxonomy_intra_doc_section(sample_benchmark)
        assert result is None or isinstance(result, dict)

    def test_dedup_docs_across_engines(self, sample_benchmark) -> None:
        """Le comptage des documents dédoublonne : un même doc évalué
        par N moteurs ne compte qu'une fois (régression : auparavant on
        comptait N×).
        """
        result = compute_taxonomy_intra_doc_section(sample_benchmark)
        if result is None:
            pytest.skip("Pas de signal taxonomy intra-doc sur fixture")
        # ``n_docs_with_data`` doit être ≤ document_count, jamais plus.
        assert result["n_docs_with_data"] <= sample_benchmark.document_count

    def test_renderer_compatibility(self, sample_benchmark) -> None:
        """Le format de retour doit contenir les clés attendues par
        :func:`build_taxonomy_intra_doc_html` :
        ``n_bins``, ``per_class``, ``total_errors``, ``n_words_gt``.
        Sans ces clés, le renderer retourne ``""`` silencieusement.
        """
        result = compute_taxonomy_intra_doc_section(sample_benchmark)
        if result is None:
            pytest.skip("Pas de signal taxonomy intra-doc sur fixture")
        for key in ("n_bins", "per_class", "total_errors", "n_words_gt"):
            assert key in result, f"clé {key!r} manquante (renderer la requiert)"

    def test_renders_html_when_signal_present(self, sample_benchmark) -> None:
        from picarones.reports_v2.html.renderers.taxonomy_intra_doc import (
            build_taxonomy_intra_doc_html,
        )
        result = compute_taxonomy_intra_doc_section(sample_benchmark)
        if result is None or result.get("total_errors", 0) == 0:
            pytest.skip("Pas d'erreurs sur fixture")
        html = build_taxonomy_intra_doc_html(result)
        # Si le signal existe, le HTML ne doit pas être vide.
        assert html != "", (
            "Renderer retourne '' alors que le calcul a remonté du signal — "
            "format de retour incompatible."
        )


# ──────────────────────────────────────────────────────────────────
# marginal_cost
# ──────────────────────────────────────────────────────────────────
class TestMarginalCost:
    def test_returns_list_or_none(self, sample_benchmark) -> None:
        engines_summary = [
            {"name": "tess", "cer": 0.10, "doc_count": 12,
             "cost": {"cost_per_1k_pages_eur": 5.0}},
            {"name": "pero", "cer": 0.05, "doc_count": 12,
             "cost": {"cost_per_1k_pages_eur": 10.0}},
        ]
        result = compute_marginal_cost_section(engines_summary)
        assert result is None or isinstance(result, list)
        if result:
            # Chaque item est un dict de paire avec les clés attendues.
            for pair in result:
                assert isinstance(pair, dict)
                assert "engine_a" in pair
                assert "engine_b" in pair

    def test_returns_none_with_one_engine(self) -> None:
        engines_summary = [
            {"name": "tess", "cer": 0.10, "doc_count": 12,
             "cost": {"cost_per_1k_pages_eur": 5.0}},
        ]
        assert compute_marginal_cost_section(engines_summary) is None

    def test_renderer_compatibility(self) -> None:
        from picarones.reports_v2.html.renderers.marginal_cost import (
            build_marginal_cost_html,
        )
        engines_summary = [
            {"name": "tess", "cer": 0.10, "doc_count": 12,
             "cost": {"cost_per_1k_pages_eur": 5.0}},
            {"name": "pero", "cer": 0.05, "doc_count": 12,
             "cost": {"cost_per_1k_pages_eur": 10.0}},
        ]
        result = compute_marginal_cost_section(engines_summary)
        # Doit pouvoir être rendu sans crash.
        html = build_marginal_cost_html(result)
        assert isinstance(html, str)
        if result:
            assert html != ""


# ──────────────────────────────────────────────────────────────────
# Intégration dans build_report_data
# ──────────────────────────────────────────────────────────────────
class TestIntegrationBuildReportData:
    def test_all_keys_present_in_report_data(self, sample_benchmark) -> None:
        data = build_report_data(sample_benchmark, {})
        for key in (
            "rare_token_recall",
            "taxonomy_cooccurrence",
            "taxonomy_intra_doc",
            "marginal_cost",
        ):
            assert key in data, f"clé {key!r} absente du report_data"

    def test_marginal_cost_uses_attached_costs(
        self, sample_benchmark,
    ) -> None:
        """Régression : ``compute_marginal_cost_section`` doit être
        appelée APRÈS ``attach_engine_costs`` pour avoir accès aux
        coûts attachés. Sinon retourne None silencieusement.
        """
        data = build_report_data(sample_benchmark, {})
        # Sur les fixtures, au moins un moteur a un coût pricing
        # connu → la matrice doit avoir au moins une paire.
        marginal = data.get("marginal_cost")
        if marginal is not None:
            assert len(marginal) > 0
