"""Tests Sprint 7 — Rapport HTML v2 et analyses avancées.

Classes de tests
----------------
TestBootstrapCI           (7 tests)  — intervalles de confiance par bootstrap
TestWilcoxonTest          (10 tests) — test de Wilcoxon signé-rangé
TestPairwiseStats         (6 tests)  — matrice de tests par paires
TestReliabilityCurve      (7 tests)  — courbes de fiabilité
TestVennData              (8 tests)  — diagramme de Venn 2 et 3 ensembles
TestErrorClustering       (8 tests)  — clustering des patterns d'erreurs
TestCorrelationMatrix     (8 tests)  — matrice de corrélation
TestDifficultyScore       (10 tests) — score de difficulté intrinsèque par document
TestAllDifficulties       (6 tests)  — compute_all_difficulties sur un corpus
TestReportDataSprint7     (12 tests) — _build_report_data contient les nouvelles clés
TestHTMLSprint7Features   (10 tests) — HTML généré contient les nouvelles fonctionnalités
"""

from __future__ import annotations

import math
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_benchmark_s7():
    from picarones.fixtures import generate_sample_benchmark
    return generate_sample_benchmark(n_docs=8, seed=7)


@pytest.fixture
def report_data_s7(sample_benchmark_s7):
    from picarones.report.generator import _build_report_data
    imgs = sample_benchmark_s7.metadata.get("_images_b64", {})
    return _build_report_data(sample_benchmark_s7, imgs)


@pytest.fixture
def html_s7(sample_benchmark_s7):
    from picarones.report.generator import ReportGenerator
    import tempfile, pathlib
    gen = ReportGenerator(sample_benchmark_s7)
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        path = gen.generate(f.name)
    return pathlib.Path(path).read_text(encoding="utf-8")


# ===========================================================================
# TestBootstrapCI
# ===========================================================================

class TestBootstrapCI:
    def test_returns_tuple_of_two(self):
        from picarones.core.statistics import bootstrap_ci
        result = bootstrap_ci([0.1, 0.2, 0.3])
        assert isinstance(result, tuple) and len(result) == 2

    def test_lower_le_upper(self):
        from picarones.core.statistics import bootstrap_ci
        lo, hi = bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5])
        assert lo <= hi

    def test_ci_contains_mean(self):
        from picarones.core.statistics import bootstrap_ci
        values = [0.1, 0.15, 0.2, 0.12, 0.18, 0.13, 0.17]
        lo, hi = bootstrap_ci(values)
        mean = sum(values) / len(values)
        assert lo <= mean <= hi

    def test_empty_returns_zeros(self):
        from picarones.core.statistics import bootstrap_ci
        lo, hi = bootstrap_ci([])
        assert lo == 0.0 and hi == 0.0

    def test_single_value(self):
        from picarones.core.statistics import bootstrap_ci
        lo, hi = bootstrap_ci([0.25])
        assert lo <= 0.25 <= hi

    def test_reproducible_with_seed(self):
        from picarones.core.statistics import bootstrap_ci
        vals = [0.1, 0.2, 0.3, 0.15, 0.25]
        r1 = bootstrap_ci(vals, seed=1)
        r2 = bootstrap_ci(vals, seed=1)
        assert r1 == r2

    def test_wider_with_more_variance(self):
        from picarones.core.statistics import bootstrap_ci
        narrow = [0.10, 0.11, 0.10, 0.11, 0.10]
        wide   = [0.01, 0.50, 0.02, 0.49, 0.01]
        lo_n, hi_n = bootstrap_ci(narrow, n_iter=500)
        lo_w, hi_w = bootstrap_ci(wide,   n_iter=500)
        assert (hi_w - lo_w) > (hi_n - lo_n)


# ===========================================================================
# TestWilcoxonTest
# ===========================================================================

class TestWilcoxonTest:
    def test_returns_dict_with_keys(self):
        from picarones.core.statistics import wilcoxon_test
        r = wilcoxon_test([0.1]*5, [0.1]*5)
        assert "statistic" in r
        assert "p_value" in r
        assert "significant" in r
        assert "interpretation" in r

    def test_identical_series_not_significant(self):
        from picarones.core.statistics import wilcoxon_test
        vals = [0.1, 0.2, 0.3, 0.15, 0.05]
        r = wilcoxon_test(vals, vals)
        assert not r["significant"]

    def test_clearly_different_series_significant(self):
        from picarones.core.statistics import wilcoxon_test
        a = [0.01]*12
        b = [0.80]*12
        r = wilcoxon_test(a, b)
        assert r["significant"]
        assert r["p_value"] < 0.05

    def test_p_value_in_range(self):
        from picarones.core.statistics import wilcoxon_test
        a = [0.1, 0.15, 0.2, 0.08]
        b = [0.2, 0.25, 0.3, 0.18]
        r = wilcoxon_test(a, b)
        assert 0.0 <= r["p_value"] <= 1.0

    def test_interpretation_is_string(self):
        from picarones.core.statistics import wilcoxon_test
        r = wilcoxon_test([0.1, 0.2], [0.1, 0.2])
        assert isinstance(r["interpretation"], str) and len(r["interpretation"]) > 10

    def test_n_pairs_correct(self):
        from picarones.core.statistics import wilcoxon_test
        r = wilcoxon_test([0.1, 0.2, 0.3], [0.1, 0.2, 0.3])
        # tous les diffs = 0, filtrés en mode wilcox
        assert r["n_pairs"] == 0

    def test_mismatched_lengths_raises(self):
        from picarones.core.statistics import wilcoxon_test
        with pytest.raises(ValueError):
            wilcoxon_test([0.1, 0.2], [0.1])

    def test_w_plus_w_minus_present(self):
        from picarones.core.statistics import wilcoxon_test
        a = [0.1, 0.2, 0.3, 0.15, 0.25, 0.18, 0.12, 0.22, 0.08, 0.27]
        b = [0.2, 0.3, 0.4, 0.25, 0.35, 0.28, 0.22, 0.32, 0.18, 0.37]
        r = wilcoxon_test(a, b)
        assert "W_plus" in r and "W_minus" in r

    def test_significant_larger_sample(self):
        from picarones.core.statistics import wilcoxon_test
        import random
        rng = random.Random(0)
        a = [rng.uniform(0.0, 0.05) for _ in range(15)]
        b = [rng.uniform(0.3, 0.7)  for _ in range(15)]
        r = wilcoxon_test(a, b)
        assert r["significant"]

    def test_symmetry(self):
        from picarones.core.statistics import wilcoxon_test
        a = [0.1, 0.2, 0.3, 0.15, 0.25, 0.18, 0.22, 0.08, 0.27, 0.14]
        b = [0.2, 0.3, 0.4, 0.25, 0.35, 0.28, 0.32, 0.18, 0.37, 0.24]
        r_ab = wilcoxon_test(a, b)
        r_ba = wilcoxon_test(b, a)
        assert r_ab["p_value"] == pytest.approx(r_ba["p_value"], abs=1e-6)
        assert r_ab["significant"] == r_ba["significant"]


# ===========================================================================
# TestPairwiseStats
# ===========================================================================

class TestPairwiseStats:
    def test_returns_list(self):
        from picarones.core.statistics import compute_pairwise_stats
        r = compute_pairwise_stats({"A": [0.1, 0.2], "B": [0.3, 0.4]})
        assert isinstance(r, list)

    def test_correct_pair_count_2_engines(self):
        from picarones.core.statistics import compute_pairwise_stats
        r = compute_pairwise_stats({"A": [0.1]*5, "B": [0.2]*5})
        assert len(r) == 1

    def test_correct_pair_count_3_engines(self):
        from picarones.core.statistics import compute_pairwise_stats
        r = compute_pairwise_stats({
            "A": [0.1]*5, "B": [0.2]*5, "C": [0.3]*5
        })
        assert len(r) == 3

    def test_pair_has_engine_names(self):
        from picarones.core.statistics import compute_pairwise_stats
        r = compute_pairwise_stats({"A": [0.1]*5, "B": [0.2]*5})
        assert r[0]["engine_a"] in ["A", "B"]
        assert r[0]["engine_b"] in ["A", "B"]

    def test_pair_has_p_value(self):
        from picarones.core.statistics import compute_pairwise_stats
        r = compute_pairwise_stats({"A": [0.1]*5, "B": [0.2]*5})
        assert "p_value" in r[0]

    def test_single_engine_returns_empty(self):
        from picarones.core.statistics import compute_pairwise_stats
        r = compute_pairwise_stats({"A": [0.1]*5})
        assert r == []


# ===========================================================================
# TestReliabilityCurve
# ===========================================================================

class TestReliabilityCurve:
    def test_returns_list(self):
        from picarones.core.statistics import compute_reliability_curve
        r = compute_reliability_curve([0.1, 0.2, 0.3])
        assert isinstance(r, list)

    def test_correct_number_of_steps(self):
        from picarones.core.statistics import compute_reliability_curve
        r = compute_reliability_curve([0.1]*10, steps=5)
        assert len(r) == 5

    def test_pct_docs_increases(self):
        from picarones.core.statistics import compute_reliability_curve
        r = compute_reliability_curve([0.1, 0.2, 0.3, 0.4, 0.5], steps=5)
        pcts = [p["pct_docs"] for p in r]
        assert pcts == sorted(pcts)

    def test_mean_cer_increases(self):
        from picarones.core.statistics import compute_reliability_curve
        r = compute_reliability_curve([0.05, 0.10, 0.20, 0.30, 0.50], steps=5)
        cers = [p["mean_cer"] for p in r]
        assert cers[0] <= cers[-1]

    def test_empty_returns_empty(self):
        from picarones.core.statistics import compute_reliability_curve
        assert compute_reliability_curve([]) == []

    def test_last_point_includes_all(self):
        from picarones.core.statistics import compute_reliability_curve
        vals = [0.1, 0.2, 0.3]
        r = compute_reliability_curve(vals, steps=4)
        last = r[-1]
        expected = sum(vals) / len(vals)
        assert last["mean_cer"] == pytest.approx(expected, rel=1e-4)

    def test_each_point_has_required_keys(self):
        from picarones.core.statistics import compute_reliability_curve
        r = compute_reliability_curve([0.1, 0.2, 0.3], steps=3)
        for p in r:
            assert "pct_docs" in p and "mean_cer" in p


# ===========================================================================
# TestVennData
# ===========================================================================

class TestVennData:
    def test_venn2_type(self):
        from picarones.core.statistics import compute_venn_data
        r = compute_venn_data({"A": {"e1","e2"}, "B": {"e2","e3"}})
        assert r["type"] == "venn2"

    def test_venn3_type(self):
        from picarones.core.statistics import compute_venn_data
        r = compute_venn_data({"A": {"e1"}, "B": {"e2"}, "C": {"e3"}})
        assert r["type"] == "venn3"

    def test_venn2_counts_correct(self):
        from picarones.core.statistics import compute_venn_data
        r = compute_venn_data({"A": {"e1","e2","e3"}, "B": {"e2","e3","e4"}})
        assert r["only_a"] == 1
        assert r["only_b"] == 1
        assert r["both"] == 2

    def test_venn2_disjoint(self):
        from picarones.core.statistics import compute_venn_data
        r = compute_venn_data({"A": {"e1"}, "B": {"e2"}})
        assert r["both"] == 0
        assert r["only_a"] == 1
        assert r["only_b"] == 1

    def test_venn2_subset(self):
        from picarones.core.statistics import compute_venn_data
        r = compute_venn_data({"A": {"e1","e2"}, "B": {"e1","e2","e3"}})
        assert r["only_a"] == 0

    def test_venn3_abc_count(self):
        from picarones.core.statistics import compute_venn_data
        shared = {"e1","e2"}
        r = compute_venn_data({"A": shared, "B": shared, "C": shared})
        assert r["abc"] == 2

    def test_empty_returns_empty(self):
        from picarones.core.statistics import compute_venn_data
        r = compute_venn_data({})
        assert r == {}

    def test_labels_present(self):
        from picarones.core.statistics import compute_venn_data
        r = compute_venn_data({"moteur_a": {"e1"}, "moteur_b": {"e2"}})
        assert r["label_a"] == "moteur_a"
        assert r["label_b"] == "moteur_b"


# ===========================================================================
# TestErrorClustering
# ===========================================================================

class TestErrorClustering:
    def _sample_data(self):
        return [
            {"engine": "tesseract", "gt": "maiſtre Froiſſart", "hypothesis": "maiftre Froiffart"},
            {"engine": "tesseract", "gt": "nostre seigneur", "hypothesis": "noltre leigneur"},
            {"engine": "pero", "gt": "regnoit en France", "hypothesis": "regnoit en France"},
            {"engine": "pero", "gt": "en l'an de grace", "hypothesis": "en l'an de grace"},
            {"engine": "mauvais", "gt": "icy commence le prologue", "hypothesis": "icy conmence le prologue"},
            {"engine": "mauvais", "gt": "par la grace de Dieu", "hypothesis": "par la grce de Dieu"},
        ]

    def test_returns_list(self):
        from picarones.core.statistics import cluster_errors
        result = cluster_errors(self._sample_data())
        assert isinstance(result, list)

    def test_max_clusters_respected(self):
        from picarones.core.statistics import cluster_errors
        result = cluster_errors(self._sample_data(), max_clusters=3)
        assert len(result) <= 3

    def test_cluster_has_required_keys(self):
        from picarones.core.statistics import cluster_errors
        result = cluster_errors(self._sample_data())
        if result:
            c = result[0]
            assert hasattr(c, "cluster_id")
            assert hasattr(c, "label")
            assert hasattr(c, "count")
            assert hasattr(c, "examples")

    def test_as_dict_method(self):
        from picarones.core.statistics import cluster_errors
        result = cluster_errors(self._sample_data())
        if result:
            d = result[0].as_dict()
            assert "cluster_id" in d
            assert "label" in d
            assert "count" in d
            assert "examples" in d

    def test_sorted_by_count_descending(self):
        from picarones.core.statistics import cluster_errors
        result = cluster_errors(self._sample_data())
        if len(result) >= 2:
            assert result[0].count >= result[1].count

    def test_examples_capped_at_5(self):
        from picarones.core.statistics import cluster_errors
        result = cluster_errors(self._sample_data())
        for c in result:
            assert len(c.as_dict()["examples"]) <= 5

    def test_empty_data_returns_empty(self):
        from picarones.core.statistics import cluster_errors
        result = cluster_errors([])
        assert result == []

    def test_cluster_id_unique(self):
        from picarones.core.statistics import cluster_errors
        result = cluster_errors(self._sample_data())
        ids = [c.cluster_id for c in result]
        assert len(ids) == len(set(ids))


# ===========================================================================
# TestCorrelationMatrix
# ===========================================================================

class TestCorrelationMatrix:
    def _sample_metrics(self):
        return [
            {"cer": 0.1, "wer": 0.2, "quality_score": 0.8},
            {"cer": 0.2, "wer": 0.35, "quality_score": 0.6},
            {"cer": 0.05, "wer": 0.1, "quality_score": 0.9},
            {"cer": 0.3, "wer": 0.5, "quality_score": 0.5},
            {"cer": 0.15, "wer": 0.25, "quality_score": 0.75},
        ]

    def test_returns_dict_with_labels_and_matrix(self):
        from picarones.core.statistics import compute_correlation_matrix
        r = compute_correlation_matrix(self._sample_metrics())
        assert "labels" in r and "matrix" in r

    def test_matrix_is_square(self):
        from picarones.core.statistics import compute_correlation_matrix
        r = compute_correlation_matrix(self._sample_metrics())
        n = len(r["labels"])
        assert len(r["matrix"]) == n
        for row in r["matrix"]:
            assert len(row) == n

    def test_diagonal_is_one(self):
        from picarones.core.statistics import compute_correlation_matrix
        r = compute_correlation_matrix(self._sample_metrics())
        for i in range(len(r["labels"])):
            assert r["matrix"][i][i] == pytest.approx(1.0)

    def test_cer_quality_negatively_correlated(self):
        from picarones.core.statistics import compute_correlation_matrix
        r = compute_correlation_matrix(self._sample_metrics())
        labels = r["labels"]
        if "cer" in labels and "quality_score" in labels:
            i = labels.index("cer")
            j = labels.index("quality_score")
            assert r["matrix"][i][j] < 0  # plus la qualité est bonne, plus le CER est bas

    def test_symmetric_matrix(self):
        from picarones.core.statistics import compute_correlation_matrix
        r = compute_correlation_matrix(self._sample_metrics())
        n = len(r["labels"])
        for i in range(n):
            for j in range(n):
                assert r["matrix"][i][j] == pytest.approx(r["matrix"][j][i], abs=1e-6)

    def test_empty_returns_empty(self):
        from picarones.core.statistics import compute_correlation_matrix
        r = compute_correlation_matrix([])
        assert r == {"labels": [], "matrix": []}

    def test_custom_metric_keys(self):
        from picarones.core.statistics import compute_correlation_matrix
        data = [{"a": 1.0, "b": 2.0, "c": 3.0}] * 5
        r = compute_correlation_matrix(data, metric_keys=["a", "b"])
        assert r["labels"] == ["a", "b"]

    def test_values_in_range(self):
        from picarones.core.statistics import compute_correlation_matrix
        r = compute_correlation_matrix(self._sample_metrics())
        for row in r["matrix"]:
            for v in row:
                assert -1.0 <= v <= 1.0


# ===========================================================================
# TestDifficultyScore
# ===========================================================================

class TestDifficultyScore:
    def test_returns_difficulty_score(self):
        from picarones.core.difficulty import compute_difficulty_score
        ds = compute_difficulty_score("doc1", "maiſtre Froiſſart", [0.1, 0.2, 0.3])
        from picarones.core.difficulty import DifficultyScore
        assert isinstance(ds, DifficultyScore)

    def test_score_in_range(self):
        from picarones.core.difficulty import compute_difficulty_score
        ds = compute_difficulty_score("doc1", "hello world", [0.1, 0.2])
        assert 0.0 <= ds.score <= 1.0

    def test_more_variance_higher_score(self):
        from picarones.core.difficulty import compute_difficulty_score
        low_var  = compute_difficulty_score("doc1", "hello", [0.1, 0.1, 0.1])
        high_var = compute_difficulty_score("doc1", "hello", [0.0, 0.5, 1.0])
        assert high_var.score > low_var.score

    def test_bad_quality_image_harder(self):
        from picarones.core.difficulty import compute_difficulty_score
        good_img = compute_difficulty_score("doc1", "hello", [0.1], image_quality_score=0.9)
        bad_img  = compute_difficulty_score("doc1", "hello", [0.1], image_quality_score=0.1)
        assert bad_img.score > good_img.score

    def test_special_chars_increase_difficulty(self):
        from picarones.core.difficulty import compute_difficulty_score
        plain    = compute_difficulty_score("doc1", "hello world plain text", [0.1])
        heritage = compute_difficulty_score("doc1", "maiſtre Froiſſart ꝑ &", [0.1])
        assert heritage.score > plain.score

    def test_components_present(self):
        from picarones.core.difficulty import compute_difficulty_score
        ds = compute_difficulty_score("doc1", "text", [0.1, 0.2])
        assert hasattr(ds, "variance_component")
        assert hasattr(ds, "quality_component")
        assert hasattr(ds, "density_component")

    def test_as_dict_has_doc_id(self):
        from picarones.core.difficulty import compute_difficulty_score
        ds = compute_difficulty_score("folio_001", "text", [0.1])
        d = ds.as_dict()
        assert d["doc_id"] == "folio_001"

    def test_as_dict_rounded(self):
        from picarones.core.difficulty import compute_difficulty_score
        ds = compute_difficulty_score("doc1", "text", [0.1])
        d = ds.as_dict()
        assert isinstance(d["score"], float)

    def test_no_engines_gives_low_variance(self):
        from picarones.core.difficulty import compute_difficulty_score
        ds = compute_difficulty_score("doc1", "text", [])
        assert ds.cer_variance == 0.0

    def test_difficulty_label(self):
        from picarones.core.difficulty import difficulty_label
        assert difficulty_label(0.1)  == "Facile"
        assert difficulty_label(0.35) == "Modéré"
        assert difficulty_label(0.6)  == "Difficile"
        assert difficulty_label(0.9)  == "Très difficile"


# ===========================================================================
# TestAllDifficulties
# ===========================================================================

class TestAllDifficulties:
    def test_returns_dict(self):
        from picarones.core.difficulty import compute_all_difficulties
        r = compute_all_difficulties(
            ["doc1", "doc2"],
            {"doc1": "hello", "doc2": "world"},
            {"doc1": {"A": 0.1}, "doc2": {"A": 0.2}},
        )
        assert isinstance(r, dict)

    def test_all_docs_present(self):
        from picarones.core.difficulty import compute_all_difficulties
        r = compute_all_difficulties(
            ["d1", "d2", "d3"],
            {"d1": "a", "d2": "b", "d3": "c"},
            {"d1": {"E": 0.1}, "d2": {"E": 0.2}, "d3": {"E": 0.3}},
        )
        assert set(r.keys()) == {"d1", "d2", "d3"}

    def test_scores_in_range(self):
        from picarones.core.difficulty import compute_all_difficulties
        r = compute_all_difficulties(
            ["d1", "d2"],
            {"d1": "maiſtre Jean", "d2": "simple text"},
            {"d1": {"A": 0.1, "B": 0.5}, "d2": {"A": 0.1, "B": 0.1}},
        )
        for ds in r.values():
            assert 0.0 <= ds.score <= 1.0

    def test_with_image_quality(self):
        from picarones.core.difficulty import compute_all_difficulties
        r = compute_all_difficulties(
            ["d1"],
            {"d1": "text"},
            {"d1": {"A": 0.1}},
            image_quality_map={"d1": 0.3},
        )
        assert "d1" in r
        # qualité dégradée → composante élevée
        assert r["d1"].quality_component > 0.5

    def test_empty_corpus(self):
        from picarones.core.difficulty import compute_all_difficulties
        r = compute_all_difficulties([], {}, {})
        assert r == {}

    def test_missing_gt_handled(self):
        from picarones.core.difficulty import compute_all_difficulties
        r = compute_all_difficulties(
            ["d1"],
            {},  # GT manquante
            {"d1": {"A": 0.2}},
        )
        assert "d1" in r


# ===========================================================================
# TestReportDataSprint7
# ===========================================================================

class TestReportDataSprint7:
    def test_has_statistics_key(self, report_data_s7):
        assert "statistics" in report_data_s7

    def test_has_reliability_curves(self, report_data_s7):
        assert "reliability_curves" in report_data_s7

    def test_has_venn_data(self, report_data_s7):
        assert "venn_data" in report_data_s7

    def test_has_error_clusters(self, report_data_s7):
        assert "error_clusters" in report_data_s7

    def test_has_correlation_per_engine(self, report_data_s7):
        assert "correlation_per_engine" in report_data_s7

    def test_pairwise_wilcoxon_non_empty(self, report_data_s7):
        pw = report_data_s7["statistics"]["pairwise_wilcoxon"]
        assert len(pw) > 0

    def test_bootstrap_cis_count(self, report_data_s7):
        cis = report_data_s7["statistics"]["bootstrap_cis"]
        n_engines = len(report_data_s7["engines"])
        assert len(cis) == n_engines

    def test_documents_have_difficulty_score(self, report_data_s7):
        for doc in report_data_s7["documents"]:
            assert "difficulty_score" in doc
            assert 0.0 <= doc["difficulty_score"] <= 1.0

    def test_documents_have_difficulty_label(self, report_data_s7):
        for doc in report_data_s7["documents"]:
            assert "difficulty_label" in doc
            assert doc["difficulty_label"] in ("Facile", "Modéré", "Difficile", "Très difficile")

    def test_reliability_curves_count(self, report_data_s7):
        rc = report_data_s7["reliability_curves"]
        assert len(rc) == len(report_data_s7["engines"])

    def test_reliability_curves_have_points(self, report_data_s7):
        for curve in report_data_s7["reliability_curves"]:
            assert "engine" in curve
            assert "points" in curve
            assert len(curve["points"]) > 0

    def test_correlation_matrix_symmetric(self, report_data_s7):
        for entry in report_data_s7["correlation_per_engine"]:
            m = entry["matrix"]
            n = len(m)
            for i in range(n):
                for j in range(n):
                    assert m[i][j] == pytest.approx(m[j][i], abs=1e-5)


# ===========================================================================
# TestHTMLSprint7Features
# ===========================================================================

class TestHTMLSprint7Features:
    def test_html_contains_export_csv_button(self, html_s7):
        assert "exportCSV" in html_s7 or "CSV" in html_s7

    def test_html_contains_presentation_mode_button(self, html_s7):
        assert "togglePresentMode" in html_s7 or "Présentation" in html_s7

    def test_html_contains_reliability_chart(self, html_s7):
        assert "chart-reliability" in html_s7

    def test_html_contains_bootstrap_ci_chart(self, html_s7):
        assert "chart-bootstrap-ci" in html_s7

    def test_html_contains_venn_container(self, html_s7):
        assert "venn-container" in html_s7

    def test_html_contains_wilcoxon_table(self, html_s7):
        assert "wilcoxon-table" in html_s7

    def test_html_contains_error_clusters(self, html_s7):
        assert "error-clusters" in html_s7

    def test_html_contains_correlation_matrix(self, html_s7):
        assert "corr-matrix" in html_s7 or "correlation" in html_s7.lower()

    def test_html_contains_difficulty_badge(self, html_s7):
        assert "difficulty" in html_s7.lower() or "diff-badge" in html_s7

    def test_html_contains_url_state(self, html_s7):
        assert "updateURL" in html_s7 or "history.replaceState" in html_s7
