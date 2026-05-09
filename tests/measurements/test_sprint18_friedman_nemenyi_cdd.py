"""Tests Sprint 18 — Friedman, Nemenyi post-hoc et Critical Difference Diagram.

Sprint 3 du plan rapport. Vérifie :
1. Le test de Friedman donne des résultats cohérents (cas canoniques + dégénérés).
2. Le post-hoc de Nemenyi calcule une critical distance correcte et identifie
   les groupes d'ex-aequo pratiques.
3. Le rendu SVG du CDD est valide et contient les éléments attendus.
4. Le rapport HTML inclut le CDD en tête.
"""

from __future__ import annotations

import re

import pytest

from picarones.evaluation.statistics import (
    build_critical_difference_svg,
    friedman_test,
    nemenyi_posthoc,
    _nemenyi_critical_value,
    _chi_square_sf,
    _rank_row,
)


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------

class TestRankRow:
    def test_ranks_ascending(self):
        assert _rank_row([0.1, 0.2, 0.3]) == [1.0, 2.0, 3.0]

    def test_ranks_with_ties_use_mean_rank(self):
        # Deux ex-aequo au milieu → rangs 2.5 et 2.5
        assert _rank_row([0.1, 0.2, 0.2, 0.3]) == [1.0, 2.5, 2.5, 4.0]

    def test_ranks_all_tied(self):
        # Toutes les valeurs égales → tous à rang (n+1)/2
        assert _rank_row([0.5, 0.5, 0.5]) == [2.0, 2.0, 2.0]


class TestChiSquareSf:
    def test_zero_returns_one(self):
        assert _chi_square_sf(0.0, 3) == 1.0

    def test_large_returns_near_zero(self):
        assert _chi_square_sf(100.0, 3) < 0.01

    def test_monotonic_decreasing(self):
        assert _chi_square_sf(1.0, 3) > _chi_square_sf(10.0, 3)


class TestNemenyiCriticalValue:
    def test_known_values_at_alpha_05(self):
        assert _nemenyi_critical_value(3, 0.05) == pytest.approx(2.343, abs=1e-3)
        assert _nemenyi_critical_value(5, 0.05) == pytest.approx(2.728, abs=1e-3)

    def test_k_out_of_range_uses_upper_bound(self):
        # k > 50 → borne max (conservateur)
        assert _nemenyi_critical_value(100, 0.05) == _nemenyi_critical_value(50, 0.05)

    def test_k_interpolation(self):
        # k=22 n'est pas dans la table, mais entre 20 et 25 → interpolation
        q22 = _nemenyi_critical_value(22, 0.05)
        q20 = _nemenyi_critical_value(20, 0.05)
        q25 = _nemenyi_critical_value(25, 0.05)
        assert q20 < q22 < q25

    def test_k_less_than_2_returns_none(self):
        assert _nemenyi_critical_value(1, 0.05) is None


# ---------------------------------------------------------------------------
# Friedman
# ---------------------------------------------------------------------------

class TestFriedmanTest:
    def test_three_engines_clearly_different(self):
        # Moteur A bat B bat C sur tous les documents : Friedman doit rejeter H0
        data = {
            "A": [0.01, 0.02, 0.03, 0.01, 0.02, 0.03, 0.02],
            "B": [0.10, 0.11, 0.12, 0.10, 0.11, 0.12, 0.11],
            "C": [0.30, 0.31, 0.32, 0.30, 0.31, 0.32, 0.31],
        }
        result = friedman_test(data)
        assert result["significant"] is True
        assert result["p_value"] < 0.05
        assert result["n_blocks"] == 7
        assert result["n_engines"] == 3
        # Rangs : A devrait être ~1, C devrait être ~3
        assert result["mean_ranks"]["A"] < result["mean_ranks"]["B"] < result["mean_ranks"]["C"]

    def test_three_engines_no_difference(self):
        # Trois moteurs identiques — Q proche de 0, p-value proche de 1
        data = {
            "A": [0.10, 0.15, 0.20, 0.12, 0.18, 0.14],
            "B": [0.10, 0.15, 0.20, 0.12, 0.18, 0.14],
            "C": [0.10, 0.15, 0.20, 0.12, 0.18, 0.14],
        }
        result = friedman_test(data)
        # Avec données parfaitement identiques, tous rangs = 2 (moyen)
        assert all(rank == pytest.approx(2.0) for rank in result["mean_ranks"].values())
        assert not result["significant"]

    def test_degenerate_single_engine(self):
        result = friedman_test({"A": [0.1, 0.2, 0.3]})
        assert result.get("error") == "not_enough_engines"
        assert not result["significant"]

    def test_degenerate_no_documents(self):
        result = friedman_test({"A": [], "B": []})
        assert result.get("error") == "not_enough_blocks"

    def test_degenerate_single_document(self):
        # Un seul document : on ne peut pas calculer un test sur 1 bloc
        result = friedman_test({"A": [0.1], "B": [0.2], "C": [0.3]})
        assert result.get("error") == "not_enough_blocks"

    def test_handles_uneven_lengths_by_truncating(self):
        # A a 5 valeurs, B en a 4 : on tronque au minimum
        data = {
            "A": [0.1, 0.2, 0.3, 0.4, 0.5],
            "B": [0.11, 0.21, 0.31, 0.41],
        }
        result = friedman_test(data)
        assert result["n_blocks"] == 4  # troncature

    def test_tie_correction_applied(self):
        # Tous les moteurs identiques sur plusieurs documents → tie correction
        # devrait empêcher une division par zéro ou une statistique NaN
        data = {
            "A": [0.1, 0.2, 0.1, 0.2],
            "B": [0.1, 0.2, 0.1, 0.2],
            "C": [0.2, 0.1, 0.2, 0.1],
        }
        result = friedman_test(data)
        # Doit retourner un résultat valide, pas une erreur
        assert "statistic" in result
        assert result["statistic"] >= 0.0

    def test_interpretation_is_informative(self):
        data = {"A": [0.01] * 8, "B": [0.50] * 8, "C": [0.99] * 8}
        result = friedman_test(data)
        assert "Friedman" in result["interpretation"]
        assert "Q" in result["interpretation"]
        assert "p" in result["interpretation"]


# ---------------------------------------------------------------------------
# Nemenyi post-hoc
# ---------------------------------------------------------------------------

class TestNemenyiPostHoc:
    def test_cd_greater_than_zero_on_typical_case(self):
        data = {
            "A": [0.01, 0.02, 0.03] * 5,
            "B": [0.10, 0.11, 0.12] * 5,
            "C": [0.30, 0.31, 0.32] * 5,
        }
        result = nemenyi_posthoc(data)
        assert result["critical_distance"] > 0
        assert result["n_blocks"] == 15
        assert result["n_engines"] == 3

    def test_very_different_engines_are_separated(self):
        # Les trois moteurs sont très distincts → Nemenyi doit les séparer
        data = {
            "A": [0.01, 0.02, 0.01, 0.02] * 5,
            "B": [0.30, 0.31, 0.30, 0.31] * 5,
            "C": [0.60, 0.61, 0.60, 0.61] * 5,
        }
        result = nemenyi_posthoc(data)
        # Chaque moteur devrait être dans son propre groupe
        assert len(result["tied_groups"]) == 3
        # Matrice : A vs B, A vs C, B vs C tous significatifs
        sm = result["significant_matrix"]
        assert sm[0][1] and sm[0][2] and sm[1][2]

    def test_similar_engines_are_grouped(self):
        # Trois moteurs quasi identiques
        data = {
            "A": [0.10 + 0.001 * (i % 3) for i in range(20)],
            "B": [0.10 + 0.001 * ((i + 1) % 3) for i in range(20)],
            "C": [0.10 + 0.001 * ((i + 2) % 3) for i in range(20)],
        }
        result = nemenyi_posthoc(data)
        # Avec des données si proches, tous devraient être dans UN groupe
        assert len(result["tied_groups"]) == 1
        assert set(result["tied_groups"][0]) == {"A", "B", "C"}

    def test_engines_sorted_by_mean_rank(self):
        data = {
            "winner":     [0.01, 0.01, 0.01, 0.01] * 3,
            "loser":      [0.99, 0.99, 0.99, 0.99] * 3,
            "middle":     [0.50, 0.50, 0.50, 0.50] * 3,
        }
        result = nemenyi_posthoc(data)
        assert result["engines_sorted"][0] == "winner"
        assert result["engines_sorted"][-1] == "loser"

    def test_degenerate_single_engine(self):
        result = nemenyi_posthoc({"A": [0.1, 0.2]})
        assert result.get("error") == "not_enough_data"

    def test_degenerate_no_data(self):
        result = nemenyi_posthoc({})
        assert result.get("error") == "not_enough_data"

    def test_matrix_is_symmetric(self):
        data = {
            "A": [0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3],
            "B": [0.3, 0.4, 0.5, 0.6, 0.7, 0.3, 0.4, 0.5],
            "C": [0.5, 0.6, 0.7, 0.8, 0.9, 0.5, 0.6, 0.7],
        }
        result = nemenyi_posthoc(data)
        sm = result["significant_matrix"]
        k = len(sm)
        for i in range(k):
            # Diagonale = False (un moteur n'est jamais différent de lui-même)
            assert not sm[i][i]
            for j in range(k):
                assert sm[i][j] == sm[j][i], "La matrice doit être symétrique"

    def test_alpha_parameter_affects_cd(self):
        data = {
            "A": [0.1] * 10,
            "B": [0.5] * 10,
            "C": [0.9] * 10,
        }
        r05 = nemenyi_posthoc(data, alpha=0.05)
        r01 = nemenyi_posthoc(data, alpha=0.01)
        # α=0.01 est plus strict → CD plus grand
        assert r01["critical_distance"] > r05["critical_distance"]


# ---------------------------------------------------------------------------
# Rendu SVG du CDD
# ---------------------------------------------------------------------------

class TestCriticalDifferenceSVG:
    def _sample_result(self, k: int = 4, n: int = 10) -> dict:
        data = {
            f"engine_{i}": [0.1 * i + 0.01 * j for j in range(n)]
            for i in range(k)
        }
        return nemenyi_posthoc(data)

    def test_svg_is_well_formed(self):
        svg = build_critical_difference_svg(self._sample_result())
        assert svg.startswith("<svg")
        assert svg.endswith("</svg>")
        assert 'xmlns="http://www.w3.org/2000/svg"' in svg

    def test_svg_contains_cd_marker(self):
        svg = build_critical_difference_svg(self._sample_result())
        assert re.search(r"CD = \d+\.\d+", svg)

    def test_svg_contains_axis_and_ticks(self):
        res = self._sample_result(k=5)
        svg = build_critical_difference_svg(res)
        # 5 moteurs → 5 ticks d'entiers sur l'axe
        assert svg.count('class="cd-tick"') >= 5

    def test_svg_contains_engine_names(self):
        res = self._sample_result(k=3)
        svg = build_critical_difference_svg(res)
        for name in res["engines_sorted"]:
            assert name in svg

    def test_svg_shows_tied_groups_as_bars(self):
        # Données à rangs alternés : chaque moteur gagne/perd de façon croisée
        # → rangs moyens très proches → au moins un groupe d'ex-aequo détecté
        data = {
            "A": [0.10, 0.20, 0.30, 0.10, 0.20, 0.30, 0.10, 0.20, 0.30, 0.10],
            "B": [0.20, 0.30, 0.10, 0.20, 0.30, 0.10, 0.20, 0.30, 0.10, 0.20],
            "C": [0.30, 0.10, 0.20, 0.30, 0.10, 0.20, 0.30, 0.10, 0.20, 0.30],
        }
        res = nemenyi_posthoc(data)
        # Avec rangs qui se compensent, tous les moteurs sont dans un même groupe
        assert len(res["tied_groups"]) == 1
        svg = build_critical_difference_svg(res)
        # La classe cd-tie apparaît dans le <style> et dans au moins une barre
        # tracée (donc >= 2 occurrences dont 1 dans un <line class="cd-tie">)
        assert 'class="cd-tie"' in svg

    def test_svg_degenerate_fallback(self):
        svg = build_critical_difference_svg({"error": "no_common_documents"})
        assert "<svg" in svg
        assert "non calculable" in svg.lower() or "indisponible" in svg.lower()

    def test_svg_escapes_special_characters_in_engine_names(self):
        malicious = {
            "A <script>": [0.1, 0.2, 0.3, 0.4],
            "B & C": [0.2, 0.3, 0.4, 0.5],
            'D "quoted"': [0.3, 0.4, 0.5, 0.6],
        }
        res = nemenyi_posthoc(malicious)
        svg = build_critical_difference_svg(res)
        # Les caractères dangereux doivent être échappés
        assert "<script>" not in svg.replace("A &lt;script&gt;", "")
        assert "&lt;script&gt;" in svg
        assert "&amp;" in svg


# ---------------------------------------------------------------------------
# Intégration dans le rapport HTML
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def benchmark_result():
    from picarones.evaluation import synthetic as fixtures
    return fixtures.generate_sample_benchmark(n_docs=8)


class TestReportIntegration:
    def test_report_contains_cdd_section(self, benchmark_result, tmp_path):
        from picarones.reports_v2.html.generator import ReportGenerator
        out = tmp_path / "report.html"
        ReportGenerator(benchmark_result).generate(out)
        html = out.read_text(encoding="utf-8")
        assert "cdd-card" in html
        assert "Friedman" in html
        assert "Nemenyi" in html or "nemenyi" in html
        # Le SVG doit être présent
        assert 'viewBox=' in html  # SVG du CDD
        assert "cd-tie" in html

    def test_report_json_contains_friedman_and_nemenyi(self, benchmark_result, tmp_path):
        from picarones.reports_v2.html.generator import _build_report_data
        data = _build_report_data(benchmark_result, images_b64={})
        stats = data.get("statistics", {})
        assert "friedman" in stats
        assert "nemenyi" in stats
        assert "mean_ranks" in stats["friedman"]
        assert "critical_distance" in stats["nemenyi"]
        assert "tied_groups" in stats["nemenyi"]

    def test_cdd_help_section_present(self, benchmark_result, tmp_path):
        from picarones.reports_v2.html.generator import ReportGenerator
        out = tmp_path / "report.html"
        ReportGenerator(benchmark_result).generate(out)
        html = out.read_text(encoding="utf-8")
        assert 'id="cdd-help"' in html
        assert "toggleCDDHelp" in html  # la fonction est bien liée au bouton

    def test_english_locale_uses_english_cdd_labels(self, benchmark_result, tmp_path):
        from picarones.reports_v2.html.generator import ReportGenerator
        out = tmp_path / "report_en.html"
        ReportGenerator(benchmark_result, lang="en").generate(out)
        html = out.read_text(encoding="utf-8")
        # La clé i18n doit être exposée ; le rendu JS remplacera data-i18n par
        # le texte anglais côté client. On vérifie juste la présence de la clé.
        assert 'data-i18n="cdd_title"' in html


# ---------------------------------------------------------------------------
# Détecteur narratif — detect_statistical_tie
# ---------------------------------------------------------------------------

class TestStatisticalTieDetector:
    def test_detector_emits_fact_when_engines_are_tied(self):
        from picarones.reports_v2.narrative.detectors import detect_statistical_tie
        from picarones.domain.facts import FactType

        benchmark_data = {
            "statistics": {
                "nemenyi": {
                    "tied_groups": [["A", "B"], ["C"]],
                    "mean_ranks": {"A": 1.2, "B": 1.4, "C": 3.0},
                    "critical_distance": 0.9,
                    "alpha": 0.05,
                    "n_blocks": 10,
                },
            },
        }
        facts = detect_statistical_tie(benchmark_data)
        assert len(facts) == 1
        f = facts[0]
        assert f.type == FactType.STATISTICAL_TIE
        assert set(f.engines_involved) == {"A", "B"}
        assert f.payload["includes_leader"] is True
        assert f.payload["critical_distance"] == 0.9

    def test_detector_ignores_singletons(self):
        from picarones.reports_v2.narrative.detectors import detect_statistical_tie

        benchmark_data = {
            "statistics": {
                "nemenyi": {
                    "tied_groups": [["A"], ["B"], ["C"]],
                    "mean_ranks": {"A": 1.0, "B": 2.0, "C": 3.0},
                    "critical_distance": 0.5,
                    "alpha": 0.05,
                    "n_blocks": 10,
                },
            },
        }
        facts = detect_statistical_tie(benchmark_data)
        assert facts == []

    def test_detector_returns_empty_on_missing_data(self):
        from picarones.reports_v2.narrative.detectors import detect_statistical_tie
        assert detect_statistical_tie({}) == []
        assert detect_statistical_tie({"statistics": {}}) == []
        assert detect_statistical_tie({"statistics": {"nemenyi": {"error": "no_data"}}}) == []

    def test_non_leader_tie_is_high_not_critical(self):
        from picarones.reports_v2.narrative.detectors import detect_statistical_tie
        from picarones.domain.facts import FactImportance

        benchmark_data = {
            "statistics": {
                "nemenyi": {
                    "tied_groups": [["A"], ["B", "C"]],
                    "mean_ranks": {"A": 1.0, "B": 2.5, "C": 2.7},
                    "critical_distance": 0.5,
                    "alpha": 0.05,
                    "n_blocks": 10,
                },
            },
        }
        facts = detect_statistical_tie(benchmark_data)
        assert len(facts) == 1
        assert facts[0].importance == FactImportance.HIGH
        assert facts[0].payload["includes_leader"] is False
