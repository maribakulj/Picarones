"""Tests Sprint 20 — modélisation coût + vue Pareto.

Sprint 5 du plan rapport. Couvre :
  1. `pricing.py` : chargement de la table, estimation locale vs cloud.
  2. `compute_pareto_front` : cas canoniques + dégénérés.
  3. Intégration `_build_report_data` : coût annoté, front calculé, JSON ok.
  4. Détecteurs narratifs `pareto_alternative` et `cost_outlier`.
  5. Rendu HTML : section Pareto, toggles axes, notes méthodologiques.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from picarones.core.narrative import build_synthesis
from picarones.core.narrative.detectors import (
    detect_cost_outlier,
    detect_pareto_alternative,
)
from picarones.core.narrative.facts import FactType
from picarones.core.pricing import (
    build_costs_for_benchmark,
    estimate_cost,
    load_pricing_database,
)
from picarones.core.statistics import compute_pareto_front


# ---------------------------------------------------------------------------
# 1. Pricing
# ---------------------------------------------------------------------------

class TestLoadPricingDatabase:
    def test_default_file_loads(self):
        defaults, table = load_pricing_database()
        assert defaults.currency == "EUR"
        assert defaults.last_updated  # doit être rempli
        assert "tesseract" in table
        assert "gpt-4o" in table
        assert "google_vision" in table

    def test_missing_file_returns_empty(self, tmp_path):
        missing = tmp_path / "nope.yaml"
        defaults, table = load_pricing_database(missing)
        assert table == {}
        assert defaults.currency == "EUR"  # fallback


class TestEstimateCost:
    def test_cloud_api_uses_listed_price(self):
        cost = estimate_cost("google_vision")
        assert cost.type == "cloud_api"
        assert cost.cost_per_1k_pages_eur > 0
        assert cost.pricing_source_url is not None
        assert cost.api_price_per_1k_pages == cost.cost_per_1k_pages_eur

    def test_local_engine_uses_seconds_times_rate(self):
        cost = estimate_cost("tesseract")
        assert cost.type == "local"
        # 2s/page × 1000 pages / 3600 × 0.08 €/h ≈ 0.044 €
        assert cost.cost_per_1k_pages_eur == pytest.approx(0.044, abs=0.01)
        assert "Temps d'inférence" in " ".join(cost.assumptions)

    def test_measured_seconds_override_indicative(self):
        cost = estimate_cost("tesseract", measured_seconds_per_page=10.0)
        # Rate = 0.08 €/h → 10 × 1000 / 3600 × 0.08 ≈ 0.22 €
        assert cost.cost_per_1k_pages_eur == pytest.approx(0.222, abs=0.01)
        assert "mesuré" in " ".join(cost.assumptions)

    def test_pipeline_prefers_llm_model(self):
        cost = estimate_cost(
            engine_name="tesseract → gpt-4o",
            llm_model="gpt-4o",
            is_pipeline=True,
        )
        assert cost.engine_key == "gpt-4o"
        assert cost.type == "cloud_api"

    def test_unknown_engine_returns_unknown_type(self):
        cost = estimate_cost("totally-not-a-real-engine")
        assert cost.type == "unknown"
        assert cost.cost_per_1k_pages_eur is None
        assert "Aucune entrée" in " ".join(cost.assumptions)

    def test_hourly_rate_override(self):
        cheap = estimate_cost("tesseract", hourly_rate_override_eur=0.01)
        expensive = estimate_cost("tesseract", hourly_rate_override_eur=10.0)
        assert expensive.cost_per_1k_pages_eur > cheap.cost_per_1k_pages_eur

    def test_carbon_estimate_computed(self):
        cost = estimate_cost("gpt-4o")
        assert cost.co2_per_1k_pages_g is not None
        assert cost.co2_per_1k_pages_g > 0
        # kWh × grid intensity → positive et cohérent
        expected = cost.kwh_per_1k_pages * cost.grid_intensity_g_co2_per_kwh
        assert cost.co2_per_1k_pages_g == pytest.approx(expected)


class TestBuildCostsForBenchmark:
    def test_annotates_all_engines(self):
        engines = [
            {"name": "tesseract", "is_pipeline": False, "pipeline_info": {}},
            {"name": "pipeline", "is_pipeline": True,
             "pipeline_info": {"llm_model": "gpt-4o"}},
        ]
        durations = {"tesseract": 1.5, "pipeline": 12.0}
        costs = build_costs_for_benchmark(engines, durations)
        assert "tesseract" in costs
        assert "pipeline" in costs
        assert costs["tesseract"]["type"] == "local"
        assert costs["pipeline"]["type"] == "cloud_api"


# ---------------------------------------------------------------------------
# 2. Pareto
# ---------------------------------------------------------------------------

class TestComputeParetoFront:
    def test_trivial_front(self):
        points = [
            {"engine": "A", "cer": 0.05, "cost": 1.0},  # meilleur CER
            {"engine": "B", "cer": 0.10, "cost": 0.1},  # meilleur coût
            {"engine": "C", "cer": 0.08, "cost": 2.0},  # dominé par A
        ]
        front = compute_pareto_front(points)
        assert set(front) == {"A", "B"}

    def test_empty_input(self):
        assert compute_pareto_front([]) == []

    def test_single_point_is_its_own_front(self):
        assert compute_pareto_front([{"engine": "X", "cer": 0.1, "cost": 1.0}]) == ["X"]

    def test_skips_points_with_missing_values(self):
        points = [
            {"engine": "A", "cer": 0.05, "cost": 1.0},
            {"engine": "B", "cost": 0.5},  # pas de cer
            {"engine": "C", "cer": 0.10},  # pas de cost
        ]
        front = compute_pareto_front(points)
        assert front == ["A"]

    def test_three_dimensional_front(self):
        # 3 objectifs à minimiser — vérifie que le détecteur marche à k>2
        points = [
            {"name": "A", "a": 1, "b": 10, "c": 100},  # meilleur en a
            {"name": "B", "a": 10, "b": 1, "c": 100},  # meilleur en b
            {"name": "C", "a": 10, "b": 10, "c": 1},   # meilleur en c
            {"name": "D", "a": 20, "b": 20, "c": 200}, # dominé partout
        ]
        front = compute_pareto_front(
            points, objectives=("a", "b", "c"), name_key="name",
        )
        assert set(front) == {"A", "B", "C"}
        assert "D" not in front

    def test_mixed_min_max(self):
        # Minimiser CER, maximiser ancrage
        points = [
            {"engine": "A", "cer": 0.05, "anchor": 0.95},  # meilleur partout
            {"engine": "B", "cer": 0.10, "anchor": 0.85},  # dominé
            {"engine": "C", "cer": 0.08, "anchor": 0.99},  # meilleur anchor
        ]
        front = compute_pareto_front(
            points,
            objectives=("cer", "anchor"),
            minimize=(True, False),
        )
        assert set(front) == {"A", "C"}

    def test_minimize_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_pareto_front([{"engine": "A", "cer": 0.1, "cost": 1.0}],
                                 objectives=("cer", "cost"),
                                 minimize=(True,))


# ---------------------------------------------------------------------------
# 3. Détecteurs narratifs Pareto / cost
# ---------------------------------------------------------------------------

def _pareto_data(cost_points, front=None, speed_points=None, co2_points=None):
    return {
        "ranking": [{"engine": p["engine"], "mean_cer": p["cer"],
                     "documents": 10, "failed": 0} for p in cost_points],
        "pareto": {
            "cost": {"points": cost_points, "front": front or [p["engine"] for p in cost_points]},
            "speed": {"points": speed_points or [], "front": []},
            "co2": {"points": co2_points or [], "front": []},
        },
    }


class TestDetectParetoAlternative:
    def test_emits_when_alt_is_cheaper(self):
        data = _pareto_data(
            [
                {"engine": "best", "cer": 0.02, "cost": 5.0},
                {"engine": "cheap", "cer": 0.04, "cost": 0.1},
                {"engine": "dominated", "cer": 0.05, "cost": 3.0},
            ],
            front=["best", "cheap"],
        )
        # Forcer "best" comme leader
        data["ranking"] = [
            {"engine": "best", "mean_cer": 0.02, "documents": 10, "failed": 0},
            {"engine": "cheap", "mean_cer": 0.04, "documents": 10, "failed": 0},
            {"engine": "dominated", "mean_cer": 0.05, "documents": 10, "failed": 0},
        ]
        facts = detect_pareto_alternative(data)
        assert len(facts) == 1
        assert facts[0].payload["engine"] == "cheap"
        assert facts[0].payload["leader"] == "best"
        assert facts[0].payload["cost_saving_ratio"] >= 10

    def test_empty_when_front_has_only_leader(self):
        data = _pareto_data(
            [{"engine": "best", "cer": 0.02, "cost": 5.0}],
            front=["best"],
        )
        assert detect_pareto_alternative(data) == []

    def test_empty_when_no_pareto_section(self):
        assert detect_pareto_alternative({}) == []


class TestDetectCostOutlier:
    def test_flags_expensive_dominated_engine(self):
        data = _pareto_data(
            [
                {"engine": "cheap", "cer": 0.05, "cost": 0.1},
                {"engine": "normal", "cer": 0.08, "cost": 1.0},
                {"engine": "expensive_bad", "cer": 0.15, "cost": 20.0},
            ],
            front=["cheap"],
        )
        facts = detect_cost_outlier(data)
        assert any(f.payload["engine"] == "expensive_bad" for f in facts)

    def test_does_not_flag_expensive_on_front(self):
        # Un moteur cher mais sur le front = coût justifié par qualité unique
        data = _pareto_data(
            [
                {"engine": "cheap", "cer": 0.30, "cost": 0.1},
                {"engine": "normal", "cer": 0.15, "cost": 1.0},
                {"engine": "expensive_best", "cer": 0.02, "cost": 20.0},
            ],
            front=["cheap", "expensive_best"],
        )
        facts = detect_cost_outlier(data)
        names = {f.payload["engine"] for f in facts}
        assert "expensive_best" not in names


# ---------------------------------------------------------------------------
# 4. Intégration rapport HTML
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def benchmark_result():
    from picarones import fixtures
    return fixtures.generate_sample_benchmark(n_docs=8)


class TestReportIntegration:
    def test_report_contains_pareto_card(self, benchmark_result, tmp_path):
        from picarones.report.generator import ReportGenerator
        out = tmp_path / "report.html"
        ReportGenerator(benchmark_result).generate(out)
        html = out.read_text(encoding="utf-8")
        assert 'class="chart-card pareto-card"' in html
        assert 'id="pareto-chart"' in html
        assert 'setParetoAxis(\'cost\')' in html
        assert 'setParetoAxis(\'speed\')' in html
        assert 'setParetoAxis(\'co2\')' in html
        assert "pareto-experimental" in html  # étiquette expérimental

    def test_report_json_contains_pareto_data(self, benchmark_result):
        from picarones.report.generator import _build_report_data
        data = _build_report_data(benchmark_result, images_b64={})
        pareto = data.get("pareto", {})
        assert "cost" in pareto
        assert "speed" in pareto
        assert "co2" in pareto
        assert "pricing_meta" in pareto
        # Les moteurs doivent porter leur champ cost
        for e in data["engines"]:
            assert "cost" in e, f"Moteur {e.get('name')} sans champ cost"

    def test_synthesis_may_include_pareto_sentence(self, benchmark_result, tmp_path):
        # Sur la fixture de démo, pero_ocr + tesseract sont sur le front → la
        # synthèse doit remonter une alternative moins chère
        from picarones.report.generator import ReportGenerator
        out = tmp_path / "report.html"
        ReportGenerator(benchmark_result).generate(out)
        html = out.read_text(encoding="utf-8")
        m = re.search(r'<ul class="synth-list">(.*?)</ul>', html, re.DOTALL)
        assert m
        ul_content = m.group(1)
        # On n'exige pas "compromis" en dur (dépend de l'i18n) — simplement
        # qu'un moteur et "€" apparaissent (signe que pareto_alternative a tiré)
        assert "€" in ul_content or "pero_ocr" in ul_content

    def test_pricing_yaml_is_packaged(self):
        """Garde-fou : le YAML doit être accessible depuis le package."""
        from picarones.core.pricing import _DEFAULT_PRICING_PATH
        assert Path(_DEFAULT_PRICING_PATH).exists()

    def test_english_locale_renders_pareto_labels(self, benchmark_result, tmp_path):
        from picarones.report.generator import ReportGenerator
        out = tmp_path / "report_en.html"
        ReportGenerator(benchmark_result, lang="en").generate(out)
        html = out.read_text(encoding="utf-8")
        assert 'data-i18n="h_pareto"' in html
        assert 'data-i18n="pareto_axis_cost"' in html


# ---------------------------------------------------------------------------
# 5. Traçabilité des nombres (anti-hallucination pour les 2 nouveaux templates)
# ---------------------------------------------------------------------------

class TestAntiHallucinationOnPareto:
    def test_pareto_alternative_numbers_traceable(self):
        data = _pareto_data(
            [
                {"engine": "A", "cer": 0.02, "cost": 5.0},
                {"engine": "B", "cer": 0.04, "cost": 0.25},
            ],
            front=["A", "B"],
        )
        data["ranking"] = [
            {"engine": "A", "mean_cer": 0.02, "documents": 10, "failed": 0},
            {"engine": "B", "mean_cer": 0.04, "documents": 10, "failed": 0},
        ]
        # Autres infos requises par build_synthesis
        data.setdefault("meta", {"document_count": 10})
        data.setdefault("engines", [
            {"name": "A", "cer": 0.02},
            {"name": "B", "cer": 0.04},
        ])
        data.setdefault("statistics", {
            "pairwise_wilcoxon": [], "bootstrap_cis": [],
            "friedman": {}, "nemenyi": {"tied_groups": [], "mean_ranks": {}},
        })
        data.setdefault("documents", [])

        result = build_synthesis(data, "fr")
        # Chercher la phrase pareto
        pareto_sentences = [s for s in result["sentences"] if "compromis" in s or "€" in s]
        assert pareto_sentences
        # Les nombres principaux doivent venir du payload : 4 (cer_pct=4), 0.25 (cost),
        # 2 (leader_cer_pct=2), 5 (leader_cost), 20 (ratio=5/0.25)
        facts_by_type = {f["type"]: f for f in result["facts"]}
        assert FactType.PARETO_ALTERNATIVE.value in facts_by_type
        payload = facts_by_type[FactType.PARETO_ALTERNATIVE.value]["payload"]
        sentence = pareto_sentences[0]
        for k in ("cost", "leader_cost", "cer_pct", "leader_cer_pct", "cost_saving_ratio"):
            val = payload.get(k)
            if val is None:
                continue
            # Au moins une représentation du nombre doit apparaître
            variants = {str(val), str(float(val)), f"{float(val):.1f}", f"{float(val):.2f}"}
            if val == int(val):
                variants.add(str(int(val)))
            assert any(v in sentence for v in variants), (
                f"Valeur {k}={val} absente de la phrase : {sentence!r}"
            )
