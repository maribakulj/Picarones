"""Tests Sprint 79 — A.I.6 : projection de coût en volume cible.

Couvre :

1. ``project_cost_total`` :
   - Calcul linéaire en target_pages
   - target_pages = 0 → 0
   - target_pages négatif → None
   - cost_per_1k_pages_eur = None → None
2. ``project_co2_total`` : idem CO₂.
3. ``project_engine`` : retour structuré.
4. ``project_all_engines`` : N moteurs en une passe ; ValueError
   sur target négatif.
5. ``cost_gap_table`` :
   - Delta absolu/relatif vs baseline
   - Baseline = 0 → delta_rel None
   - Baseline inconnue → KeyError
6. Cas réaliste BnF : 80 000 pages BMS — 4 moteurs avec coûts
   différents → projection cohérente.
"""

from __future__ import annotations

import pytest

from picarones.measurements.cost_projection import (
    ProjectedCost,
    cost_gap_table,
    project_all_engines,
    project_co2_total,
    project_cost_total,
    project_engine,
)
from picarones.measurements.pricing import EngineCost


def _ec(name: str, cost_1k: float | None, co2_1k: float | None = None,
        type_: str = "cloud_api") -> EngineCost:
    return EngineCost(
        engine_key=name,
        type=type_,
        cost_per_1k_pages_eur=cost_1k,
        co2_per_1k_pages_g=co2_1k,
    )


# ──────────────────────────────────────────────────────────────────────────
# 1. project_cost_total
# ──────────────────────────────────────────────────────────────────────────


class TestProjectCostTotal:
    def test_linear_scaling(self) -> None:
        ec = _ec("x", 10.0)  # 10 €/1000 pages
        assert project_cost_total(ec, 1000) == 10.0
        assert project_cost_total(ec, 10000) == 100.0
        assert project_cost_total(ec, 80000) == 800.0

    def test_zero_pages(self) -> None:
        ec = _ec("x", 10.0)
        assert project_cost_total(ec, 0) == 0.0

    def test_negative_returns_none(self) -> None:
        ec = _ec("x", 10.0)
        assert project_cost_total(ec, -100) is None

    def test_no_cost_data(self) -> None:
        ec = _ec("x", None)
        assert project_cost_total(ec, 1000) is None

    def test_fractional_pages(self) -> None:
        ec = _ec("x", 10.0)
        # 500 pages → 5 €
        assert project_cost_total(ec, 500) == 5.0


# ──────────────────────────────────────────────────────────────────────────
# 2. project_co2_total
# ──────────────────────────────────────────────────────────────────────────


class TestProjectCo2:
    def test_linear_scaling(self) -> None:
        ec = _ec("x", None, co2_1k=50.0)  # 50 g/1000 pages
        assert project_co2_total(ec, 10000) == 500.0

    def test_no_co2_data(self) -> None:
        ec = _ec("x", 10.0)  # cost mais pas co2
        assert project_co2_total(ec, 1000) is None


# ──────────────────────────────────────────────────────────────────────────
# 3. project_engine
# ──────────────────────────────────────────────────────────────────────────


class TestProjectEngine:
    def test_full_struct(self) -> None:
        ec = _ec("tess", 5.0, co2_1k=20.0, type_="local")
        proj = project_engine(ec, 80000)
        assert isinstance(proj, ProjectedCost)
        assert proj.engine_key == "tess"
        assert proj.target_pages == 80000
        assert proj.cost_total_eur == 400.0
        assert proj.co2_total_g == 1600.0
        assert proj.cost_per_1k_pages_eur == 5.0
        assert proj.type == "local"

    def test_as_dict(self) -> None:
        ec = _ec("tess", 5.0)
        proj = project_engine(ec, 1000)
        d = proj.as_dict()
        assert d["engine_key"] == "tess"
        assert d["cost_total_eur"] == 5.0


# ──────────────────────────────────────────────────────────────────────────
# 4. project_all_engines
# ──────────────────────────────────────────────────────────────────────────


class TestProjectAll:
    def test_multi_engines(self) -> None:
        engines = {
            "tess": _ec("tess", 1.0, type_="local"),
            "mistral": _ec("mistral", 3.5, type_="cloud_api"),
            "gpt4o": _ec("gpt4o", 7.5, type_="cloud_api"),
        }
        result = project_all_engines(engines, 10000)
        assert result["tess"].cost_total_eur == 10.0
        assert result["mistral"].cost_total_eur == 35.0
        assert result["gpt4o"].cost_total_eur == 75.0

    def test_negative_target_raises(self) -> None:
        with pytest.raises(ValueError, match="target_pages"):
            project_all_engines({"x": _ec("x", 1.0)}, -1)

    def test_engine_without_data_kept(self) -> None:
        engines = {
            "known": _ec("known", 1.0),
            "unknown": _ec("unknown", None),
        }
        result = project_all_engines(engines, 1000)
        assert result["known"].cost_total_eur == 1.0
        assert "unknown" in result
        assert result["unknown"].cost_total_eur is None


# ──────────────────────────────────────────────────────────────────────────
# 5. cost_gap_table
# ──────────────────────────────────────────────────────────────────────────


class TestCostGapTable:
    def _projections(self) -> dict:
        return {
            "tess": project_engine(_ec("tess", 1.0), 10000),
            "mistral": project_engine(_ec("mistral", 5.0), 10000),
        }

    def test_gap_vs_baseline(self) -> None:
        projs = self._projections()
        gaps = cost_gap_table(projs, baseline_engine="tess")
        # tess : 10 €, mistral : 50 €
        assert gaps["tess"]["delta_abs"] == 0.0
        assert gaps["mistral"]["delta_abs"] == 40.0
        assert gaps["mistral"]["delta_rel"] == 4.0  # 40/10

    def test_unknown_baseline_raises(self) -> None:
        with pytest.raises(KeyError):
            cost_gap_table(self._projections(), "nonexistent")

    def test_baseline_zero_relative_none(self) -> None:
        projs = {
            "tess": project_engine(_ec("tess", 0.0, type_="local"), 10000),
            "mistral": project_engine(_ec("mistral", 5.0), 10000),
        }
        gaps = cost_gap_table(projs, "tess")
        # tess à 0 → relative non calculable
        assert gaps["mistral"]["delta_rel"] is None
        # absolute reste calculable
        assert gaps["mistral"]["delta_abs"] == 50.0

    def test_engine_without_data_skipped_in_calc(self) -> None:
        projs = {
            "known": project_engine(_ec("known", 1.0), 1000),
            "unknown": project_engine(_ec("unknown", None), 1000),
        }
        gaps = cost_gap_table(projs, "known")
        assert gaps["unknown"]["delta_abs"] is None
        assert gaps["unknown"]["delta_rel"] is None


# ──────────────────────────────────────────────────────────────────────────
# 6. Cas réaliste BnF
# ──────────────────────────────────────────────────────────────────────────


class TestRealisticBnF:
    def test_80k_pages_bms_scenario(self) -> None:
        # Scénario : registres paroissiaux, 80 000 pages
        engines = {
            "tesseract": _ec("tesseract", 0.04, type_="local"),
            "pero":      _ec("pero", 0.0, type_="local"),
            "mistral":   _ec("mistral", 3.5, type_="cloud_api"),
            "gpt4o":     _ec("gpt4o", 7.5, type_="cloud_api"),
        }
        target = 80000
        result = project_all_engines(engines, target)
        # Vérifications quantitatives
        assert result["tesseract"].cost_total_eur == pytest.approx(3.2)
        assert result["pero"].cost_total_eur == 0.0
        assert result["mistral"].cost_total_eur == 280.0
        assert result["gpt4o"].cost_total_eur == 600.0

        # Gap vs Tesseract : Mistral coûte 276,8 € de plus
        gaps = cost_gap_table(result, "tesseract")
        assert gaps["mistral"]["delta_abs"] == pytest.approx(276.8)
