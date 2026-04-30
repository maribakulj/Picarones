"""Tests Sprint 81 — A.I.8 : robustesse projetée sur corpus réel.

Couvre :

1. ``_interpolate_cer`` :
   - Niveau exact sur la courbe → CER exact
   - Interpolation entre 2 points
   - Clip lower/upper
   - Pas de cer valide → None
2. ``_extract_quality_value`` : mapping default + custom.
3. ``project_robustness_on_corpus`` :
   - 1 moteur × 1 dégradation × N docs → projection cohérente
   - Multi-moteurs / multi-dégradations
   - Document sans qualité → ignoré
   - Aucune courbe → projection vide
   - Aucun doc → entry omis
   - n_docs_above_critical correct
4. ``aggregate_projection_per_engine`` :
   - Total deficit sur N types
   - Worst degradation type identifié
"""

from __future__ import annotations

import pytest

from picarones.core.robustness_projection import (
    _extract_quality_value,
    _interpolate_cer,
    aggregate_projection_per_engine,
    project_robustness_on_corpus,
)


# ──────────────────────────────────────────────────────────────────────────
# 1. _interpolate_cer
# ──────────────────────────────────────────────────────────────────────────


class TestInterpolate:
    def test_exact_match(self) -> None:
        assert _interpolate_cer(
            [0, 5, 10, 20], [0.05, 0.10, 0.20, 0.50], 10,
        ) == 0.20

    def test_linear_interpolation(self) -> None:
        # Entre 5 (CER 0.10) et 10 (CER 0.20), niveau 7.5 → CER 0.15
        assert _interpolate_cer(
            [5, 10], [0.10, 0.20], 7.5,
        ) == pytest.approx(0.15)

    def test_clip_lower(self) -> None:
        # Niveau en-dessous du min → CER au min
        assert _interpolate_cer([5, 10], [0.10, 0.20], -1) == 0.10

    def test_clip_upper(self) -> None:
        assert _interpolate_cer([5, 10], [0.10, 0.20], 100) == 0.20

    def test_empty_levels(self) -> None:
        assert _interpolate_cer([], [], 5) is None

    def test_all_cer_none(self) -> None:
        assert _interpolate_cer([0, 5], [None, None], 3) is None

    def test_some_cer_none_skipped(self) -> None:
        # Le None est ignoré, on interpole entre les valides
        result = _interpolate_cer(
            [0, 5, 10], [0.10, None, 0.30], 5,
        )
        # Interpolé entre (0, 0.10) et (10, 0.30) à level 5 → 0.20
        assert result == pytest.approx(0.20)


# ──────────────────────────────────────────────────────────────────────────
# 2. _extract_quality_value
# ──────────────────────────────────────────────────────────────────────────


class TestExtractQuality:
    def test_default_mapping(self) -> None:
        q = {"noise_level": 15.0, "blur_score": 200.0}
        assert _extract_quality_value(q, "noise") == 15.0
        assert _extract_quality_value(q, "blur") == 200.0

    def test_unknown_degradation(self) -> None:
        assert _extract_quality_value({}, "unknown") is None

    def test_missing_field(self) -> None:
        assert _extract_quality_value({}, "noise") is None

    def test_custom_mapping(self) -> None:
        q = {"my_noise_metric": 22.0}
        result = _extract_quality_value(
            q, "noise", custom_mapping={"noise": "my_noise_metric"},
        )
        assert result == 22.0


# ──────────────────────────────────────────────────────────────────────────
# 3. project_robustness_on_corpus
# ──────────────────────────────────────────────────────────────────────────


class TestProjection:
    def _curve(self, engine="t", deg="noise") -> dict:
        return {
            "engine_name": engine,
            "degradation_type": deg,
            "levels": [0, 5, 10, 20],
            "cer_values": [0.05, 0.10, 0.20, 0.50],
            "critical_threshold_level": 10,
            "cer_threshold": 0.20,
        }

    def test_single_curve_single_doc(self) -> None:
        curve = self._curve()
        # Un doc avec niveau de bruit 7.5 → CER 0.15
        qualities = [{"noise_level": 7.5}]
        result = project_robustness_on_corpus([curve], qualities)
        assert "t" in result
        deg_data = result["t"]["noise"]
        assert deg_data["n_docs"] == 1
        assert deg_data["n_docs_with_data"] == 1
        assert deg_data["expected_cer_mean"] == pytest.approx(0.15)
        assert deg_data["baseline_cer"] == pytest.approx(0.05)
        assert deg_data["deficit_vs_baseline"] == pytest.approx(0.10)

    def test_doc_above_critical(self) -> None:
        curve = self._curve()
        # 3 docs : 2 sous le seuil critique (niveau 5 → CER 0.10),
        # 1 au-dessus (niveau 15 → CER 0.35)
        qualities = [
            {"noise_level": 5}, {"noise_level": 5}, {"noise_level": 15},
        ]
        result = project_robustness_on_corpus([curve], qualities)
        deg = result["t"]["noise"]
        # critical_threshold_cer = 0.20 → 1 doc au-dessus
        assert deg["n_docs_above_critical"] == 1

    def test_doc_without_data_ignored(self) -> None:
        curve = self._curve()
        qualities = [
            {"noise_level": 5},
            {},  # pas de noise_level
        ]
        result = project_robustness_on_corpus([curve], qualities)
        deg = result["t"]["noise"]
        assert deg["n_docs"] == 2
        assert deg["n_docs_with_data"] == 1

    def test_multiple_engines_and_types(self) -> None:
        curves = [
            self._curve("alpha", "noise"),
            self._curve("alpha", "blur"),
            self._curve("beta", "noise"),
        ]
        qualities = [{"noise_level": 5, "blur_score": 5}]
        result = project_robustness_on_corpus(curves, qualities)
        assert "alpha" in result
        assert "beta" in result
        assert "noise" in result["alpha"]
        assert "blur" in result["alpha"]

    def test_no_curves_returns_empty(self) -> None:
        assert project_robustness_on_corpus([], [{"noise_level": 5}]) == {}

    def test_no_docs_omits_entry(self) -> None:
        curve = self._curve()
        result = project_robustness_on_corpus([curve], [])
        # Pas d'entry pour t/noise puisque per_doc_cer est vide
        assert result == {}

    def test_critical_threshold_override(self) -> None:
        curve = self._curve()
        # Niveau 5 → CER 0.10, niveau 10 → CER 0.20
        qualities = [{"noise_level": 7}, {"noise_level": 10}]
        # Avec critical=0.15, le doc à niveau 7 (CER ≈ 0.14) est sous, niveau 10 (CER 0.20) est au-dessus
        result = project_robustness_on_corpus(
            [curve], qualities, critical_threshold=0.15,
        )
        assert result["t"]["noise"]["n_docs_above_critical"] >= 1


# ──────────────────────────────────────────────────────────────────────────
# 4. aggregate_projection_per_engine
# ──────────────────────────────────────────────────────────────────────────


class TestAggregate:
    def test_total_deficit_summed(self) -> None:
        projection = {
            "t": {
                "noise": {"deficit_vs_baseline": 0.10},
                "blur": {"deficit_vs_baseline": 0.05},
            },
        }
        agg = aggregate_projection_per_engine(projection)
        assert agg["t"]["total_expected_deficit"] == pytest.approx(0.15)
        assert agg["t"]["n_degradation_types"] == 2

    def test_worst_degradation_identified(self) -> None:
        projection = {
            "t": {
                "noise": {"deficit_vs_baseline": 0.05},
                "blur": {"deficit_vs_baseline": 0.20},
                "rotation": {"deficit_vs_baseline": 0.02},
            },
        }
        agg = aggregate_projection_per_engine(projection)
        assert agg["t"]["worst_degradation_type"] == "blur"
        assert agg["t"]["worst_degradation_deficit"] == 0.20

    def test_none_deficit_skipped(self) -> None:
        projection = {
            "t": {
                "noise": {"deficit_vs_baseline": 0.05},
                "blur": {"deficit_vs_baseline": None},
            },
        }
        agg = aggregate_projection_per_engine(projection)
        assert agg["t"]["total_expected_deficit"] == pytest.approx(0.05)
        assert agg["t"]["n_degradation_types"] == 1

    def test_empty_projection(self) -> None:
        assert aggregate_projection_per_engine({}) == {}
