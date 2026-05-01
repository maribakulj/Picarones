"""Tests Sprint 39 — métriques de calibration (ECE, MCE, reliability).

Le module ``picarones.measurements.calibration`` expose :

- ``CalibrationBin`` : un bin du reliability diagram
- ``reliability_diagram(confidences, is_correct, n_bins=10)``
- ``expected_calibration_error`` (ECE)
- ``maximum_calibration_error`` (MCE)
- ``compute_calibration_metrics`` : vue agrégée

Les tests vérifient :

1. **Calibration parfaite** : confidences uniformes égales à la précision
   du bin → ECE = MCE = 0.
2. **Sur-confiance extrême** : confidence = 1.0 mais 50 % correct →
   ECE = 0.5 et MCE = 0.5.
3. **Sous-confiance extrême** : confidence = 0.5 mais 100 % correct →
   ECE = 0.5.
4. **Calibration constante** : confidence = c, accuracy = a → ECE = |c-a|.
5. **Reliability diagram** : binning correct, bornes correctes,
   bin 1.0 inclus dans le dernier bin.
6. **Bins vides** correctement gérés (avg_confidence/accuracy = None,
   count = 0, gap = None).
7. **Listes vides** → ECE = 0, MCE = 0.
8. **Garde-fous** : longueurs incompatibles → ValueError ;
   confidence hors [0, 1] → ValueError ; n_bins < 1 → ValueError.
9. **n_bins paramétrable** : 5 bins vs 20 bins, bornes adaptées.
10. **compute_calibration_metrics** : structure de retour complète et
    cohérente avec les fonctions individuelles.
11. **CalibrationBin.gap** : comportement attendu (None pour bin vide).
"""

from __future__ import annotations

import pytest

from picarones.measurements.calibration import (
    CalibrationBin,
    compute_calibration_metrics,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_diagram,
)


# ──────────────────────────────────────────────────────────────────────────
# 1. Calibration parfaite
# ──────────────────────────────────────────────────────────────────────────


class TestPerfectCalibration:
    def test_uniform_confidence_matching_accuracy_per_bin(self) -> None:
        """Toutes les prédictions à confidence 0.75, 75 % correctes.
        Le seul bin non vide est [0.7, 0.8) avec gap = 0.
        """
        confs = [0.75] * 100
        correct = [1] * 75 + [0] * 25
        assert expected_calibration_error(confs, correct) == pytest.approx(0.0, abs=1e-9)
        assert maximum_calibration_error(confs, correct) == pytest.approx(0.0, abs=1e-9)

    def test_two_bins_each_perfectly_calibrated(self) -> None:
        # Bin [0.2, 0.3) : 25 % correct, 25 % conf
        # Bin [0.8, 0.9) : 85 % correct, 85 % conf
        confs = [0.25] * 100 + [0.85] * 100
        correct = [1] * 25 + [0] * 75 + [1] * 85 + [0] * 15
        assert expected_calibration_error(confs, correct) == pytest.approx(0.0, abs=1e-9)


# ──────────────────────────────────────────────────────────────────────────
# 2-3. Cas extrêmes
# ──────────────────────────────────────────────────────────────────────────


class TestExtremeCases:
    def test_extreme_overconfidence(self) -> None:
        # Le moteur dit "100 % sûr" mais a tort une fois sur deux
        confs = [1.0] * 10
        correct = [1] * 5 + [0] * 5
        assert expected_calibration_error(confs, correct) == pytest.approx(0.5)
        assert maximum_calibration_error(confs, correct) == pytest.approx(0.5)

    def test_extreme_underconfidence(self) -> None:
        # Le moteur dit "50 % sûr" mais a toujours raison
        confs = [0.5] * 10
        correct = [1] * 10
        assert expected_calibration_error(confs, correct) == pytest.approx(0.5)
        assert maximum_calibration_error(confs, correct) == pytest.approx(0.5)


# ──────────────────────────────────────────────────────────────────────────
# 4. Calibration constante (gap = |c - a|)
# ──────────────────────────────────────────────────────────────────────────


class TestConstantBias:
    @pytest.mark.parametrize("conf,acc", [(0.6, 0.4), (0.3, 0.7), (0.95, 0.85)])
    def test_constant_bias_is_absolute_gap(
        self, conf: float, acc: float
    ) -> None:
        """Avec un seul bin non vide, ECE = |conf - acc|."""
        n = 100
        confs = [conf] * n
        n_correct = int(round(acc * n))
        correct = [1] * n_correct + [0] * (n - n_correct)
        ece = expected_calibration_error(confs, correct)
        # acc effective = n_correct/n (peut différer légèrement de acc cible
        # par arrondi entier)
        actual_acc = n_correct / n
        assert ece == pytest.approx(abs(conf - actual_acc), abs=1e-9)


# ──────────────────────────────────────────────────────────────────────────
# 5. Reliability diagram — binning
# ──────────────────────────────────────────────────────────────────────────


class TestReliabilityDiagramBinning:
    def test_default_returns_10_bins(self) -> None:
        bins = reliability_diagram([0.5], [1])
        assert len(bins) == 10

    def test_bin_bounds_are_equidistant(self) -> None:
        bins = reliability_diagram([], [], n_bins=5)
        widths = [b.bin_high - b.bin_low for b in bins]
        for w in widths:
            assert w == pytest.approx(0.2, abs=1e-9)
        assert bins[0].bin_low == pytest.approx(0.0)
        assert bins[-1].bin_high == pytest.approx(1.0)

    def test_confidence_1_falls_in_last_bin(self) -> None:
        bins = reliability_diagram([1.0, 1.0, 1.0], [1, 0, 1], n_bins=10)
        # Toutes les prédictions doivent être dans le dernier bin
        assert bins[-1].count == 3
        assert sum(b.count for b in bins[:-1]) == 0

    def test_predictions_assigned_to_correct_bin(self) -> None:
        bins = reliability_diagram(
            [0.05, 0.15, 0.55, 0.95],
            [0, 1, 1, 0],
            n_bins=10,
        )
        # bin [0.0, 0.1) → 1 prédiction
        assert bins[0].count == 1
        # bin [0.1, 0.2) → 1
        assert bins[1].count == 1
        # bin [0.5, 0.6) → 1
        assert bins[5].count == 1
        # bin [0.9, 1.0] → 1
        assert bins[9].count == 1

    def test_avg_confidence_and_accuracy_per_bin(self) -> None:
        # Bin [0.6, 0.7) : confidences 0.6, 0.65 ; correct 1, 0
        bins = reliability_diagram([0.6, 0.65], [1, 0], n_bins=10)
        b6 = bins[6]
        assert b6.count == 2
        assert b6.avg_confidence == pytest.approx((0.6 + 0.65) / 2)
        assert b6.accuracy == pytest.approx(0.5)


# ──────────────────────────────────────────────────────────────────────────
# 6. Bins vides
# ──────────────────────────────────────────────────────────────────────────


class TestEmptyBins:
    def test_empty_bin_has_none_avg_and_accuracy(self) -> None:
        bins = reliability_diagram([0.95], [1], n_bins=10)
        # Tous les bins sauf le dernier sont vides
        for b in bins[:-1]:
            assert b.count == 0
            assert b.avg_confidence is None
            assert b.accuracy is None
            assert b.gap is None

    def test_ece_skips_empty_bins(self) -> None:
        # Avec un seul bin non vide à gap 0, ECE doit être 0
        bins = reliability_diagram([0.55] * 10, [1] * 6 + [0] * 4)
        assert expected_calibration_error([0.55] * 10, [1] * 6 + [0] * 4) == \
            pytest.approx(0.05)
        # Confirmer que beaucoup de bins sont vides
        empty = [b for b in bins if b.count == 0]
        assert len(empty) == 9


# ──────────────────────────────────────────────────────────────────────────
# 7. Listes vides
# ──────────────────────────────────────────────────────────────────────────


class TestEmptyInputs:
    def test_empty_lists_return_zero(self) -> None:
        assert expected_calibration_error([], []) == 0.0
        assert maximum_calibration_error([], []) == 0.0

    def test_empty_reliability_diagram(self) -> None:
        bins = reliability_diagram([], [], n_bins=10)
        assert len(bins) == 10
        assert all(b.count == 0 for b in bins)


# ──────────────────────────────────────────────────────────────────────────
# 8. Garde-fous
# ──────────────────────────────────────────────────────────────────────────


class TestGuards:
    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Longueurs"):
            expected_calibration_error([0.5, 0.5], [1])

    def test_confidence_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="hors"):
            expected_calibration_error([1.5], [1])

    def test_negative_confidence_raises(self) -> None:
        with pytest.raises(ValueError, match="hors"):
            expected_calibration_error([-0.1], [1])

    def test_invalid_n_bins_raises(self) -> None:
        with pytest.raises(ValueError, match="n_bins"):
            reliability_diagram([0.5], [1], n_bins=0)

    def test_n_bins_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="n_bins"):
            reliability_diagram([0.5], [1], n_bins=-3)


# ──────────────────────────────────────────────────────────────────────────
# 9. n_bins paramétrable
# ──────────────────────────────────────────────────────────────────────────


class TestVariableNBins:
    @pytest.mark.parametrize("n_bins,expected_width", [
        (5, 0.2), (10, 0.1), (20, 0.05), (1, 1.0),
    ])
    def test_bin_width_scales_with_n_bins(
        self, n_bins: int, expected_width: float
    ) -> None:
        bins = reliability_diagram([], [], n_bins=n_bins)
        assert len(bins) == n_bins
        for b in bins:
            assert (b.bin_high - b.bin_low) == pytest.approx(expected_width)

    def test_finer_bins_can_only_increase_or_keep_ece(self) -> None:
        """À distribution donnée, n_bins plus grand révèle des écarts
        masqués par un binning grossier — ECE ne décroît pas."""
        confs = [0.6, 0.65, 0.7, 0.95, 0.95]
        correct = [1, 0, 1, 1, 0]
        ece_5 = expected_calibration_error(confs, correct, n_bins=5)
        ece_20 = expected_calibration_error(confs, correct, n_bins=20)
        assert ece_20 >= ece_5 - 1e-9


# ──────────────────────────────────────────────────────────────────────────
# 10. compute_calibration_metrics
# ──────────────────────────────────────────────────────────────────────────


class TestComputeCalibrationMetrics:
    def test_returns_full_structure(self) -> None:
        confs = [0.6, 0.7, 0.95, 0.95]
        correct = [1, 0, 1, 1]
        out = compute_calibration_metrics(confs, correct, n_bins=10)
        assert set(out.keys()) >= {
            "ece", "mce", "n_bins", "n_predictions",
            "overall_accuracy", "overall_confidence", "bins",
        }
        assert out["n_predictions"] == 4
        assert out["overall_accuracy"] == pytest.approx(3 / 4)
        assert out["overall_confidence"] == pytest.approx((0.6 + 0.7 + 0.95 + 0.95) / 4)
        assert len(out["bins"]) == 10

    def test_ece_matches_function(self) -> None:
        confs = [0.55, 0.65, 0.75, 0.85, 0.95]
        correct = [1, 0, 1, 0, 1]
        out = compute_calibration_metrics(confs, correct)
        assert out["ece"] == pytest.approx(
            expected_calibration_error(confs, correct), abs=1e-9
        )
        assert out["mce"] == pytest.approx(
            maximum_calibration_error(confs, correct), abs=1e-9
        )

    def test_bin_dicts_contain_gap(self) -> None:
        out = compute_calibration_metrics([0.55] * 4, [1, 1, 0, 1])
        # Bin [0.5, 0.6) : avg_conf = 0.55, accuracy = 0.75, gap = 0.20
        b5 = out["bins"][5]
        assert b5["count"] == 4
        assert b5["gap"] == pytest.approx(0.20, abs=1e-9)


# ──────────────────────────────────────────────────────────────────────────
# 11. CalibrationBin.gap
# ──────────────────────────────────────────────────────────────────────────


class TestCalibrationBinGap:
    def test_gap_for_empty_bin_is_none(self) -> None:
        b = CalibrationBin(0.0, 0.1, None, None, 0)
        assert b.gap is None

    def test_gap_is_absolute_difference(self) -> None:
        b = CalibrationBin(0.5, 0.6, 0.55, 0.30, 10)
        assert b.gap == pytest.approx(0.25)

    def test_gap_symmetric(self) -> None:
        b1 = CalibrationBin(0.5, 0.6, 0.55, 0.30, 10)
        b2 = CalibrationBin(0.5, 0.6, 0.30, 0.55, 10)
        assert b1.gap == pytest.approx(b2.gap)
