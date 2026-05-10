"""Tests Sprint 83 — A.II.4 : métriques de fiabilité.

Couvre :

1. ``cohen_kappa`` :
   - accord parfait → κ = 1
   - hasard → κ = 0
   - désaccord pire que hasard → κ < 0
   - séquences de tailles incompatibles → None
   - séquence vide → None
   - un seul label (pe = 1) → convention 1.0 ou 0.0
2. ``krippendorff_alpha`` :
   - accord total
   - missing values gérées
   - corpus avec un seul label → None
3. ``compute_iaa`` :
   - GT identiques → κ = 1, α = 1
   - différence ponctuelle → κ ∈ ]0, 1[
   - inputs vides → None
4. ``compute_multirun_stability`` :
   - 1 run → None
   - 3 runs identiques → divergence = 0, n_distinct = 1
   - 3 runs différents → divergence > 0
   - reference fournie → cer_per_run + variance + cv
5. Helper ``_aligned_char_pairs`` (privé mais central).
"""

from __future__ import annotations

import pytest

from picarones.evaluation.metrics.reliability import (
    _aligned_char_pairs,
    cohen_kappa,
    compute_iaa,
    compute_multirun_stability,
    krippendorff_alpha,
)


# ──────────────────────────────────────────────────────────────────────────
# 1. cohen_kappa
# ──────────────────────────────────────────────────────────────────────────


class TestCohenKappa:
    def test_perfect_agreement(self) -> None:
        assert cohen_kappa(["a", "b", "c"], ["a", "b", "c"]) == 1.0

    def test_total_disagreement_with_balanced_classes(self) -> None:
        # 4 obs, 2 classes équiprobables, désaccord total possible
        # quand A = [1,1,2,2] et B = [2,2,1,1]
        kappa = cohen_kappa([1, 1, 2, 2], [2, 2, 1, 1])
        assert kappa == pytest.approx(-1.0)

    def test_chance_level(self) -> None:
        # A = [1,2,1,2], B = [1,1,2,2] → po=0.5, pe=0.5 → κ=0
        kappa = cohen_kappa([1, 2, 1, 2], [1, 1, 2, 2])
        assert kappa == pytest.approx(0.0)

    def test_size_mismatch(self) -> None:
        assert cohen_kappa([1, 2], [1]) is None

    def test_empty(self) -> None:
        assert cohen_kappa([], []) is None

    def test_single_label_identical(self) -> None:
        # pe = 1 indéfini ; convention identité → 1.0
        assert cohen_kappa(["a", "a"], ["a", "a"]) == 1.0


# ──────────────────────────────────────────────────────────────────────────
# 2. krippendorff_alpha
# ──────────────────────────────────────────────────────────────────────────


class TestKrippendorffAlpha:
    def test_perfect_agreement(self) -> None:
        units = [["a", "a", "a"], ["b", "b", "b"], ["c", "c", "c"]]
        assert krippendorff_alpha(units) == 1.0

    def test_handles_missing_values(self) -> None:
        # Première unité 2 valides, seconde 3 valides
        units = [["a", "a", None], ["b", "b", "b"]]
        assert krippendorff_alpha(units) == 1.0

    def test_single_label_returns_none(self) -> None:
        # Un seul label dans tout le corpus → De = 0 → indéfini
        units = [["a", "a"], ["a", "a"]]
        assert krippendorff_alpha(units) is None

    def test_empty_returns_none(self) -> None:
        assert krippendorff_alpha([]) is None

    def test_units_with_less_than_two_skipped(self) -> None:
        # Toutes les unités ont moins de 2 valides → None
        units = [["a"], ["b"], [None]]
        assert krippendorff_alpha(units) is None


# ──────────────────────────────────────────────────────────────────────────
# 3. compute_iaa
# ──────────────────────────────────────────────────────────────────────────


class TestComputeIAA:
    def test_identical_transcriptions(self) -> None:
        result = compute_iaa("hello world", "hello world")
        assert result is not None
        assert result["cohen_kappa"] == 1.0
        assert result["agreement_rate"] == 1.0

    def test_partial_disagreement(self) -> None:
        result = compute_iaa("hello", "hallo")
        assert result is not None
        assert 0 < result["agreement_rate"] < 1
        assert 0 < (result["cohen_kappa"] or 0) < 1

    def test_empty_returns_none(self) -> None:
        assert compute_iaa("", "") is None

    def test_one_empty_returns_none(self) -> None:
        # `_aligned_char_pairs` ne peut produire que des opcodes
        # `insert` ou `delete` ici → pas d'alignement bilatéral
        assert compute_iaa("abc", "") is None

    def test_returns_n_aligned_chars(self) -> None:
        result = compute_iaa("hello", "hello")
        assert result["n_aligned_chars"] == 5


# ──────────────────────────────────────────────────────────────────────────
# 4. compute_multirun_stability
# ──────────────────────────────────────────────────────────────────────────


class TestMultirunStability:
    def test_single_run_returns_none(self) -> None:
        assert compute_multirun_stability(["hello"]) is None

    def test_three_identical_runs(self) -> None:
        result = compute_multirun_stability(
            ["hello world", "hello world", "hello world"],
        )
        assert result["n_runs"] == 3
        assert result["pairwise_disagreement_mean"] == 0.0
        assert result["pairwise_disagreement_max"] == 0.0
        assert result["identical_run_rate"] == 1.0
        assert result["n_distinct_outputs"] == 1
        # Pas de référence → cer_per_run None
        assert result["cer_per_run"] is None

    def test_three_distinct_runs(self) -> None:
        result = compute_multirun_stability(
            ["a b c", "a b d", "a c d"],
        )
        assert result["pairwise_disagreement_mean"] > 0
        assert result["identical_run_rate"] == 0.0
        assert result["n_distinct_outputs"] == 3

    def test_with_reference_computes_cer_metrics(self) -> None:
        result = compute_multirun_stability(
            ["hello world", "helo world", "hello word"],
            reference="hello world",
        )
        assert result["cer_per_run"] is not None
        assert len(result["cer_per_run"]) == 3
        assert result["cer_mean"] is not None
        assert result["cer_stdev"] is not None
        assert result["cer_cv"] is not None
        assert result["cer_cv"] > 0

    def test_with_reference_perfect_runs(self) -> None:
        # 3 runs identiques égaux à la référence
        result = compute_multirun_stability(
            ["abc"] * 3,
            reference="abc",
        )
        assert result["cer_mean"] == 0.0
        assert result["cer_stdev"] == 0.0
        # CV indéfini (mean=0) → None
        assert result["cer_cv"] is None

    def test_partial_identical_pairs(self) -> None:
        # Run1 == Run2, Run3 différent
        result = compute_multirun_stability(
            ["a b c", "a b c", "x y z"],
        )
        # 3 paires : (1,2) identiques, (1,3) (2,3) différentes
        assert result["identical_run_rate"] == pytest.approx(1.0 / 3.0)


# ──────────────────────────────────────────────────────────────────────────
# 5. _aligned_char_pairs
# ──────────────────────────────────────────────────────────────────────────


class TestAlignedCharPairs:
    def test_identical(self) -> None:
        pairs = _aligned_char_pairs("abc", "abc")
        assert pairs == [("a", "a"), ("b", "b"), ("c", "c")]

    def test_substitution(self) -> None:
        pairs = _aligned_char_pairs("abc", "axc")
        assert ("b", "x") in pairs

    def test_insertion_skipped(self) -> None:
        pairs = _aligned_char_pairs("ac", "abc")
        # 'b' inséré dans b → pas de paire bilatérale pour cette
        # position
        assert all(a != "" and b != "" for a, b in pairs)
        # Les caractères communs alignés sont a et c
        assert ("a", "a") in pairs
        assert ("c", "c") in pairs

    def test_both_empty(self) -> None:
        assert _aligned_char_pairs("", "") == []
