"""Tests Sprint 53 — Reading order F1 (ICDAR 2015).

Couvre :

1. **Cas canoniques** :
   - Séquences identiques → F1 = 1.0
   - Séquences strictement inversées → F1 = 0.0
   - Permutation locale → F1 calculé sur les paires conservées
   - Insertion d'une région → F1 = recall × precision sur paires
2. **Cas dégénérés** :
   - Une séquence vide → F1 = 0
   - Deux séquences vides → F1 = 0
   - Une seule région → pas de paire, F1 = 0
   - Doublons dans une séquence → traitement déterministe
3. **Comptages détaillés** :
   - TP, FP, FN cohérents
   - common/ref_only/hyp_only correctement séparés
4. **Intégration registre typé** :
   - ``reading_order_f1`` est sélectionné pour la jonction
     ``(READING_ORDER, READING_ORDER)``
   - Le shortcut retourne la même valeur que
     ``compute_reading_order_metrics["f1"]``
"""

from __future__ import annotations

import pytest

from picarones.core.metric_registry import compute_at_junction, select_metrics
from picarones.core.modules import ArtifactType
from picarones.core.reading_order import (
    compute_reading_order_metrics,
    reading_order_f1,
)


# ──────────────────────────────────────────────────────────────────────────
# 1. Cas canoniques
# ──────────────────────────────────────────────────────────────────────────


class TestCanonicalCases:
    def test_identical_sequences_f1_one(self) -> None:
        m = compute_reading_order_metrics(
            ["r1", "r2", "r3", "r4"],
            ["r1", "r2", "r3", "r4"],
        )
        assert m["f1"] == pytest.approx(1.0)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)
        assert m["false_positives"] == 0
        assert m["false_negatives"] == 0

    def test_strictly_reversed_f1_zero(self) -> None:
        m = compute_reading_order_metrics(
            ["a", "b", "c"],
            ["c", "b", "a"],
        )
        # Les 3 paires (a,b), (a,c), (b,c) sont toutes inversées
        # côté hypothèse → 0 TP, 3 FN, 3 FP, F1 = 0
        assert m["f1"] == 0.0
        assert m["true_positives"] == 0
        assert m["false_positives"] == 3
        assert m["false_negatives"] == 3

    def test_local_permutation(self) -> None:
        # GT : a, b, c, d → 6 paires.  Échange interne b↔c → 5 paires
        # préservées (toutes sauf b-c qui devient c-b).
        m = compute_reading_order_metrics(
            ["a", "b", "c", "d"],
            ["a", "c", "b", "d"],
        )
        assert m["true_positives"] == 5
        assert m["false_negatives"] == 1
        assert m["false_positives"] == 1
        assert m["f1"] == pytest.approx(5 / 6)

    def test_insertion_preserves_existing_pairs(self) -> None:
        # GT : a, b, c → 3 paires.  Hypothèse insère X au milieu :
        # a, X, b, c → 6 paires (a-X, a-b, a-c, X-b, X-c, b-c).
        # 3 TP (paires GT préservées) + 3 FP (paires inventées avec X).
        m = compute_reading_order_metrics(
            ["a", "b", "c"],
            ["a", "X", "b", "c"],
        )
        assert m["true_positives"] == 3
        assert m["false_negatives"] == 0
        assert m["false_positives"] == 3
        # Recall = 1, precision = 0.5, F1 = 2/3
        assert m["recall"] == pytest.approx(1.0)
        assert m["precision"] == pytest.approx(0.5)
        assert m["f1"] == pytest.approx(2 / 3)

    def test_deletion_preserves_remaining_pairs(self) -> None:
        # GT : a, b, c → 3 paires.  Hypothèse supprime b : a, c → 1 paire.
        m = compute_reading_order_metrics(
            ["a", "b", "c"],
            ["a", "c"],
        )
        # TP = 1 (paire a-c), FN = 2 (a-b, b-c manquent côté hyp)
        assert m["true_positives"] == 1
        assert m["false_negatives"] == 2
        assert m["false_positives"] == 0


# ──────────────────────────────────────────────────────────────────────────
# 2. Cas dégénérés
# ──────────────────────────────────────────────────────────────────────────


class TestDegenerateCases:
    def test_both_empty(self) -> None:
        m = compute_reading_order_metrics([], [])
        # Convention : pas de récompense gratuite
        assert m["f1"] == 0.0
        assert m["true_positives"] == 0

    def test_only_reference_empty(self) -> None:
        m = compute_reading_order_metrics([], ["a", "b"])
        assert m["f1"] == 0.0
        # TP = 0 par construction
        assert m["true_positives"] == 0
        # 1 paire FP côté hypothèse
        assert m["false_positives"] == 1

    def test_only_hypothesis_empty(self) -> None:
        m = compute_reading_order_metrics(["a", "b"], [])
        assert m["f1"] == 0.0
        # 1 FN côté GT
        assert m["false_negatives"] == 1

    def test_single_region(self) -> None:
        # Pas de paire possible avec une seule région
        m = compute_reading_order_metrics(["a"], ["a"])
        assert m["n_ref_pairs"] == 0
        assert m["n_hyp_pairs"] == 0
        assert m["f1"] == 0.0  # convention de bord (pas de paire à matcher)

    def test_none_inputs(self) -> None:
        m = compute_reading_order_metrics(None, None)
        assert m["f1"] == 0.0

    def test_duplicates_treated_first_occurrence(self) -> None:
        # GT : a, b, a, c → on garde "a, b, c" (première occurrence)
        # → 3 paires.  Hypothèse : a, b, c → 3 paires.  F1 = 1.
        m = compute_reading_order_metrics(
            ["a", "b", "a", "c"],
            ["a", "b", "c"],
        )
        assert m["f1"] == pytest.approx(1.0)


# ──────────────────────────────────────────────────────────────────────────
# 3. Comptages détaillés
# ──────────────────────────────────────────────────────────────────────────


class TestDetailedCounts:
    def test_common_and_disjoint_regions(self) -> None:
        m = compute_reading_order_metrics(
            ["a", "b", "c"],
            ["b", "c", "d"],
        )
        assert m["common_regions"] == ["b", "c"]
        assert m["ref_only_regions"] == ["a"]
        assert m["hyp_only_regions"] == ["d"]

    def test_n_pairs_consistent(self) -> None:
        m = compute_reading_order_metrics(
            ["a", "b", "c", "d"],
            ["e", "f"],
        )
        # GT : C(4, 2) = 6 paires
        assert m["n_ref_pairs"] == 6
        # Hyp : C(2, 2) = 1 paire
        assert m["n_hyp_pairs"] == 1


# ──────────────────────────────────────────────────────────────────────────
# 4. Intégration registre typé
# ──────────────────────────────────────────────────────────────────────────


class TestRegistryIntegration:
    def test_metric_registered_for_reading_order_pair(self) -> None:
        # Force l'import qui peuple le registre
        import picarones.core.reading_order  # noqa: F401

        selected = select_metrics(
            (ArtifactType.READING_ORDER, ArtifactType.READING_ORDER),
        )
        names = {spec.name for spec in selected}
        assert "reading_order_f1" in names

    def test_compute_at_junction_returns_f1(self) -> None:
        out = compute_at_junction(
            ["a", "b", "c"],
            ["a", "b", "c"],
            (ArtifactType.READING_ORDER, ArtifactType.READING_ORDER),
        )
        assert out["reading_order_f1"] == pytest.approx(1.0)

    def test_shortcut_matches_full_call(self) -> None:
        ref = ["r1", "r2", "r3", "r4"]
        hyp = ["r1", "r3", "r2", "r4"]
        full = compute_reading_order_metrics(ref, hyp)
        assert reading_order_f1(ref, hyp) == pytest.approx(full["f1"])
