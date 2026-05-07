"""Tests Sprint 84 — A.II.5 : recherchabilité fuzzy.

Couvre :

1. ``levenshtein_distance`` : invariants + cas standard.
2. ``compute_searchability`` :
   - identité → recall = 1
   - aucun match → recall = 0
   - GT vide → recall None
   - hypothèse vide → recall = 0
   - max_distance = 0 → match exact uniquement
   - max_distance large
   - case insensitive par défaut
   - case sensitive opt-in
   - multiplicité (un token hyp utilisé une seule fois)
   - missed_tokens préserve la casse GT
   - ValueError pour max_distance < 0
3. Cas réaliste : CER élevé mais findability élevée.
4. ``searchability_recall_metric`` enregistré dans le registre typé.
"""

from __future__ import annotations

import pytest

from picarones.measurements.searchability import (
    compute_searchability,
    levenshtein_distance,
    searchability_recall_metric,
)


# ──────────────────────────────────────────────────────────────────────────
# 1. levenshtein_distance
# ──────────────────────────────────────────────────────────────────────────


class TestLevenshtein:
    def test_identity(self) -> None:
        assert levenshtein_distance("hello", "hello") == 0

    def test_one_substitution(self) -> None:
        assert levenshtein_distance("hello", "hallo") == 1

    def test_one_deletion(self) -> None:
        assert levenshtein_distance("hello", "helo") == 1

    def test_one_insertion(self) -> None:
        assert levenshtein_distance("helo", "hello") == 1

    def test_disjoint(self) -> None:
        assert levenshtein_distance("abc", "xyz") == 3

    def test_empty_left(self) -> None:
        assert levenshtein_distance("", "abc") == 3

    def test_empty_right(self) -> None:
        assert levenshtein_distance("abc", "") == 3

    def test_both_empty(self) -> None:
        assert levenshtein_distance("", "") == 0

    def test_classical_kitten(self) -> None:
        # Cas standard de la littérature : kitten → sitting = 3
        assert levenshtein_distance("kitten", "sitting") == 3


# ──────────────────────────────────────────────────────────────────────────
# 2. compute_searchability
# ──────────────────────────────────────────────────────────────────────────


class TestSearchability:
    def test_identical_texts(self) -> None:
        r = compute_searchability("le roi signa", "le roi signa")
        assert r["recall"] == 1.0
        assert r["missed_tokens"] == []
        assert r["n_gt_tokens"] == 3
        assert r["n_searchable"] == 3

    def test_completely_different(self) -> None:
        r = compute_searchability("alpha beta gamma", "rouge bleu vert")
        assert r["recall"] == 0.0
        assert sorted(r["missed_tokens"]) == ["alpha", "beta", "gamma"]

    def test_empty_gt_returns_none_recall(self) -> None:
        r = compute_searchability("", "anything")
        assert r["recall"] is None
        assert r["n_gt_tokens"] == 0

    def test_empty_hypothesis_zero_recall(self) -> None:
        r = compute_searchability("le roi", "")
        assert r["recall"] == 0.0
        assert r["missed_tokens"] == ["le", "roi"]

    def test_max_distance_zero_requires_exact(self) -> None:
        # « hallo » à distance 1 de « hello » → exclu si max_distance = 0
        r = compute_searchability(
            "hello world", "hallo world", max_distance=0,
        )
        assert r["n_searchable"] == 1  # « world » seulement
        assert "hello" in r["missed_tokens"]

    def test_max_distance_two_default(self) -> None:
        r = compute_searchability("Charles", "Charlse")  # 1 swap → distance 2
        assert r["recall"] == 1.0

    def test_max_distance_large_matches_loosely(self) -> None:
        r = compute_searchability(
            "completely different",
            "ompletely ifferent",
            max_distance=2,
        )
        assert r["recall"] == 1.0

    def test_case_insensitive_by_default(self) -> None:
        r = compute_searchability("Le Roi", "le roi")
        assert r["recall"] == 1.0

    def test_case_sensitive_opt_in(self) -> None:
        # « Le » distance 1 de « le » (casse) → exclu si exact
        r = compute_searchability(
            "Le Roi", "le roi", max_distance=0, case_sensitive=True,
        )
        assert r["n_searchable"] == 0

    def test_multiplicity_each_hyp_used_once(self) -> None:
        # GT : « le le », hyp : « le » → un seul matché
        r = compute_searchability("le le", "le")
        assert r["n_searchable"] == 1
        assert r["missed_tokens"] == ["le"]

    def test_missed_tokens_preserve_gt_case(self) -> None:
        r = compute_searchability("Charlemagne", "absent")
        assert r["missed_tokens"] == ["Charlemagne"]

    def test_negative_max_distance_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_searchability("a", "b", max_distance=-1)

    def test_default_max_distance_is_two(self) -> None:
        r = compute_searchability("a", "b")
        assert r["max_distance"] == 2


# ──────────────────────────────────────────────────────────────────────────
# 3. Cas réaliste : findability robuste à un CER élevé
# ──────────────────────────────────────────────────────────────────────────


class TestRealisticCase:
    def test_high_cer_low_findability(self) -> None:
        """Erreurs concentrées sur quelques mots → findability faible."""
        gt = "le roi Charles VII signa la charte royale en 1450"
        # « Charles » ↔ « Charlemagne » : distance 5 → non retrouvé
        # « 1450 » ↔ « 1480 » : distance 1 → retrouvé
        # « charte » remplacé par « lettre » : distance 5 → non retrouvé
        hyp = "le roi Charlemagne VII signa la lettre royale en 1480"
        r = compute_searchability(gt, hyp)
        assert r["n_searchable"] < r["n_gt_tokens"]
        assert "Charles" in r["missed_tokens"]
        assert "charte" in r["missed_tokens"]

    def test_high_cer_high_findability(self) -> None:
        """Erreurs réparties (≤ 2 par mot) → findability élevée."""
        gt = "maistre Pierre du Bois écrivit cette charte"
        # 1 faute par mot, distance ≤ 2
        hyp = "maitre Piere du Boys ecrivit cete charte"
        r = compute_searchability(gt, hyp)
        # Le CER est non négligeable mais tous les mots restent
        # retrouvables en mode fuzzy
        assert r["recall"] == 1.0


# ──────────────────────────────────────────────────────────────────────────
# 4. Intégration registre typé
# ──────────────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_metric_registered(self) -> None:
        from picarones.core.metric_registry import select_metrics
        from picarones.domain.artifacts import ArtifactType

        metrics = select_metrics(
            (ArtifactType.TEXT, ArtifactType.TEXT),
        )
        names = [m.name for m in metrics]
        assert "searchability_recall" in names

    def test_metric_callable(self) -> None:
        v = searchability_recall_metric("hello world", "helo world")
        assert v == 1.0

    def test_metric_returns_zero_for_empty_gt(self) -> None:
        # Convention : registre typé attend un float, pas None
        v = searchability_recall_metric("", "anything")
        assert v == 0.0

    def test_metric_via_compute_at_junction(self) -> None:
        from picarones.core.metric_registry import compute_at_junction
        from picarones.domain.artifacts import ArtifactType

        results = compute_at_junction(
            "le roi", "le roi",
            (ArtifactType.TEXT, ArtifactType.TEXT),
        )
        assert "searchability_recall" in results
        assert results["searchability_recall"] == 1.0
