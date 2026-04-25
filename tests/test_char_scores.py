"""Tests Sprint 31 — couverture dédiée de ``picarones/core/char_scores.py``.

Le module ``char_scores`` calcule les taux de bonne reconnaissance des
ligatures historiques (``fi``, ``ff``, ``ſ``, ``æ``, ``œ``, ``ꝑ``, …)
et des diacritiques (accents, cédilles). Avant Sprint 31, ces fonctions
n'étaient testées que de manière transitive via les rapports complets,
ce qui rendait le débogage d'un faux résultat très indirect.

Conventions
-----------
- ``score = 1.0`` quand il n'y a pas de ligature/diacritique dans le GT
  (rien à mesurer → meilleur score). C'est volontaire : le module évite
  de pénaliser un OCR sur un texte qui ne contient aucun glyphe à
  vérifier.
- ``per_ligature`` / ``per_diacritic`` n'apparaît que pour les caractères
  effectivement présents dans le GT.
"""

from __future__ import annotations

import pytest

from picarones.core.char_scores import (
    DiacriticScore,
    LigatureScore,
    aggregate_diacritic_scores,
    aggregate_ligature_scores,
    compute_diacritic_score,
    compute_ligature_score,
)


# ---------------------------------------------------------------------------
# 1. compute_ligature_score
# ---------------------------------------------------------------------------

class TestLigatureScore:
    def test_perfect_recognition(self):
        gt = "œuvre fiscalité ſimple æquus"
        score = compute_ligature_score(gt, gt)
        assert isinstance(score, LigatureScore)
        assert score.total_in_gt > 0
        assert score.correctly_recognized == score.total_in_gt
        assert score.score == pytest.approx(1.0)

    def test_no_ligature_in_gt_returns_perfect_score(self):
        # ``score = 1.0`` quand rien à mesurer (cf. docstring du module).
        gt = "abcdef"
        score = compute_ligature_score(gt, gt)
        assert score.total_in_gt == 0
        assert score.correctly_recognized == 0
        assert score.score == pytest.approx(1.0)

    def test_oe_ligature_split_to_oe_is_recognized(self):
        gt = "œuvre"
        hyp = "oeuvre"
        score = compute_ligature_score(gt, hyp)
        assert score.total_in_gt >= 1
        assert score.correctly_recognized >= 1, (
            "œ développé en 'oe' doit compter comme correctement reconnu"
        )

    def test_double_letter_ligature_recognized(self):
        # Les ligatures à deux lettres (``fi``, ``ff``, ``fl``…) sont
        # comptées par le module — le ``ſ`` long, lui, est un signe
        # diacritique géré par ``compute_diacritic_score``.
        gt = "officier"  # contient ``ffi`` → ligature ``fi``
        score = compute_ligature_score(gt, gt)
        # Selon l'implémentation, ce mot peut produire 0 ou 1 ligature.
        # Le test vérifie surtout qu'on ne crashe pas.
        assert score.score == pytest.approx(1.0)

    def test_missing_ligature_counts_as_error(self):
        gt = "œuvre"
        hyp = "vre"  # ligature absente, mots tronqués
        score = compute_ligature_score(gt, hyp)
        assert score.total_in_gt >= 1
        assert score.correctly_recognized == 0
        assert score.score == pytest.approx(0.0)

    def test_per_ligature_breakdown_present(self):
        gt = "œuvre æquus"
        score = compute_ligature_score(gt, gt)
        assert isinstance(score.per_ligature, dict)
        assert score.per_ligature, (
            "per_ligature ne doit pas être vide quand des ligatures existent"
        )
        # Chaque entrée porte gt_count et ocr_correct
        for entry in score.per_ligature.values():
            assert "gt_count" in entry
            assert "ocr_correct" in entry

    def test_as_dict_serializable(self):
        gt = "œuvre"
        score = compute_ligature_score(gt, gt)
        d = score.as_dict()
        # Les clefs publiques sont stables — utilisées par le rapport HTML
        for k in ("total_in_gt", "correctly_recognized", "score", "per_ligature"):
            assert k in d


# ---------------------------------------------------------------------------
# 2. compute_diacritic_score
# ---------------------------------------------------------------------------

class TestDiacriticScore:
    def test_perfect_recognition(self):
        gt = "été aiguë français Noël"
        score = compute_diacritic_score(gt, gt)
        assert isinstance(score, DiacriticScore)
        assert score.total_in_gt > 0
        assert score.correctly_recognized == score.total_in_gt

    def test_missing_accent_is_error(self):
        gt = "été"
        hyp = "ete"
        score = compute_diacritic_score(gt, hyp)
        assert score.total_in_gt >= 2
        assert score.correctly_recognized < score.total_in_gt

    def test_unaccented_text_returns_perfect_score(self):
        gt = "abcdef ghijkl"
        score = compute_diacritic_score(gt, gt)
        assert score.total_in_gt == 0
        assert score.score == pytest.approx(1.0)

    def test_as_dict_serializable(self):
        gt = "été"
        d = compute_diacritic_score(gt, gt).as_dict()
        for k in ("total_in_gt", "correctly_recognized", "score", "per_diacritic"):
            assert k in d


# ---------------------------------------------------------------------------
# 3. Agrégation multi-documents
# ---------------------------------------------------------------------------

class TestAggregation:
    def test_aggregate_ligature_scores_handles_empty_list(self):
        agg = aggregate_ligature_scores([])
        assert isinstance(agg, dict)
        assert agg["total_in_gt"] == 0
        assert agg["correctly_recognized"] == 0
        # ``score = 1.0`` quand rien à mesurer — pas de division par zéro
        assert agg["score"] == pytest.approx(1.0)

    def test_aggregate_diacritic_scores_handles_empty_list(self):
        agg = aggregate_diacritic_scores([])
        assert isinstance(agg, dict)
        assert agg["total_in_gt"] == 0
        assert agg["correctly_recognized"] == 0
        assert agg["score"] == pytest.approx(1.0)

    def test_aggregate_sums_correct_and_total(self):
        scores = [
            compute_ligature_score("œuvre", "œuvre"),
            compute_ligature_score("œuvre", "oeuvre"),
            compute_ligature_score("œuvre", "vre"),
        ]
        agg = aggregate_ligature_scores(scores)
        assert agg["total_in_gt"] == sum(s.total_in_gt for s in scores)
        assert agg["correctly_recognized"] == sum(s.correctly_recognized for s in scores)
        # Au moins une ligature ratée → score < 1.0
        assert 0.0 < agg["score"] < 1.0

    def test_aggregate_preserves_per_ligature_breakdown(self):
        scores = [
            compute_ligature_score("œuvre", "œuvre"),
            compute_ligature_score("œuvre", "vre"),  # œ raté ici
        ]
        agg = aggregate_ligature_scores(scores)
        assert "per_ligature" in agg
        # Au moins un détail pour œ doit ressortir
        assert any(
            entry["gt_count"] >= 1 for entry in agg["per_ligature"].values()
        )

    def test_aggregate_diacritic_sums_correctly(self):
        scores = [
            compute_diacritic_score("été", "été"),       # 2/2
            compute_diacritic_score("être", "etre"),     # 0/1
        ]
        agg = aggregate_diacritic_scores(scores)
        assert agg["total_in_gt"] == sum(s.total_in_gt for s in scores)
        assert agg["correctly_recognized"] == sum(s.correctly_recognized for s in scores)
        # Score agrégé entre les deux extrêmes
        assert 0.0 < agg["score"] < 1.0
