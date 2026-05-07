"""Tests Sprint 93 — A.II.7 : métriques d'image prédictives.

Couvre :

1. ``compute_paleographic_complexity`` :
   - cas trivial → score ≈ 0
   - cas extrême → score ≈ 1
   - poids surchargés
   - bornes [0, 1]
   - garde-fous (None, weights nuls)
2. ``compute_corpus_homogeneity`` :
   - corpus uniforme → score ≈ 0
   - corpus hétérogène → score haut
   - lt 2 docs → None
3. ``aggregate_corpus_predictive`` :
   - cas réaliste BnF
   - empty
4. Vue HTML :
   - adaptive
   - anti-injection
   - FR + EN
5. Complétude i18n FR/EN.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from picarones.measurements.image_predictive import (
    DEFAULT_COMPLEXITY_WEIGHTS,
    aggregate_corpus_predictive,
    compute_corpus_homogeneity,
    compute_paleographic_complexity,
)
from picarones.reports_v2.html.renderers.image_predictive import (
    build_image_predictive_html,
)


def _load_labels(lang: str) -> dict:
    p = (
        Path(__file__).parent.parent.parent
        / "picarones" / "reports_v2" / "i18n" / f"{lang}.json"
    )
    return json.loads(p.read_text(encoding="utf-8"))


# ──────────────────────────────────────────────────────────────────────────
# 1. compute_paleographic_complexity
# ──────────────────────────────────────────────────────────────────────────


class TestComplexity:
    def test_trivial_document(self) -> None:
        q = {"noise_level": 0.05, "sharpness_score": 0.95,
             "contrast_score": 0.9, "rotation_degrees": 0.0}
        r = compute_paleographic_complexity(q)
        # Très faible : ≤ 0.1
        assert r["score"] < 0.1

    def test_extreme_document(self) -> None:
        q = {"noise_level": 0.95, "sharpness_score": 0.05,
             "contrast_score": 0.05, "rotation_degrees": 30.0}
        r = compute_paleographic_complexity(q)
        # Très élevé : ≥ 0.9
        assert r["score"] > 0.9

    def test_score_bounds(self) -> None:
        # Valeurs fantaisistes hors plage → clip
        q = {"noise_level": 5.0, "sharpness_score": -1.0,
             "contrast_score": 2.0, "rotation_degrees": 1000.0}
        r = compute_paleographic_complexity(q)
        assert 0.0 <= r["score"] <= 1.0

    def test_components_returned(self) -> None:
        q = {"noise_level": 0.5, "sharpness_score": 0.5,
             "contrast_score": 0.5, "rotation_degrees": 15.0}
        r = compute_paleographic_complexity(q)
        assert set(r["components"].keys()) == {
            "noise", "blur", "low_contrast", "rotation",
        }

    def test_custom_weights(self) -> None:
        # Tout poids sur le bruit → score = noise_level
        q = {"noise_level": 0.7, "sharpness_score": 1.0,
             "contrast_score": 1.0, "rotation_degrees": 0}
        r = compute_paleographic_complexity(q, weights={
            "noise_level": 1.0, "blur": 0.0,
            "low_contrast": 0.0, "rotation": 0.0,
        })
        assert r["score"] == pytest.approx(0.7)

    def test_default_weights_sum_to_one(self) -> None:
        assert sum(DEFAULT_COMPLEXITY_WEIGHTS.values()) == pytest.approx(
            1.0,
        )

    def test_none_returns_none(self) -> None:
        assert compute_paleographic_complexity(None) is None
        assert compute_paleographic_complexity({}) is None

    def test_zero_weights_returns_none(self) -> None:
        q = {"noise_level": 0.5, "sharpness_score": 0.5,
             "contrast_score": 0.5, "rotation_degrees": 5}
        assert compute_paleographic_complexity(
            q, weights={"noise_level": 0, "blur": 0,
                        "low_contrast": 0, "rotation": 0},
        ) is None


# ──────────────────────────────────────────────────────────────────────────
# 2. compute_corpus_homogeneity
# ──────────────────────────────────────────────────────────────────────────


class TestHomogeneity:
    def test_uniform_corpus(self) -> None:
        q = {"noise_level": 0.1, "sharpness_score": 0.8,
             "contrast_score": 0.7, "rotation_degrees": 1.0}
        r = compute_corpus_homogeneity([q, q, q])
        # Variance nulle sur tous les docs
        assert r["score"] == 0.0

    def test_heterogeneous_corpus(self) -> None:
        a = {"noise_level": 0.05, "sharpness_score": 0.95,
             "contrast_score": 0.9, "rotation_degrees": 0.0}
        b = {"noise_level": 0.95, "sharpness_score": 0.05,
             "contrast_score": 0.05, "rotation_degrees": 30.0}
        r = compute_corpus_homogeneity([a, b, a, b])
        assert r["score"] > 0.5

    def test_lt_two_returns_none(self) -> None:
        assert compute_corpus_homogeneity([]) is None
        assert compute_corpus_homogeneity([{"noise_level": 0.5}]) is None

    def test_per_feature_keys(self) -> None:
        q1 = {"noise_level": 0.1, "sharpness_score": 0.8,
              "contrast_score": 0.7, "rotation_degrees": 0}
        q2 = {"noise_level": 0.5, "sharpness_score": 0.4,
              "contrast_score": 0.3, "rotation_degrees": 5}
        r = compute_corpus_homogeneity([q1, q2])
        assert "noise_level" in r["per_feature"]
        for slot in r["per_feature"].values():
            assert "mean" in slot and "stdev" in slot and "normalised" in slot


# ──────────────────────────────────────────────────────────────────────────
# 3. aggregate_corpus_predictive
# ──────────────────────────────────────────────────────────────────────────


class TestAggregate:
    def test_realistic_bnf(self) -> None:
        # Mélange de docs trivial et difficile
        docs = [
            {"noise_level": 0.1, "sharpness_score": 0.9,
             "contrast_score": 0.85, "rotation_degrees": 0},
            {"noise_level": 0.6, "sharpness_score": 0.3,
             "contrast_score": 0.4, "rotation_degrees": 12},
            {"noise_level": 0.15, "sharpness_score": 0.85,
             "contrast_score": 0.8, "rotation_degrees": 1},
        ]
        agg = aggregate_corpus_predictive(docs)
        assert agg["n_docs"] == 3
        # Min < mean < max
        assert agg["complexity_min"] < agg["complexity_mean"]
        assert agg["complexity_mean"] < agg["complexity_max"]
        assert agg["homogeneity"] is not None

    def test_empty_returns_none(self) -> None:
        assert aggregate_corpus_predictive([]) is None

    def test_single_doc_no_homogeneity(self) -> None:
        # 1 doc → complexity OK mais homogeneity None
        agg = aggregate_corpus_predictive([
            {"noise_level": 0.1, "sharpness_score": 0.8,
             "contrast_score": 0.7, "rotation_degrees": 0},
        ])
        assert agg["n_docs"] == 1
        assert agg["homogeneity"] is None


# ──────────────────────────────────────────────────────────────────────────
# 4. Vue HTML
# ──────────────────────────────────────────────────────────────────────────


class TestRender:
    def test_empty_returns_empty(self) -> None:
        assert build_image_predictive_html(None) == ""
        assert build_image_predictive_html({"n_docs": 0}) == ""

    def test_renders_complete(self) -> None:
        agg = aggregate_corpus_predictive([
            {"noise_level": 0.1, "sharpness_score": 0.9,
             "contrast_score": 0.85, "rotation_degrees": 0},
            {"noise_level": 0.6, "sharpness_score": 0.3,
             "contrast_score": 0.4, "rotation_degrees": 12},
        ])
        html = build_image_predictive_html(agg, _load_labels("fr"))
        assert "<table" in html
        assert "Complexité paléographique" in html
        assert "Homogénéité" in html

    def test_anti_injection(self) -> None:
        # On ne peut pas injecter via image_quality (champs
        # numériques) mais on vérifie tout de même qu'on n'expose
        # pas de label brut. Testons via une labels personnalisée.
        agg = aggregate_corpus_predictive([
            {"noise_level": 0.1, "sharpness_score": 0.8,
             "contrast_score": 0.7, "rotation_degrees": 0},
            {"noise_level": 0.5, "sharpness_score": 0.4,
             "contrast_score": 0.3, "rotation_degrees": 5},
        ])
        html = build_image_predictive_html(
            agg,
            {"imgpred_title": "<script>alert(1)</script>"},
        )
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html

    def test_renders_in_english(self) -> None:
        agg = aggregate_corpus_predictive([
            {"noise_level": 0.1, "sharpness_score": 0.8,
             "contrast_score": 0.7, "rotation_degrees": 0},
            {"noise_level": 0.5, "sharpness_score": 0.4,
             "contrast_score": 0.3, "rotation_degrees": 5},
        ])
        html = build_image_predictive_html(agg, _load_labels("en"))
        assert "Corpus image profile" in html


# ──────────────────────────────────────────────────────────────────────────
# 5. Complétude i18n
# ──────────────────────────────────────────────────────────────────────────


_KEYS = {
    "imgpred_title", "imgpred_note", "imgpred_complexity",
    "imgpred_homogeneity", "imgpred_score", "imgpred_mean",
    "imgpred_median", "imgpred_min", "imgpred_max", "imgpred_stdev",
    "imgpred_docs", "imgpred_feature", "imgpred_feat_mean",
    "imgpred_feat_stdev", "imgpred_feat_norm",
    "imgpred_feat_noise", "imgpred_feat_sharpness",
    "imgpred_feat_contrast", "imgpred_feat_rotation",
}


class TestI18n:
    def test_fr(self) -> None:
        d = _load_labels("fr")
        assert not _KEYS - d.keys()

    def test_en(self) -> None:
        d = _load_labels("en")
        assert not _KEYS - d.keys()
