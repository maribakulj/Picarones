"""Tests Sprint 89 — A.II.8b : spécialisation inter-moteurs.

Couvre :

1. ``compute_specialization_score`` : symétrie, plage [0, 1].
2. ``classify_specialization`` : seuils par défaut + custom.
3. ``compute_specialization_matrix`` : structure, symétrie, max_pair.
4. ``top_specialized_pairs`` : tri, n, min_score.
5. Vue HTML : adaptive, anti-injection, FR + EN.
6. Complétude i18n FR/EN.
"""

from __future__ import annotations

import json
from pathlib import Path

from picarones.measurements.specialization import (
    DEFAULT_THRESHOLDS,
    classify_specialization,
    compute_specialization_matrix,
    compute_specialization_score,
    top_specialized_pairs,
)
from picarones.reports_v2.html.renderers.specialization import (
    build_specialization_html,
)


def _load_labels(lang: str) -> dict:
    p = (
        Path(__file__).parent.parent.parent
        / "picarones" / "reports_v2" / "i18n" / f"{lang}.json"
    )
    return json.loads(p.read_text(encoding="utf-8"))


# ──────────────────────────────────────────────────────────────────────────
# 1. compute_specialization_score
# ──────────────────────────────────────────────────────────────────────────


class TestScore:
    def test_identical_profiles_zero(self) -> None:
        tax = {"a": 50, "b": 50}
        assert compute_specialization_score(tax, tax) < 0.001

    def test_disjoint_profiles_one(self) -> None:
        tax_a = {"a": 100}
        tax_b = {"b": 100}
        assert compute_specialization_score(tax_a, tax_b) > 0.95

    def test_symmetric(self) -> None:
        a = {"x": 70, "y": 30}
        b = {"x": 20, "y": 80}
        s_ab = compute_specialization_score(a, b)
        s_ba = compute_specialization_score(b, a)
        assert abs(s_ab - s_ba) < 1e-9

    def test_bounded_zero_one(self) -> None:
        a = {"x": 1, "y": 0, "z": 0}
        b = {"x": 0, "y": 0, "z": 1}
        score = compute_specialization_score(a, b)
        assert 0.0 <= score <= 1.0


# ──────────────────────────────────────────────────────────────────────────
# 2. classify_specialization
# ──────────────────────────────────────────────────────────────────────────


class TestClassify:
    def test_below_similar_threshold(self) -> None:
        assert classify_specialization(0.05) == "similar"

    def test_distinct_band(self) -> None:
        assert classify_specialization(0.20) == "distinct"

    def test_highly_specialized_above(self) -> None:
        assert classify_specialization(0.50) == "highly_specialized"

    def test_custom_thresholds(self) -> None:
        custom = (("low", 0.5), ("high", 1.01))
        assert classify_specialization(0.30, custom) == "low"
        assert classify_specialization(0.80, custom) == "high"

    def test_default_thresholds_exposed(self) -> None:
        assert isinstance(DEFAULT_THRESHOLDS, tuple)
        assert len(DEFAULT_THRESHOLDS) >= 2


# ──────────────────────────────────────────────────────────────────────────
# 3. compute_specialization_matrix
# ──────────────────────────────────────────────────────────────────────────


class TestMatrix:
    def test_returns_none_when_lt_two(self) -> None:
        assert compute_specialization_matrix({}) is None
        assert compute_specialization_matrix({"a": {"x": 1}}) is None

    def test_diagonal_zero(self) -> None:
        tax = {
            "a": {"x": 1, "y": 0},
            "b": {"x": 0, "y": 1},
        }
        m = compute_specialization_matrix(tax)
        for i in range(len(m["engines"])):
            assert m["matrix"][i][i] == 0.0

    def test_symmetric(self) -> None:
        tax = {
            "a": {"x": 1, "y": 0},
            "b": {"x": 0, "y": 1},
            "c": {"x": 1, "y": 1},
        }
        m = compute_specialization_matrix(tax)
        n = len(m["engines"])
        for i in range(n):
            for j in range(n):
                assert m["matrix"][i][j] == m["matrix"][j][i]

    def test_max_pair_identifies_most_specialized(self) -> None:
        # A vs B totalement disjoints, C similaire à A.
        tax = {
            "a": {"x": 100, "y": 0},
            "b": {"x": 0, "y": 100},
            "c": {"x": 95, "y": 5},
        }
        m = compute_specialization_matrix(tax)
        # La paire la plus spécialisée doit être (a, b)
        assert set(m["max_pair"]) == {"a", "b"}


# ──────────────────────────────────────────────────────────────────────────
# 4. top_specialized_pairs
# ──────────────────────────────────────────────────────────────────────────


class TestTop:
    def _matrix(self) -> dict:
        return compute_specialization_matrix({
            "a": {"x": 100, "y": 0},
            "b": {"x": 0, "y": 100},
            "c": {"x": 95, "y": 5},
        })

    def test_sorted_descending(self) -> None:
        pairs = top_specialized_pairs(self._matrix(), n=10)
        scores = [p["score"] for p in pairs]
        assert scores == sorted(scores, reverse=True)

    def test_caps_at_n(self) -> None:
        pairs = top_specialized_pairs(self._matrix(), n=1)
        assert len(pairs) == 1

    def test_min_score_filter(self) -> None:
        pairs = top_specialized_pairs(
            self._matrix(), n=10, min_score=0.99,
        )
        # Seules les paires (a,b) et éventuellement (b,c) au-dessus
        assert all(p["score"] >= 0.99 for p in pairs)

    def test_none_input_returns_empty(self) -> None:
        assert top_specialized_pairs(None) == []


# ──────────────────────────────────────────────────────────────────────────
# 5. Vue HTML
# ──────────────────────────────────────────────────────────────────────────


class TestRender:
    def test_empty_returns_empty(self) -> None:
        assert build_specialization_html(None) == ""
        assert build_specialization_html({}) == ""

    def test_single_engine_returns_empty(self) -> None:
        assert build_specialization_html({"a": {"x": 1}}) == ""

    def test_renders_table(self) -> None:
        tax = {
            "tess": {"visual_confusion": 80, "lacuna": 20},
            "pero": {"visual_confusion": 5, "lacuna": 95},
        }
        html = build_specialization_html(tax, _load_labels("fr"))
        assert "<table" in html
        assert "tess" in html
        assert "pero" in html
        # Catégorie traduite
        assert "Forte spécialisation" in html

    def test_anti_injection(self) -> None:
        tax = {
            "<script>alert(1)</script>": {"x": 100},
            "pero": {"y": 100},
        }
        html = build_specialization_html(tax, _load_labels("fr"))
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html

    def test_renders_in_english(self) -> None:
        tax = {
            "a": {"x": 100, "y": 0},
            "b": {"x": 0, "y": 100},
        }
        html = build_specialization_html(tax, _load_labels("en"))
        assert "Inter-engine specialisation" in html
        assert "Highly specialised" in html


# ──────────────────────────────────────────────────────────────────────────
# 6. Complétude i18n
# ──────────────────────────────────────────────────────────────────────────


_KEYS = {
    "specialization_title", "specialization_note",
    "specialization_engine_a", "specialization_engine_b",
    "specialization_score", "specialization_category",
    "specialization_cat_similar", "specialization_cat_distinct",
    "specialization_cat_highly_specialized",
}


class TestI18n:
    def test_fr(self) -> None:
        d = _load_labels("fr")
        assert not _KEYS - d.keys()

    def test_en(self) -> None:
        d = _load_labels("en")
        assert not _KEYS - d.keys()
