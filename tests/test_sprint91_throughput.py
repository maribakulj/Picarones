"""Tests Sprint 91 — A.II.6 : throughput effectif + coût marginal.

Couvre :

1. ``compute_effective_throughput`` : formule, garde-fous, cas
   limite.
2. ``aggregate_effective_throughput`` : agrégation par moteur.
3. ``compute_marginal_cost`` : cas standard, dominé, non
   comparable.
4. ``compute_marginal_cost_matrix`` : tri, paires, n-engines.
5. Cas réaliste BnF.
6. Vue HTML : adaptive, anti-injection, FR + EN.
7. Complétude i18n FR/EN.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from picarones.measurements.marginal_cost import (
    compute_marginal_cost,
    compute_marginal_cost_matrix,
)
from picarones.measurements.throughput import (
    aggregate_effective_throughput,
    compute_effective_throughput,
)
from picarones.report.throughput_render import build_throughput_html


def _load_labels(lang: str) -> dict:
    p = (
        Path(__file__).parent.parent
        / "picarones" / "report" / "i18n" / f"{lang}.json"
    )
    return json.loads(p.read_text(encoding="utf-8"))


# ──────────────────────────────────────────────────────────────────────────
# 1. compute_effective_throughput
# ──────────────────────────────────────────────────────────────────────────


class TestEffectiveThroughput:
    def test_basic_formula(self) -> None:
        # 100 pages en 600s OCR + 50 erreurs × 5s = 250s correction
        # → 850s total, 100 pages → 423.53 pages/h
        r = compute_effective_throughput(100, 600, 50)
        assert r is not None
        assert r["correction_time_seconds"] == 250.0
        assert r["total_seconds"] == 850.0
        assert r["pages_per_hour_effective"] == pytest.approx(
            100 / 850 * 3600,
        )

    def test_raw_throughput(self) -> None:
        r = compute_effective_throughput(100, 600, 0)
        # Pas d'erreurs → effective == raw
        assert r["pages_per_hour_effective"] == r["pages_per_hour_raw"]
        assert r["drag_ratio"] == 0.0

    def test_custom_time_per_error(self) -> None:
        r = compute_effective_throughput(
            100, 600, 50, time_per_error_seconds=10,
        )
        assert r["correction_time_seconds"] == 500.0

    def test_zero_pages_returns_none(self) -> None:
        assert compute_effective_throughput(0, 100, 5) is None

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_effective_throughput(10, -1, 0)
        with pytest.raises(ValueError):
            compute_effective_throughput(10, 1, -5)
        with pytest.raises(ValueError):
            compute_effective_throughput(
                10, 1, 0, time_per_error_seconds=-1,
            )

    def test_zero_duration_zero_errors_returns_none(self) -> None:
        # Aucun temps total → indéfini
        assert compute_effective_throughput(10, 0, 0) is None

    def test_drag_ratio_high_when_many_errors(self) -> None:
        r = compute_effective_throughput(100, 200, 200)
        # 200s OCR + 1000s correction = 1200s, drag = 1000/1200 ≈ 0.83
        assert r["drag_ratio"] > 0.8


# ──────────────────────────────────────────────────────────────────────────
# 2. aggregate_effective_throughput
# ──────────────────────────────────────────────────────────────────────────


class TestAggregate:
    def test_aggregates_multiple(self) -> None:
        agg = aggregate_effective_throughput([
            {"engine_name": "a", "n_pages": 10,
             "duration_seconds": 60, "n_errors": 5},
            {"engine_name": "b", "n_pages": 20,
             "duration_seconds": 120, "n_errors": 0},
        ])
        assert agg is not None
        names = [e["engine_name"] for e in agg["engines"]]
        assert names == ["a", "b"]

    def test_empty_returns_none(self) -> None:
        assert aggregate_effective_throughput([]) is None

    def test_skips_invalid(self) -> None:
        agg = aggregate_effective_throughput([
            {"engine_name": "a", "n_pages": 0, "duration_seconds": 0,
             "n_errors": 0},  # n_pages=0 → None, ignoré
            {"engine_name": "b", "n_pages": 10,
             "duration_seconds": 60, "n_errors": 0},
        ])
        assert len(agg["engines"]) == 1


# ──────────────────────────────────────────────────────────────────────────
# 3. compute_marginal_cost
# ──────────────────────────────────────────────────────────────────────────


class TestMarginalCost:
    def test_standard_case(self) -> None:
        # Tess (0€, 200 err) → Mistral (5€, 50 err) :
        # 5/150 = 0.033€ par erreur évitée
        r = compute_marginal_cost(0, 200, 5, 50)
        assert r["cost_per_avoided_error"] == pytest.approx(5 / 150)
        assert r["n_errors_avoided"] == 150
        assert r["dominated"] is False

    def test_dominated_case(self) -> None:
        # B moins cher ET plus précis → dominé
        r = compute_marginal_cost(10, 100, 8, 60)
        assert r["dominated"] is True
        assert r["cost_per_avoided_error"] < 0

    def test_b_worse_returns_none(self) -> None:
        assert compute_marginal_cost(0, 50, 5, 80) is None

    def test_equal_errors_returns_none(self) -> None:
        # Pas de réduction d'erreur → indéfini
        assert compute_marginal_cost(0, 100, 10, 100) is None

    def test_invalid_returns_none(self) -> None:
        assert compute_marginal_cost(None, 100, 10, 50) is None


# ──────────────────────────────────────────────────────────────────────────
# 4. compute_marginal_cost_matrix
# ──────────────────────────────────────────────────────────────────────────


class TestMarginalMatrix:
    def test_basic(self) -> None:
        m = compute_marginal_cost_matrix({
            "tess":    {"cost": 0, "errors": 200},
            "mistral": {"cost": 5, "errors": 50},
        })
        assert m is not None
        # Une seule paire valide : tess → mistral
        assert len(m["pairs"]) == 1
        p = m["pairs"][0]
        assert p["engine_a"] == "tess"
        assert p["engine_b"] == "mistral"

    def test_sorted_by_marginal_cost_ascending(self) -> None:
        m = compute_marginal_cost_matrix({
            "a": {"cost": 0,  "errors": 100},
            "b": {"cost": 5,  "errors": 50},
            "c": {"cost": 50, "errors": 25},
        })
        costs = [p["cost_per_avoided_error"] for p in m["pairs"]]
        assert costs == sorted(costs)

    def test_lt_two_returns_none(self) -> None:
        assert compute_marginal_cost_matrix({}) is None
        assert compute_marginal_cost_matrix({"a": {"cost": 0, "errors": 0}}) is None

    def test_skips_invalid_data(self) -> None:
        m = compute_marginal_cost_matrix({
            "a": {"cost": 0,    "errors": 100},
            "b": {"cost": None, "errors": 50},
        })
        assert m is None  # toutes les paires impliquant b échouent


# ──────────────────────────────────────────────────────────────────────────
# 5. Cas réaliste BnF
# ──────────────────────────────────────────────────────────────────────────


class TestRealistic:
    def test_local_beats_fast_cloud_on_effective(self) -> None:
        # Tesseract local : 100 pages en 600s OCR, 50 erreurs
        # GPT-4o cloud : 100 pages en 200s OCR mais 200 erreurs
        tess = compute_effective_throughput(100, 600, 50)
        gpt = compute_effective_throughput(100, 200, 200)
        # Brut : gpt 4× plus rapide
        assert gpt["pages_per_hour_raw"] > tess["pages_per_hour_raw"]
        # Effectif : tesseract gagne
        assert (
            tess["pages_per_hour_effective"]
            > gpt["pages_per_hour_effective"]
        )


# ──────────────────────────────────────────────────────────────────────────
# 6. Vue HTML
# ──────────────────────────────────────────────────────────────────────────


class TestRender:
    def test_empty_returns_empty(self) -> None:
        assert build_throughput_html(None) == ""
        assert build_throughput_html({"engines": []}) == ""

    def test_renders_table(self) -> None:
        agg = aggregate_effective_throughput([
            {"engine_name": "tess", "n_pages": 100,
             "duration_seconds": 600, "n_errors": 50},
        ])
        html = build_throughput_html(agg, _load_labels("fr"))
        assert "<table" in html
        assert "tess" in html
        # Le drag ≈ 29.4 % apparaît
        assert "29.4" in html

    def test_anti_injection(self) -> None:
        agg = aggregate_effective_throughput([
            {"engine_name": "<script>alert(1)</script>",
             "n_pages": 10, "duration_seconds": 60, "n_errors": 0},
        ])
        html = build_throughput_html(agg, _load_labels("fr"))
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html

    def test_renders_in_english(self) -> None:
        agg = aggregate_effective_throughput([
            {"engine_name": "tess", "n_pages": 10,
             "duration_seconds": 60, "n_errors": 0},
        ])
        html = build_throughput_html(agg, _load_labels("en"))
        assert "Effective throughput" in html

    def test_sorted_by_effective_descending(self) -> None:
        agg = aggregate_effective_throughput([
            {"engine_name": "slow", "n_pages": 100,
             "duration_seconds": 3600, "n_errors": 0},
            {"engine_name": "fast", "n_pages": 100,
             "duration_seconds": 100, "n_errors": 0},
        ])
        html = build_throughput_html(agg, _load_labels("fr"))
        assert html.index("fast") < html.index("slow")


# ──────────────────────────────────────────────────────────────────────────
# 7. Complétude i18n
# ──────────────────────────────────────────────────────────────────────────


_KEYS = {
    "throughput_title", "throughput_note", "throughput_engine",
    "throughput_raw", "throughput_effective", "throughput_drag",
    "throughput_pages", "throughput_errors",
}


class TestI18n:
    def test_fr(self) -> None:
        d = _load_labels("fr")
        assert not _KEYS - d.keys()

    def test_en(self) -> None:
        d = _load_labels("en")
        assert not _KEYS - d.keys()
