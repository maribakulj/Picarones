"""Tests Sprint 92 — A.II.9 : métriques longitudinales.

Couvre :

1. ``compute_linear_trend`` : pente, R², garde-fous.
2. ``detect_change_point`` : index correct, garde-fous.
3. ``compute_engine_longitudinal`` : intégration entries.
4. ``compute_corpus_longitudinal`` : agrégation multi-moteurs.
5. Détecteur ``regression_in_history`` :
   - silence sans data
   - silence si tendance plate
   - HIGH si Δ ≥ 5 pts
   - réagit à change-point seul
   - traçabilité anti-hallucination FR + EN.
6. Vue HTML : adaptive, anti-injection, FR + EN.
7. Complétude i18n.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from picarones.measurements.longitudinal import (
    compute_corpus_longitudinal,
    compute_engine_longitudinal,
    compute_linear_trend,
    detect_change_point,
)
from picarones.measurements.narrative import build_synthesis
from picarones.measurements.narrative.detectors import detect_regression_in_history
from picarones.core.facts import FactImportance, FactType
from picarones.report.longitudinal_render import build_longitudinal_html


def _load_labels(lang: str) -> dict:
    p = (
        Path(__file__).parent.parent.parent
        / "picarones" / "reports_v2" / "i18n" / f"{lang}.json"
    )
    return json.loads(p.read_text(encoding="utf-8"))


# ──────────────────────────────────────────────────────────────────────────
# 1. compute_linear_trend
# ──────────────────────────────────────────────────────────────────────────


class TestLinearTrend:
    def test_perfect_trend(self) -> None:
        series = [
            ("2025-01-01", 0.04), ("2025-02-01", 0.05),
            ("2025-03-01", 0.06),
        ]
        t = compute_linear_trend(series)
        assert t.r_squared > 0.99
        assert t.slope > 0  # CER monte → pente positive
        assert t.n_runs == 3

    def test_flat_series(self) -> None:
        series = [
            ("2025-01-01", 0.05), ("2025-02-01", 0.05),
            ("2025-03-01", 0.05),
        ]
        t = compute_linear_trend(series)
        # Série plate : pente ≈ 0. R² mathématiquement indéterminé
        # (variance nulle sur y) ; le code accepte 0 ou 1 selon
        # l'arrondi flottant.
        assert t.slope == pytest.approx(0.0, abs=1e-9)
        assert t.r_squared in (0.0, 1.0) or 0.0 <= t.r_squared <= 1.0

    def test_lt_two_returns_none(self) -> None:
        assert compute_linear_trend([("2025-01-01", 0.05)]) is None
        assert compute_linear_trend([]) is None

    def test_invalid_timestamps_skipped(self) -> None:
        # Tous invalides → < 2 valides
        assert compute_linear_trend([
            ("invalid", 0.05), ("garbage", 0.06),
        ]) is None

    def test_same_timestamp_returns_none(self) -> None:
        # Tous les t identiques → variance nulle
        assert compute_linear_trend([
            ("2025-01-01", 0.05), ("2025-01-01", 0.06),
            ("2025-01-01", 0.07),
        ]) is None


# ──────────────────────────────────────────────────────────────────────────
# 2. detect_change_point
# ──────────────────────────────────────────────────────────────────────────


class TestChangePoint:
    def test_clean_break(self) -> None:
        # 3 points à 0.04 puis 3 points à 0.07
        series = [
            ("2025-01-01", 0.04), ("2025-01-15", 0.04),
            ("2025-02-01", 0.04), ("2025-02-15", 0.07),
            ("2025-03-01", 0.07), ("2025-03-15", 0.07),
        ]
        cp = detect_change_point(series, min_segment_size=3)
        assert cp is not None
        assert cp.index == 3
        assert cp.delta == pytest.approx(0.03)

    def test_too_few_points(self) -> None:
        series = [
            ("2025-01-01", 0.04), ("2025-02-01", 0.05),
        ]
        assert detect_change_point(series, min_segment_size=3) is None

    def test_uniform_series_returns_change_with_delta_zero(self) -> None:
        series = [
            ("2025-01-01", 0.05), ("2025-02-01", 0.05),
            ("2025-03-01", 0.05), ("2025-04-01", 0.05),
            ("2025-05-01", 0.05), ("2025-06-01", 0.05),
        ]
        cp = detect_change_point(series, min_segment_size=3)
        # delta = 0
        assert cp is not None
        assert abs(cp.delta) < 1e-9


# ──────────────────────────────────────────────────────────────────────────
# 3. compute_engine_longitudinal
# ──────────────────────────────────────────────────────────────────────────


class TestEngineLongitudinal:
    def _entries(self) -> list[dict]:
        return [
            {"engine_name": "tess", "corpus_name": "bnf",
             "timestamp": ts, "cer_mean": cer}
            for ts, cer in [
                ("2025-01-01", 0.04), ("2025-02-01", 0.045),
                ("2025-03-01", 0.05), ("2025-04-01", 0.06),
                ("2025-05-01", 0.07), ("2025-06-01", 0.08),
            ]
        ]

    def test_basic(self) -> None:
        r = compute_engine_longitudinal(
            self._entries(), "tess", corpus_name="bnf",
        )
        assert r is not None
        assert r["n_runs"] == 6
        assert r["trend"]["slope"] > 0
        assert r["absolute_delta_pct"] == pytest.approx(4.0, abs=0.01)

    def test_filters_corpus(self) -> None:
        entries = self._entries() + [
            {"engine_name": "tess", "corpus_name": "other",
             "timestamp": "2025-07-01", "cer_mean": 0.99},
        ]
        r = compute_engine_longitudinal(
            entries, "tess", corpus_name="bnf",
        )
        # L'entrée "other" ne doit pas polluer
        assert r["n_runs"] == 6

    def test_min_runs_threshold(self) -> None:
        # min_runs_for_trend=10 > n_runs=6
        r = compute_engine_longitudinal(
            self._entries(), "tess", corpus_name="bnf",
            min_runs_for_trend=10,
        )
        assert r is None

    def test_change_point_threshold(self) -> None:
        # Avec un threshold immense, le change-point est supprimé
        r = compute_engine_longitudinal(
            self._entries(), "tess",
            change_point_threshold=1.0,
        )
        assert r["change_point"] is None


# ──────────────────────────────────────────────────────────────────────────
# 4. compute_corpus_longitudinal
# ──────────────────────────────────────────────────────────────────────────


class TestCorpusLongitudinal:
    def test_multiple_engines(self) -> None:
        entries: list[dict] = []
        for engine in ("tess", "pero"):
            for i, cer in enumerate([0.04, 0.045, 0.05, 0.06]):
                entries.append({
                    "engine_name": engine, "corpus_name": "bnf",
                    "timestamp": f"2025-0{i + 1}-01",
                    "cer_mean": cer,
                })
        out = compute_corpus_longitudinal(entries, corpus_name="bnf")
        names = [e["engine_name"] for e in out]
        assert "tess" in names
        assert "pero" in names

    def test_empty(self) -> None:
        assert compute_corpus_longitudinal([]) == []


# ──────────────────────────────────────────────────────────────────────────
# 5. Détecteur regression_in_history
# ──────────────────────────────────────────────────────────────────────────


class TestDetector:
    def test_silent_without_data(self) -> None:
        assert detect_regression_in_history({}) == []
        assert detect_regression_in_history(
            {"longitudinal_trends": []},
        ) == []

    def test_silent_when_flat(self) -> None:
        data = {"longitudinal_trends": [{
            "engine_name": "tess", "n_runs": 5,
            "trend": {"slope": 1e-7, "r_squared": 0.0,
                      "intercept": 0.05, "n_runs": 5},
            "change_point": None,
            "first_cer": 0.05, "last_cer": 0.05,
            "absolute_delta": 0.0, "absolute_delta_pct": 0.0,
        }]}
        assert detect_regression_in_history(data) == []

    def test_emits_when_slope_high(self) -> None:
        # Slope > 1 pt CER / 365 jours
        data = {"longitudinal_trends": [{
            "engine_name": "tess", "n_runs": 5,
            "trend": {"slope": 0.0005, "r_squared": 0.9,
                      "intercept": 0.04, "n_runs": 5},
            "change_point": None,
            "first_cer": 0.04, "last_cer": 0.06,
            "absolute_delta": 0.02, "absolute_delta_pct": 2.0,
        }]}
        facts = detect_regression_in_history(data)
        assert len(facts) == 1
        assert facts[0].type == FactType.REGRESSION_IN_HISTORY
        assert facts[0].importance == FactImportance.MEDIUM
        assert facts[0].payload["pattern"] == "trend"

    def test_emits_high_when_delta_large(self) -> None:
        # |Δ| ≥ 5 pts → HIGH
        data = {"longitudinal_trends": [{
            "engine_name": "tess", "n_runs": 8,
            "trend": {"slope": 0.001, "r_squared": 0.95,
                      "intercept": 0.04, "n_runs": 8},
            "change_point": None,
            "first_cer": 0.04, "last_cer": 0.10,
            "absolute_delta": 0.06, "absolute_delta_pct": 6.0,
        }]}
        facts = detect_regression_in_history(data)
        assert facts[0].importance == FactImportance.HIGH

    def test_emits_on_change_point_only(self) -> None:
        # Slope nul mais rupture brutale
        data = {"longitudinal_trends": [{
            "engine_name": "tess", "n_runs": 8,
            "trend": {"slope": 1e-8, "r_squared": 0.0,
                      "intercept": 0.04, "n_runs": 8},
            "change_point": {
                "index": 4, "timestamp": "2025-03-01",
                "mean_before": 0.04, "mean_after": 0.07,
                "delta": 0.03, "n_before": 4, "n_after": 4,
            },
            "first_cer": 0.04, "last_cer": 0.07,
            "absolute_delta": 0.03, "absolute_delta_pct": 3.0,
        }]}
        facts = detect_regression_in_history(data)
        assert len(facts) == 1
        assert facts[0].payload["pattern"] == "change_point"
        assert "change_point_timestamp" in facts[0].payload

    def test_silent_when_lt_three_runs(self) -> None:
        data = {"longitudinal_trends": [{
            "engine_name": "tess", "n_runs": 2,
            "trend": {"slope": 0.001, "r_squared": 0.9,
                      "intercept": 0.04, "n_runs": 2},
            "change_point": None,
            "absolute_delta": 0.05,
        }]}
        assert detect_regression_in_history(data) == []


# ──────────────────────────────────────────────────────────────────────────
# 6. Anti-hallucination synthesis
# ──────────────────────────────────────────────────────────────────────────


def _payload_numbers(payload: dict) -> set[str]:
    out: set[str] = set()
    for v in payload.values():
        if isinstance(v, (int, float)):
            out.add(str(v))
            if isinstance(v, float) and v.is_integer():
                out.add(str(int(v)))
    return out


def _numbers_in(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


class TestAntiHallucination:
    def _build(self, lang: str) -> tuple[list[str], dict]:
        data = {
            "ranking": [{"engine": "tess", "mean_cer": 0.07}],
            "engines": [{"name": "tess", "mean_cer": 0.07}],
            "meta": {"document_count": 5},
            "longitudinal_trends": [{
                "engine_name": "tess", "n_runs": 8,
                "trend": {"slope": 0.0002, "r_squared": 0.91,
                          "intercept": 0.04, "n_runs": 8},
                "change_point": None,
                "first_cer": 0.04, "last_cer": 0.07,
                "absolute_delta": 0.03,
                "absolute_delta_pct": 3.0,
                "first_cer_pct": 4.0, "last_cer_pct": 7.0,
            }],
        }
        synthesis = build_synthesis(data, lang=lang, max_facts=10)
        facts = detect_regression_in_history(data)
        return synthesis["sentences"], facts[0].payload

    def _find(self, sentences: list[str], lang: str) -> str:
        marker = "modèles" if lang == "fr" else "models"
        for s in sentences:
            if marker in s:
                return s
        raise AssertionError(f"phrase introuvable : {sentences}")

    def test_fr_traceable(self) -> None:
        sentences, payload = self._build("fr")
        sentence = self._find(sentences, "fr")
        rendered = _numbers_in(sentence)
        allowed = _payload_numbers(payload)
        assert rendered.issubset(allowed), (
            f"non traçable : {rendered - allowed}"
        )

    def test_en_traceable(self) -> None:
        sentences, payload = self._build("en")
        sentence = self._find(sentences, "en")
        rendered = _numbers_in(sentence)
        allowed = _payload_numbers(payload)
        assert rendered.issubset(allowed), (
            f"non traçable : {rendered - allowed}"
        )


# ──────────────────────────────────────────────────────────────────────────
# 7. Vue HTML
# ──────────────────────────────────────────────────────────────────────────


class TestRender:
    def test_empty_returns_empty(self) -> None:
        assert build_longitudinal_html(None) == ""
        assert build_longitudinal_html([]) == ""

    def test_renders_table(self) -> None:
        trends = [{
            "engine_name": "tess", "n_runs": 8,
            "trend": {"slope": 0.0001, "r_squared": 0.85},
            "change_point": {
                "timestamp": "2025-03-01", "delta": 0.025,
            },
            "first_cer": 0.04, "last_cer": 0.07,
            "absolute_delta": 0.03, "absolute_delta_pct": 3.0,
        }]
        html = build_longitudinal_html(trends, _load_labels("fr"))
        assert "<table" in html
        assert "tess" in html
        # Δ +3.00
        assert "+3.00" in html
        # change-point
        assert "2025-03-01" in html

    def test_anti_injection(self) -> None:
        trends = [{
            "engine_name": "<script>alert(1)</script>",
            "n_runs": 5,
            "trend": {"slope": 0.001, "r_squared": 0.9},
            "change_point": None,
            "first_cer": 0.04, "last_cer": 0.05,
            "absolute_delta": 0.01, "absolute_delta_pct": 1.0,
        }]
        html = build_longitudinal_html(trends, _load_labels("fr"))
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html

    def test_renders_in_english(self) -> None:
        trends = [{
            "engine_name": "tess", "n_runs": 5,
            "trend": {"slope": 0.001, "r_squared": 0.9},
            "change_point": None,
            "first_cer": 0.04, "last_cer": 0.05,
            "absolute_delta": 0.01, "absolute_delta_pct": 1.0,
        }]
        html = build_longitudinal_html(trends, _load_labels("en"))
        assert "Evolution over time" in html


# ──────────────────────────────────────────────────────────────────────────
# 8. Complétude i18n
# ──────────────────────────────────────────────────────────────────────────


_KEYS = {
    "longitudinal_title", "longitudinal_note", "longitudinal_engine",
    "longitudinal_n_runs", "longitudinal_first", "longitudinal_last",
    "longitudinal_delta", "longitudinal_slope", "longitudinal_r2",
    "longitudinal_change",
}


class TestI18n:
    def test_fr(self) -> None:
        d = _load_labels("fr")
        assert not _KEYS - d.keys()

    def test_en(self) -> None:
        d = _load_labels("en")
        assert not _KEYS - d.keys()
