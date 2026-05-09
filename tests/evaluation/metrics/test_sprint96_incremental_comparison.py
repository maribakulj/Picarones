"""Tests Sprint 96 — B.5 : comparaison incrémentale.

Couvre :

1. ``compare_isolated_effect`` :
   - cas standard 4×2 → effet du LLM isolé
   - mean_rank correct
   - best/worst identifiés
   - higher_is_better inverse l'ordre
   - lt 2 runs → None
   - varying_slot inconnu → None
   - schémas de slots incompatibles ignorés
   - acceptation de dicts compatibles
2. Vue HTML :
   - adaptive
   - tri par rang moyen
   - marquage best ★ / worst ▼
   - anti-injection
3. Cas réaliste 5 OCR × 2 LLM.
4. Complétude i18n FR/EN.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from picarones.evaluation.metrics.incremental_comparison import (
    PipelineRun,
    compare_isolated_effect,
)
from picarones.reports.html.renderers.incremental_comparison import (
    build_incremental_comparison_html,
)


def _load_labels(lang: str) -> dict:
    p = (
        Path(__file__).parent.parent.parent.parent
        / "picarones" / "reports" / "i18n" / f"{lang}.json"
    )
    return json.loads(p.read_text(encoding="utf-8"))


# ──────────────────────────────────────────────────────────────────────────
# 1. compare_isolated_effect
# ──────────────────────────────────────────────────────────────────────────


class TestIsolatedEffect:
    def _runs_4x2(self) -> list[PipelineRun]:
        runs = []
        for ocr, q in [
            ("tess", 0.05), ("pero", 0.04),
            ("mistral", 0.03), ("vlm", 0.06),
        ]:
            for llm, boost in [("gpt", -0.01), ("claude", -0.005)]:
                runs.append(PipelineRun(
                    name=f"{ocr}+{llm}",
                    slots={"ocr": ocr, "llm": llm},
                    score=q + boost,
                ))
        return runs

    def test_basic_4x2(self) -> None:
        r = compare_isolated_effect(self._runs_4x2(), "llm")
        assert r is not None
        assert r["varying_slot"] == "llm"
        assert r["n_runs"] == 8
        assert r["n_groups"] == 4
        assert sorted(r["values"]) == ["claude", "gpt"]

    def test_mean_rank(self) -> None:
        # gpt domine systématiquement → rang moyen 1.0
        r = compare_isolated_effect(self._runs_4x2(), "llm")
        assert r["per_value"]["gpt"]["mean_rank"] == 1.0
        assert r["per_value"]["claude"]["mean_rank"] == 2.0

    def test_best_worst(self) -> None:
        r = compare_isolated_effect(self._runs_4x2(), "llm")
        assert r["best_value"] == "gpt"
        assert r["worst_value"] == "claude"

    def test_higher_is_better_inverts(self) -> None:
        # Score = F1 (haut = mieux)
        runs = [
            PipelineRun("a+x", {"a": "1", "b": "x"}, 0.95),
            PipelineRun("a+y", {"a": "1", "b": "y"}, 0.80),
            PipelineRun("c+x", {"a": "2", "b": "x"}, 0.92),
            PipelineRun("c+y", {"a": "2", "b": "y"}, 0.75),
        ]
        r = compare_isolated_effect(runs, "b", higher_is_better=True)
        assert r["best_value"] == "x"
        assert r["worst_value"] == "y"
        assert r["per_value"]["x"]["mean_rank"] == 1.0

    def test_lt_two_returns_none(self) -> None:
        assert compare_isolated_effect([], "x") is None
        assert compare_isolated_effect(
            [PipelineRun("a", {"x": "1"}, 0.1)], "x",
        ) is None

    def test_unknown_slot_returns_none(self) -> None:
        runs = [
            PipelineRun("a", {"ocr": "tess"}, 0.1),
            PipelineRun("b", {"ocr": "pero"}, 0.05),
        ]
        assert compare_isolated_effect(runs, "ghost") is None

    def test_incompatible_schemas_skipped(self) -> None:
        # 2 runs avec schéma {ocr, llm}, 1 run avec schéma {ocr}
        runs = [
            PipelineRun("a", {"ocr": "tess", "llm": "g"}, 0.04),
            PipelineRun("b", {"ocr": "pero", "llm": "g"}, 0.03),
            PipelineRun("c", {"ocr": "mistral"}, 0.02),
        ]
        r = compare_isolated_effect(runs, "ocr")
        # Le 3e run a un schéma incompatible (pas de "llm") → ignoré
        # quand on commence avec {ocr, llm}
        assert r is not None
        assert r["n_runs"] == 3  # tous les runs avec varying_slot
        # Mais seuls 2 sont dans des groupes
        assert sum(g["n_members"] for g in r["groups"]) == 2

    def test_accepts_dicts(self) -> None:
        runs = [
            {"name": "a", "slots": {"ocr": "tess", "llm": "g"}, "score": 0.05},
            {"name": "b", "slots": {"ocr": "tess", "llm": "c"}, "score": 0.04},
        ]
        r = compare_isolated_effect(runs, "llm")
        assert r is not None

    def test_ties_handled(self) -> None:
        # Scores identiques → rangs moyens
        runs = [
            PipelineRun("a", {"x": "1", "y": "p"}, 0.05),
            PipelineRun("b", {"x": "1", "y": "q"}, 0.05),  # ex aequo
        ]
        r = compare_isolated_effect(runs, "y")
        # Rangs : 1.5 et 1.5
        assert r["per_value"]["p"]["mean_rank"] == 1.5
        assert r["per_value"]["q"]["mean_rank"] == 1.5


# ──────────────────────────────────────────────────────────────────────────
# 2. Vue HTML
# ──────────────────────────────────────────────────────────────────────────


class TestRender:
    def test_empty_returns_empty(self) -> None:
        assert build_incremental_comparison_html(None) == ""
        assert build_incremental_comparison_html(
            {"per_value": {}},
        ) == ""

    def test_renders_table(self) -> None:
        runs = [
            PipelineRun("a", {"ocr": "tess", "llm": "g"}, 0.04),
            PipelineRun("b", {"ocr": "tess", "llm": "c"}, 0.05),
            PipelineRun("c", {"ocr": "pero", "llm": "g"}, 0.03),
            PipelineRun("d", {"ocr": "pero", "llm": "c"}, 0.04),
        ]
        analysis = compare_isolated_effect(runs, "llm")
        html = build_incremental_comparison_html(
            analysis, _load_labels("fr"),
        )
        assert "<table" in html
        assert "g" in html and "c" in html
        # Marqueur best ★
        assert "★" in html
        # Marqueur worst ▼
        assert "▼" in html

    def test_sorted_by_rank(self) -> None:
        runs = [
            PipelineRun("a", {"x": "1", "y": "good"}, 0.02),
            PipelineRun("b", {"x": "1", "y": "bad"}, 0.10),
        ]
        analysis = compare_isolated_effect(runs, "y")
        html = build_incremental_comparison_html(
            analysis, _load_labels("fr"),
        )
        # good apparaît avant bad
        assert html.index("good") < html.index("bad")

    def test_anti_injection_value(self) -> None:
        runs = [
            PipelineRun("a", {"x": "1", "y": "<script>alert(1)</script>"}, 0.04),
            PipelineRun("b", {"x": "1", "y": "ok"}, 0.05),
        ]
        analysis = compare_isolated_effect(runs, "y")
        html = build_incremental_comparison_html(
            analysis, _load_labels("fr"),
        )
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html

    def test_anti_injection_varying_slot(self) -> None:
        runs = [
            PipelineRun("a", {"x": "1", "<svg/>": "p"}, 0.04),
            PipelineRun("b", {"x": "1", "<svg/>": "q"}, 0.05),
        ]
        analysis = compare_isolated_effect(runs, "<svg/>")
        html = build_incremental_comparison_html(
            analysis, _load_labels("fr"),
        )
        assert "<svg/>" not in html
        assert "&lt;svg" in html

    def test_renders_in_english(self) -> None:
        runs = [
            PipelineRun("a", {"x": "1", "y": "p"}, 0.04),
            PipelineRun("b", {"x": "1", "y": "q"}, 0.05),
        ]
        analysis = compare_isolated_effect(runs, "y")
        html = build_incremental_comparison_html(
            analysis, _load_labels("en"),
        )
        assert "Controlled comparison" in html


# ──────────────────────────────────────────────────────────────────────────
# 3. Cas réaliste
# ──────────────────────────────────────────────────────────────────────────


class TestRealistic:
    def test_5_ocr_x_2_llm(self) -> None:
        # Produit cartésien complet
        ocr_quality = {
            "tess": 0.06, "pero": 0.05, "mistral": 0.03,
            "google": 0.04, "azure": 0.045,
        }
        runs = []
        for ocr, q in ocr_quality.items():
            for llm, boost in [("gpt-4o", -0.01), ("claude", -0.005)]:
                runs.append(PipelineRun(
                    name=f"{ocr}+{llm}",
                    slots={"ocr": ocr, "llm": llm},
                    score=q + boost,
                ))
        # Effet isolé du LLM
        r_llm = compare_isolated_effect(runs, "llm")
        assert r_llm["n_groups"] == 5
        assert r_llm["best_value"] == "gpt-4o"  # systématiquement meilleur
        # Effet isolé de l'OCR
        r_ocr = compare_isolated_effect(runs, "ocr")
        assert r_ocr["n_groups"] == 2
        assert r_ocr["best_value"] == "mistral"
        assert r_ocr["worst_value"] == "tess"


# ──────────────────────────────────────────────────────────────────────────
# 4. Complétude i18n
# ──────────────────────────────────────────────────────────────────────────


_KEYS = {
    "incr_title", "incr_note", "incr_slot_label", "incr_value",
    "incr_mean", "incr_stdev", "incr_rank", "incr_n_obs", "incr_groups",
}


class TestI18n:
    def test_fr(self) -> None:
        d = _load_labels("fr")
        assert not _KEYS - d.keys()

    def test_en(self) -> None:
        d = _load_labels("en")
        assert not _KEYS - d.keys()


# ──────────────────────────────────────────────────────────────────────────
# 5. PipelineRun
# ──────────────────────────────────────────────────────────────────────────


class TestPipelineRun:
    def test_as_dict(self) -> None:
        r = PipelineRun("a", {"x": "1"}, 0.05)
        d = r.as_dict()
        assert d["name"] == "a"
        assert d["slots"] == {"x": "1"}
        assert d["score"] == pytest.approx(0.05)

    def test_immutable(self) -> None:
        r = PipelineRun("a", {"x": "1"}, 0.05)
        with pytest.raises(Exception):
            r.score = 0.10  # type: ignore[misc]
