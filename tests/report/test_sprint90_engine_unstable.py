"""Tests Sprint 90 — A.II.4 finition : détecteur narratif
``engine_unstable`` + vue HTML stabilité multi-runs.

Couvre :

1. ``FactType.ENGINE_UNSTABLE`` enregistré + arbiter order.
2. ``detect_engine_unstable`` :
   - silence si pas de ``multirun_stability``
   - silence si CV faible et identical_rate haut
   - HIGH si CV > 10 %
   - HIGH si identical_rate < 50 %
3. Templates FR/EN : rendu factuel.
4. Anti-hallucination : chaque chiffre rendu provient du payload.
5. Vue HTML : adaptive, anti-injection, FR + EN.
6. Complétude i18n FR/EN.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from picarones.measurements.narrative import build_synthesis
from picarones.measurements.narrative.detectors import detect_engine_unstable
from picarones.core.facts import FactImportance, FactType
from picarones.report.multirun_stability_render import (
    build_multirun_stability_html,
)


def _load_labels(lang: str) -> dict:
    p = (
        Path(__file__).parent.parent.parent
        / "picarones" / "report" / "i18n" / f"{lang}.json"
    )
    return json.loads(p.read_text(encoding="utf-8"))


# ──────────────────────────────────────────────────────────────────────────
# 1. Modèle / registre
# ──────────────────────────────────────────────────────────────────────────


class TestFactType:
    def test_enum_value(self) -> None:
        assert FactType.ENGINE_UNSTABLE.value == "engine_unstable"

    def test_in_arbiter_fallback_order(self) -> None:
        from picarones.measurements.narrative.arbiter import _FALLBACK_TYPE_ORDER
        assert FactType.ENGINE_UNSTABLE in _FALLBACK_TYPE_ORDER


# ──────────────────────────────────────────────────────────────────────────
# 2. Détecteur
# ──────────────────────────────────────────────────────────────────────────


class TestDetector:
    def test_silent_without_data(self) -> None:
        assert detect_engine_unstable({}) == []
        assert detect_engine_unstable({"multirun_stability": []}) == []

    def test_silent_when_stable(self) -> None:
        # CV bas + tous identiques → pas de Fact
        data = {"multirun_stability": [{
            "engine_name": "tess", "n_runs": 3,
            "cer_mean": 0.04, "cer_stdev": 0.001, "cer_cv": 0.025,
            "identical_run_rate": 1.0, "n_distinct_outputs": 1,
        }]}
        assert detect_engine_unstable(data) == []

    def test_emits_when_cv_high(self) -> None:
        data = {"multirun_stability": [{
            "engine_name": "gpt-4o", "n_runs": 5,
            "cer_mean": 0.10, "cer_stdev": 0.025, "cer_cv": 0.25,
            "identical_run_rate": 0.10, "n_distinct_outputs": 5,
        }]}
        facts = detect_engine_unstable(data)
        assert len(facts) == 1
        assert facts[0].type == FactType.ENGINE_UNSTABLE
        assert facts[0].importance == FactImportance.HIGH
        assert facts[0].payload["engine"] == "gpt-4o"
        assert facts[0].payload["cer_cv_pct"] == 25.0

    def test_emits_when_identical_rate_low(self) -> None:
        # CV bas mais runs très différents → divergence détectée
        data = {"multirun_stability": [{
            "engine_name": "vlm", "n_runs": 4,
            "cer_mean": 0.05, "cer_stdev": 0.0025, "cer_cv": 0.05,
            "identical_run_rate": 0.20, "n_distinct_outputs": 4,
        }]}
        facts = detect_engine_unstable(data)
        assert len(facts) == 1
        assert facts[0].payload["identical_run_rate_pct"] == 20.0

    def test_silent_when_lt_two_runs(self) -> None:
        data = {"multirun_stability": [{
            "engine_name": "tess", "n_runs": 1,
            "cer_cv": 0.5, "identical_run_rate": 0.0,
        }]}
        assert detect_engine_unstable(data) == []

    def test_silent_when_engine_missing(self) -> None:
        data = {"multirun_stability": [{
            "n_runs": 3, "cer_cv": 0.30,
            "identical_run_rate": 0.0,
        }]}
        assert detect_engine_unstable(data) == []

    def test_multiple_engines(self) -> None:
        data = {"multirun_stability": [
            {"engine_name": "tess", "n_runs": 3,
             "cer_cv": 0.02, "identical_run_rate": 1.0},
            {"engine_name": "gpt-4o", "n_runs": 3,
             "cer_cv": 0.30, "identical_run_rate": 0.0},
        ]}
        facts = detect_engine_unstable(data)
        # Seul gpt-4o instable
        assert len(facts) == 1
        assert facts[0].payload["engine"] == "gpt-4o"


# ──────────────────────────────────────────────────────────────────────────
# 3. Anti-hallucination : tout chiffre rendu vient du payload
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
            "ranking": [{"engine": "gpt-4o", "mean_cer": 0.10}],
            "engines": [{"name": "gpt-4o", "mean_cer": 0.10}],
            "meta": {"document_count": 5},
            "multirun_stability": [{
                "engine_name": "gpt-4o", "n_runs": 4,
                "cer_mean": 0.103, "cer_stdev": 0.025,
                "cer_cv": 0.243, "identical_run_rate": 0.16,
                "n_distinct_outputs": 4,
            }],
        }
        synthesis = build_synthesis(data, lang=lang, max_facts=10)
        facts = detect_engine_unstable(data)
        return synthesis["sentences"], facts[0].payload

    def _find_unstable_sentence(
        self, sentences: list[str], lang: str,
    ) -> str:
        marker = "prudence" if lang == "fr" else "caution"
        for s in sentences:
            if marker in s:
                return s
        raise AssertionError(
            f"phrase ENGINE_UNSTABLE introuvable parmi {sentences}",
        )

    def test_fr_numbers_traceable(self) -> None:
        sentences, payload = self._build("fr")
        sentence = self._find_unstable_sentence(sentences, "fr")
        rendered = _numbers_in(sentence)
        allowed = _payload_numbers(payload)
        assert rendered.issubset(allowed), (
            f"non traçable : {rendered - allowed}"
        )

    def test_en_numbers_traceable(self) -> None:
        sentences, payload = self._build("en")
        sentence = self._find_unstable_sentence(sentences, "en")
        rendered = _numbers_in(sentence)
        allowed = _payload_numbers(payload)
        assert rendered.issubset(allowed), (
            f"non traçable : {rendered - allowed}"
        )


# ──────────────────────────────────────────────────────────────────────────
# 4. Vue HTML
# ──────────────────────────────────────────────────────────────────────────


class TestRender:
    def test_empty_returns_empty(self) -> None:
        assert build_multirun_stability_html(None) == ""
        assert build_multirun_stability_html([]) == ""

    def test_renders_table(self) -> None:
        stab = [{
            "engine_name": "gpt-4o", "n_runs": 5,
            "cer_mean": 0.10, "cer_stdev": 0.025, "cer_cv": 0.25,
            "identical_run_rate": 0.10, "n_distinct_outputs": 5,
        }]
        html = build_multirun_stability_html(stab, _load_labels("fr"))
        assert "<table" in html
        assert "gpt-4o" in html
        # CV formaté : 25.0
        assert "25.0" in html

    def test_anti_injection(self) -> None:
        stab = [{
            "engine_name": "<script>alert(1)</script>", "n_runs": 2,
            "cer_cv": 0.3, "identical_run_rate": 0.0,
        }]
        html = build_multirun_stability_html(stab, _load_labels("fr"))
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html

    def test_handles_missing_cv(self) -> None:
        # cer_cv None → cellule —, pas de crash
        stab = [{
            "engine_name": "tess", "n_runs": 2,
            "cer_mean": 0.0, "cer_stdev": 0.0, "cer_cv": None,
            "identical_run_rate": 1.0, "n_distinct_outputs": 1,
        }]
        html = build_multirun_stability_html(stab, _load_labels("fr"))
        assert "<table" in html
        assert "—" in html

    def test_renders_in_english(self) -> None:
        stab = [{
            "engine_name": "tess", "n_runs": 3,
            "cer_cv": 0.05, "identical_run_rate": 0.66,
            "n_distinct_outputs": 2,
        }]
        html = build_multirun_stability_html(stab, _load_labels("en"))
        assert "Multi-run stability" in html


# ──────────────────────────────────────────────────────────────────────────
# 5. Complétude i18n
# ──────────────────────────────────────────────────────────────────────────


_KEYS = {
    "stability_title", "stability_note", "stability_engine",
    "stability_n_runs", "stability_cer", "stability_cv",
    "stability_identical", "stability_distinct",
}


class TestI18n:
    def test_fr(self) -> None:
        d = _load_labels("fr")
        assert not _KEYS - d.keys()

    def test_en(self) -> None:
        d = _load_labels("en")
        assert not _KEYS - d.keys()
