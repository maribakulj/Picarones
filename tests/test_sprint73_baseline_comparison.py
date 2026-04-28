"""Tests Sprint 73 — A.I.3 : détecteur ``engine_off_baseline``.

Couvre :

1. ``compute_engine_baseline`` :
   - Cas standard : ≥ min_runs, écart > seuil → off_baseline=True
   - Écart faible → off_baseline=False
   - Moins de min_runs → ``None``
   - Baseline = 0 → ``relative_delta = None`` (et off si CER > 0)
   - ``current_run_id`` exclu de la baseline
   - Filtre par engine + corpus respecté
   - CER historiques None ignorés
2. ``compute_corpus_difficulty_percentile`` :
   - Calcul de percentile correct
   - ``harder_than_usual`` au-dessus de P75
   - ``easier_than_usual`` en-dessous de P25
   - Moins de min_runs → ``None``
3. Détecteur ``detect_engine_off_baseline`` :
   - Silencieux si pas de ``baseline_comparisons``
   - Émet 1 Fact par moteur off_baseline
   - Importance HIGH si |delta| ≥ 50 %, MEDIUM sinon
   - Payload contient les nombres exacts pour traçabilité
4. Rendu narratif : chaque nombre rendu est traçable au payload
   (anti-hallucination, FR + EN).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pytest

from picarones.core.baseline_comparison import (
    compute_corpus_difficulty_percentile,
    compute_engine_baseline,
)
from picarones.core.narrative.detectors import detect_engine_off_baseline
from picarones.core.narrative.facts import FactImportance, FactType
from picarones.core.narrative.renderer import render_fact


# ──────────────────────────────────────────────────────────────────────────
# Mock BenchmarkHistory
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class _Entry:
    run_id: str
    engine_name: str
    corpus_name: str
    cer_mean: Optional[float]
    metadata: dict = field(default_factory=dict)


class _MockHistory:
    def __init__(self, entries: list[_Entry]) -> None:
        self._entries = entries

    def query(
        self,
        engine: Optional[str] = None,
        corpus: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 100,
    ) -> list[Any]:
        out = []
        for e in self._entries:
            if engine and e.engine_name != engine:
                continue
            if corpus and e.corpus_name != corpus:
                continue
            out.append(e)
        return out[:limit]


# ──────────────────────────────────────────────────────────────────────────
# 1. compute_engine_baseline
# ──────────────────────────────────────────────────────────────────────────


class TestEngineBaseline:
    def test_off_baseline_higher(self) -> None:
        # 10 runs historiques à 4 % CER, run courant à 5,2 % → +30 %
        history = _MockHistory([
            _Entry(f"r{i}", "tess", "corpus_A", 0.04)
            for i in range(10)
        ])
        result = compute_engine_baseline(
            history, "tess", "corpus_A", current_cer=0.052,
        )
        assert result is not None
        assert result["n_runs"] == 10
        assert result["cer_current"] == 0.052
        assert result["cer_historical_mean"] == pytest.approx(0.04)
        assert result["absolute_delta"] == pytest.approx(0.012)
        assert result["relative_delta"] == pytest.approx(0.30)
        assert result["off_baseline"] is True

    def test_within_baseline(self) -> None:
        history = _MockHistory([
            _Entry(f"r{i}", "tess", "c", 0.04)
            for i in range(10)
        ])
        # Run courant à 4,1 % → écart 2,5 %, sous le seuil 20 %
        result = compute_engine_baseline(
            history, "tess", "c", current_cer=0.041,
        )
        assert result is not None
        assert result["off_baseline"] is False

    def test_min_runs_filter(self) -> None:
        # Seulement 4 runs → sous le min_runs=5
        history = _MockHistory([
            _Entry(f"r{i}", "tess", "c", 0.04) for i in range(4)
        ])
        assert compute_engine_baseline(
            history, "tess", "c", current_cer=0.05,
        ) is None

    def test_custom_min_runs(self) -> None:
        history = _MockHistory([
            _Entry(f"r{i}", "tess", "c", 0.04) for i in range(3)
        ])
        # min_runs=2 → assez
        result = compute_engine_baseline(
            history, "tess", "c", current_cer=0.05, min_runs=2,
        )
        assert result is not None
        assert result["n_runs"] == 3

    def test_current_run_excluded(self) -> None:
        history = _MockHistory([
            _Entry("current", "tess", "c", 0.20),  # run courant déjà loggé
            *[_Entry(f"r{i}", "tess", "c", 0.04) for i in range(5)],
        ])
        result = compute_engine_baseline(
            history, "tess", "c", current_cer=0.05,
            current_run_id="current",
        )
        assert result is not None
        # Le 0,20 ne doit pas tirer la moyenne historique
        assert result["n_runs"] == 5
        assert result["cer_historical_mean"] == pytest.approx(0.04)

    def test_filter_by_engine_and_corpus(self) -> None:
        history = _MockHistory([
            *[_Entry(f"r{i}", "tess", "corpus_A", 0.04) for i in range(5)],
            # Mêmes runs sur autre corpus — ne doivent pas compter
            *[_Entry(f"o{i}", "tess", "corpus_B", 0.20) for i in range(5)],
            # Autre moteur, même corpus — ne doivent pas compter
            *[_Entry(f"p{i}", "pero", "corpus_A", 0.99) for i in range(5)],
        ])
        result = compute_engine_baseline(
            history, "tess", "corpus_A", current_cer=0.05,
        )
        assert result is not None
        assert result["n_runs"] == 5
        assert result["cer_historical_mean"] == pytest.approx(0.04)

    def test_cer_none_ignored(self) -> None:
        history = _MockHistory([
            _Entry("r1", "tess", "c", None),
            _Entry("r2", "tess", "c", -0.5),  # négatif → ignoré
            *[_Entry(f"r{i}", "tess", "c", 0.04) for i in range(3, 8)],
        ])
        result = compute_engine_baseline(
            history, "tess", "c", current_cer=0.05,
        )
        assert result is not None
        assert result["n_runs"] == 5

    def test_baseline_zero_returns_none_relative(self) -> None:
        history = _MockHistory([
            _Entry(f"r{i}", "tess", "c", 0.0) for i in range(5)
        ])
        result = compute_engine_baseline(
            history, "tess", "c", current_cer=0.05,
        )
        assert result is not None
        assert result["relative_delta"] is None
        assert result["off_baseline"] is False  # not calculable

    def test_invalid_current_cer(self) -> None:
        history = _MockHistory([
            _Entry(f"r{i}", "tess", "c", 0.04) for i in range(5)
        ])
        assert compute_engine_baseline(
            history, "tess", "c", current_cer=None,  # type: ignore
        ) is None
        assert compute_engine_baseline(
            history, "tess", "c", current_cer=-0.1,
        ) is None


# ──────────────────────────────────────────────────────────────────────────
# 2. compute_corpus_difficulty_percentile
# ──────────────────────────────────────────────────────────────────────────


class TestCorpusDifficultyPercentile:
    def test_percentile_calculation(self) -> None:
        history = _MockHistory([
            _Entry(f"r{i}", "x", "c", 0.04, metadata={"difficulty": d})
            for i, d in enumerate([0.1, 0.2, 0.3, 0.4, 0.5])
        ])
        result = compute_corpus_difficulty_percentile(history, 0.45)
        assert result is not None
        # 4 sur 5 valeurs ≤ 0.45 → P80
        assert result["percentile"] == pytest.approx(80.0)
        assert result["n_runs"] == 5

    def test_harder_than_usual(self) -> None:
        history = _MockHistory([
            _Entry(f"r{i}", "x", "c", 0.04, metadata={"difficulty": 0.1 * i})
            for i in range(1, 11)  # 10 valeurs : 0.1 .. 1.0
        ])
        # 0.95 → percentile 90 → harder
        result = compute_corpus_difficulty_percentile(history, 0.95)
        assert result is not None
        assert result["harder_than_usual"] is True
        assert result["easier_than_usual"] is False

    def test_easier_than_usual(self) -> None:
        history = _MockHistory([
            _Entry(f"r{i}", "x", "c", 0.04, metadata={"difficulty": 0.1 * i})
            for i in range(1, 11)
        ])
        result = compute_corpus_difficulty_percentile(history, 0.05)
        assert result is not None
        assert result["easier_than_usual"] is True
        assert result["harder_than_usual"] is False

    def test_min_runs_filter(self) -> None:
        history = _MockHistory([
            _Entry("r1", "x", "c", 0.04, metadata={"difficulty": 0.5}),
        ])
        assert compute_corpus_difficulty_percentile(history, 0.5) is None


# ──────────────────────────────────────────────────────────────────────────
# 3. Détecteur narratif
# ──────────────────────────────────────────────────────────────────────────


class TestDetector:
    def test_silent_without_baseline_data(self) -> None:
        assert detect_engine_off_baseline({}) == []
        assert detect_engine_off_baseline(
            {"baseline_comparisons": []},
        ) == []

    def test_silent_when_off_baseline_false(self) -> None:
        facts = detect_engine_off_baseline({
            "baseline_comparisons": [
                {
                    "engine_name": "t", "cer_current": 0.04,
                    "cer_historical_mean": 0.04, "n_runs": 10,
                    "relative_delta": 0.0, "off_baseline": False,
                },
            ],
        })
        assert facts == []

    def test_silent_when_relative_delta_none(self) -> None:
        # Baseline = 0 → relative None → on s'abstient
        facts = detect_engine_off_baseline({
            "baseline_comparisons": [
                {
                    "engine_name": "t", "cer_current": 0.05,
                    "cer_historical_mean": 0.0, "n_runs": 10,
                    "relative_delta": None, "off_baseline": True,
                },
            ],
        })
        assert facts == []

    def test_emits_fact_for_off_baseline(self) -> None:
        facts = detect_engine_off_baseline({
            "baseline_comparisons": [
                {
                    "engine_name": "tess", "cer_current": 0.052,
                    "cer_historical_mean": 0.041, "n_runs": 12,
                    "relative_delta": 0.268, "off_baseline": True,
                },
            ],
        })
        assert len(facts) == 1
        f = facts[0]
        assert f.type == FactType.ENGINE_OFF_BASELINE
        assert f.importance == FactImportance.MEDIUM
        assert f.payload["engine"] == "tess"
        assert f.payload["cer_current_pct"] == 5.2
        assert f.payload["cer_historical_mean_pct"] == 4.1
        assert f.payload["n_runs"] == 12
        assert f.payload["relative_delta_pct"] == 26.8
        assert f.payload["direction"] == "higher"
        assert f.engines_involved == ("tess",)

    def test_high_importance_above_50pct(self) -> None:
        facts = detect_engine_off_baseline({
            "baseline_comparisons": [
                {
                    "engine_name": "x", "cer_current": 0.08,
                    "cer_historical_mean": 0.04, "n_runs": 10,
                    "relative_delta": 1.0, "off_baseline": True,
                },
            ],
        })
        assert facts[0].importance == FactImportance.HIGH

    def test_multiple_engines(self) -> None:
        facts = detect_engine_off_baseline({
            "baseline_comparisons": [
                {
                    "engine_name": "tess", "cer_current": 0.05,
                    "cer_historical_mean": 0.04, "n_runs": 10,
                    "relative_delta": 0.25, "off_baseline": True,
                },
                {
                    "engine_name": "pero", "cer_current": 0.03,
                    "cer_historical_mean": 0.04, "n_runs": 10,
                    "relative_delta": -0.25, "off_baseline": True,
                },
            ],
        })
        assert len(facts) == 2
        assert facts[1].payload["direction"] == "lower"


# ──────────────────────────────────────────────────────────────────────────
# 4. Traçabilité anti-hallucination
# ──────────────────────────────────────────────────────────────────────────


class TestTraceability:
    @pytest.mark.parametrize("lang", ["fr", "en"])
    def test_each_number_in_rendered_text_is_in_payload(
        self, lang: str,
    ) -> None:
        import re
        facts = detect_engine_off_baseline({
            "baseline_comparisons": [
                {
                    "engine_name": "tess", "cer_current": 0.052,
                    "cer_historical_mean": 0.041, "n_runs": 12,
                    "relative_delta": 0.268, "off_baseline": True,
                },
            ],
        })
        text = render_fact(facts[0], lang=lang)
        assert text  # non vide
        # Chaque nombre dans le texte doit venir du payload (ou d'une
        # constante de template — ici aucune)
        payload_nums = {
            "5.2", "4.1", "12", "26.8",
        }
        rendered_nums = set(re.findall(r"\d+\.?\d*", text))
        for num in rendered_nums:
            assert num in payload_nums, (
                f"nombre rendu {num!r} non traçable au payload"
            )
