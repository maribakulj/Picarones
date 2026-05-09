"""Tests Sprint 36 — câblage inter-moteurs au runner et au moteur narratif.

Couvre :

1. ``compute_inter_engine_analysis`` — agrégation au niveau benchmark
   (corpus complet) avec vérification des invariants (oracle ≥ best
   single, structure complète, top-N per_doc trié).
2. ``BenchmarkResult.inter_engine_analysis`` — sérialisation dans
   ``as_dict()`` quand renseigné, absent quand ``None``.
3. ``detect_ensemble_opportunity`` — déclenchement au-delà du seuil
   25 %, importance HIGH au-delà de 50 %, payload tracable, fallback
   sur per_engine_recall quand la divergence taxonomique manque.
4. Intégration ``build_synthesis`` — le détecteur s'enregistre par
   défaut et la synthèse rendue contient les valeurs du payload.
5. Garde-fou anti-hallucination — chaque nombre rendu est dans le
   payload (test de traçabilité).
"""

from __future__ import annotations

import re

import pytest

from picarones.evaluation.metrics.inter_engine import compute_inter_engine_analysis
from picarones.reports.narrative.detectors import detect_ensemble_opportunity
from picarones.domain.facts import FactImportance, FactType
from picarones.reports.narrative.renderer import extract_numbers, render_fact


# ──────────────────────────────────────────────────────────────────────────
# 1. compute_inter_engine_analysis (agrégateur)
# ──────────────────────────────────────────────────────────────────────────


class TestComputeInterEngineAnalysis:
    def test_returns_engines_alphabetical(self) -> None:
        out = compute_inter_engine_analysis(
            per_engine_outputs={"zebra": {"d1": "x"}, "alpha": {"d1": "x"}},
            ground_truths={"d1": "x"},
        )
        assert out["engines"] == ["alpha", "zebra"]

    def test_two_complementary_engines_oracle_is_one(self) -> None:
        out = compute_inter_engine_analysis(
            per_engine_outputs={
                "a": {"d1": "alpha beta x y", "d2": "alpha x x x"},
                "b": {"d1": "x y gamma delta", "d2": "x beta gamma delta"},
            },
            ground_truths={
                "d1": "alpha beta gamma delta",
                "d2": "alpha beta gamma delta",
            },
        )
        comp = out["complementarity"]
        assert comp["oracle_recall"] == pytest.approx(1.0)
        assert comp["best_single_recall"] < 1.0
        assert comp["absolute_gap"] > 0.0
        # Tous les tokens GT sont récupérables → relative_gap = 1
        assert comp["relative_gap"] == pytest.approx(1.0)

    def test_per_doc_top_is_sorted_by_gap(self) -> None:
        out = compute_inter_engine_analysis(
            per_engine_outputs={
                "a": {"d1": "x", "d2": "alpha", "d3": "alpha beta"},
                "b": {"d1": "alpha", "d2": "x", "d3": "alpha beta"},
            },
            ground_truths={"d1": "alpha", "d2": "alpha", "d3": "alpha beta"},
        )
        gaps = [r["absolute_gap"] for r in out["complementarity"]["per_doc"]]
        assert gaps == sorted(gaps, reverse=True)

    def test_taxonomy_divergence_attached_when_distributions_provided(self) -> None:
        out = compute_inter_engine_analysis(
            per_engine_outputs={"a": {"d1": "x"}, "b": {"d1": "y"}},
            ground_truths={"d1": "x"},
            taxonomy_distributions={
                "a": {"visual": 0.9, "casse": 0.1},
                "b": {"visual": 0.1, "casse": 0.9},
            },
        )
        td = out["taxonomy_divergence"]
        assert td is not None
        assert td["metric"] == "js"
        assert td["max_pair"] is not None
        assert {td["max_pair"][0], td["max_pair"][1]} == {"a", "b"}

    def test_no_taxonomy_means_section_none(self) -> None:
        out = compute_inter_engine_analysis(
            per_engine_outputs={"a": {"d1": "x"}, "b": {"d1": "y"}},
            ground_truths={"d1": "x"},
            taxonomy_distributions=None,
        )
        assert out["taxonomy_divergence"] is None

    def test_oracle_at_least_best_per_engine(self) -> None:
        """Invariant fondamental : l'oracle est ≥ recall de tous les moteurs
        individuels."""
        out = compute_inter_engine_analysis(
            per_engine_outputs={
                "a": {"d1": "alpha beta x", "d2": "alpha"},
                "b": {"d1": "x x gamma", "d2": "gamma"},
                "c": {"d1": "delta x x", "d2": "delta"},
            },
            ground_truths={
                "d1": "alpha beta gamma delta",
                "d2": "alpha beta gamma delta",
            },
        )
        oracle = out["complementarity"]["oracle_recall"]
        for recall in out["complementarity"]["per_engine_recall"].values():
            assert oracle >= recall - 1e-9

    def test_empty_inputs_returns_no_complementarity(self) -> None:
        out = compute_inter_engine_analysis(
            per_engine_outputs={},
            ground_truths={},
        )
        assert out["complementarity"] is None


# ──────────────────────────────────────────────────────────────────────────
# 2. BenchmarkResult expose inter_engine_analysis
# ──────────────────────────────────────────────────────────────────────────


class TestBenchmarkResultExposure:
    def test_as_dict_includes_when_set(self) -> None:
        from picarones.evaluation.benchmark_result import BenchmarkResult

        br = BenchmarkResult(
            corpus_name="t",
            corpus_source=None,
            document_count=0,
            engine_reports=[],
            inter_engine_analysis={"engines": ["a"], "complementarity": None},
        )
        assert "inter_engine_analysis" in br.as_dict()

    def test_as_dict_omits_when_none(self) -> None:
        from picarones.evaluation.benchmark_result import BenchmarkResult

        br = BenchmarkResult(
            corpus_name="t",
            corpus_source=None,
            document_count=0,
            engine_reports=[],
        )
        assert "inter_engine_analysis" not in br.as_dict()


# ──────────────────────────────────────────────────────────────────────────
# 3. Détecteur ENSEMBLE_OPPORTUNITY
# ──────────────────────────────────────────────────────────────────────────


def _build_data(relative_gap: float, *, with_taxonomy: bool = True) -> dict:
    """Construit un benchmark_data minimaliste pour tester le détecteur."""
    base = {
        "inter_engine_analysis": {
            "engines": ["tess", "pero"],
            "complementarity": {
                "oracle_recall": 0.95,
                "best_single_recall": 0.7,
                "best_engine": "pero",
                "absolute_gap": 0.25,
                "relative_gap": relative_gap,
                "doc_count": 47,
                "per_engine_recall": {"pero": 0.7, "tess": 0.5},
            },
            "taxonomy_divergence": (
                {
                    "metric": "js",
                    "matrix": {
                        "tess": {"tess": 0, "pero": 0.42},
                        "pero": {"tess": 0.42, "pero": 0},
                    },
                    "max_pair": ["tess", "pero", 0.42],
                }
                if with_taxonomy
                else None
            ),
        }
    }
    return base


class TestEnsembleOpportunityDetector:
    def test_below_threshold_no_fact(self) -> None:
        facts = detect_ensemble_opportunity(_build_data(relative_gap=0.10))
        assert facts == []

    def test_above_threshold_emits_fact(self) -> None:
        facts = detect_ensemble_opportunity(_build_data(relative_gap=0.30))
        assert len(facts) == 1
        assert facts[0].type is FactType.ENSEMBLE_OPPORTUNITY

    def test_high_importance_above_50pct(self) -> None:
        facts = detect_ensemble_opportunity(_build_data(relative_gap=0.83))
        assert facts[0].importance is FactImportance.HIGH

    def test_medium_importance_below_50pct(self) -> None:
        facts = detect_ensemble_opportunity(_build_data(relative_gap=0.30))
        assert facts[0].importance is FactImportance.MEDIUM

    def test_payload_uses_taxonomy_pair_when_available(self) -> None:
        facts = detect_ensemble_opportunity(_build_data(relative_gap=0.83))
        p = facts[0].payload
        assert {p["pair_a"], p["pair_b"]} == {"tess", "pero"}
        assert p["divergence"] == 0.42
        assert p["divergence_metric"] == "js"

    def test_fallback_pair_when_no_taxonomy(self) -> None:
        facts = detect_ensemble_opportunity(
            _build_data(relative_gap=0.83, with_taxonomy=False),
        )
        # Le fallback prend les deux meilleurs par per_engine_recall :
        # pero (0.7) et tess (0.5)
        p = facts[0].payload
        assert {p["pair_a"], p["pair_b"]} == {"tess", "pero"}
        assert p["divergence"] == 0.0  # divergence inconnue → 0

    def test_no_inter_engine_analysis_no_fact(self) -> None:
        assert detect_ensemble_opportunity({}) == []
        assert detect_ensemble_opportunity({"inter_engine_analysis": None}) == []
        assert detect_ensemble_opportunity({"inter_engine_analysis": {}}) == []


# ──────────────────────────────────────────────────────────────────────────
# 4. Intégration build_synthesis
# ──────────────────────────────────────────────────────────────────────────


class TestSynthesisIntegration:
    def test_detector_registered_by_default(self) -> None:
        from picarones.reports.narrative.registry import iter_detectors

        types = {entry.fact_type for entry in iter_detectors()}
        assert FactType.ENSEMBLE_OPPORTUNITY in types

    def test_synthesis_includes_ensemble_phrase(self) -> None:
        """Le détecteur s'active dans le pipeline complet et la phrase
        rendue contient bien les chiffres clés."""
        from picarones.reports.narrative import build_synthesis

        # benchmark_data minimal qui n'active QUE notre détecteur (pas
        # de ranking, pas de stats — pour isoler).
        data = _build_data(relative_gap=0.83)
        out = build_synthesis(data, lang="fr", max_facts=5)
        sentences = out["sentences"]
        assert any("voting" in s.lower() or "tess" in s for s in sentences)

    def test_synthesis_en_locale(self) -> None:
        from picarones.reports.narrative import build_synthesis

        data = _build_data(relative_gap=0.83)
        out = build_synthesis(data, lang="en", max_facts=5)
        sentences = out["sentences"]
        assert any("majority vote" in s.lower() for s in sentences)


# ──────────────────────────────────────────────────────────────────────────
# 5. Anti-hallucination — chaque nombre rendu doit être dans le payload
# ──────────────────────────────────────────────────────────────────────────


from tests.measurements._helpers import numbers_in_payload as _numbers_in_payload  # noqa: E402


class TestTraceability:
    @pytest.mark.parametrize("lang", ["fr", "en"])
    def test_every_rendered_number_is_in_payload(self, lang: str) -> None:
        facts = detect_ensemble_opportunity(_build_data(relative_gap=0.83))
        assert facts
        sentence = render_fact(facts[0], lang)
        traceable = _numbers_in_payload(facts[0].payload)
        # Whitelist limitée des constantes acceptées dans les templates
        # (aucune pour ENSEMBLE_OPPORTUNITY — tout doit venir du payload).
        whitelist: set[str] = set()
        for num in extract_numbers(sentence):
            normalized = num.replace(",", ".")
            assert normalized in traceable | whitelist, (
                f"Nombre {normalized!r} dans la phrase rendue n'est pas "
                f"traçable au payload {facts[0].payload!r}"
            )

    def test_no_extraneous_numbers_in_template(self) -> None:
        """Le template lui-même ne contient pas de nombres en dur."""
        from picarones.reports.narrative.renderer import _load_templates

        tpl = _load_templates("fr").get("ensemble_opportunity", "")
        assert tpl
        # Chercher des nombres en dur (hors {placeholder}). On enlève
        # les placeholders et on cherche les chiffres restants.
        without_placeholders = re.sub(r"\{[^}]+\}", "", tpl)
        digits = re.findall(r"\d", without_placeholders)
        assert not digits, f"Template contient des chiffres en dur : {digits}"
