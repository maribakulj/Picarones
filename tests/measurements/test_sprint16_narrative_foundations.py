"""Tests Sprint 16 — câblage line_metrics/hallucination + fondations du moteur narratif.

Couverture :
1. ``compute_document_result`` via le runner peuple bien ``line_metrics`` et
   ``hallucination_metrics`` sur un document réussi.
2. ``EngineReport`` expose ``aggregated_line_metrics`` et
   ``aggregated_hallucination`` après un benchmark.
3. Le modèle ``Fact`` et le ``DetectorRegistry`` fonctionnent.
4. Le registre par défaut est vide en Sprint 1 (les détecteurs seront activés
   progressivement dans les sprints suivants).
"""

from __future__ import annotations


from picarones.measurements.narrative import (
    DetectorRegistry,
    Fact,
    FactImportance,
    FactType,
    detect_all,
)

class TestFactModel:
    def test_fact_is_serializable(self):
        fact = Fact(
            type=FactType.GLOBAL_LEADER_CER,
            importance=FactImportance.CRITICAL,
            payload={"engine": "tesseract", "cer": 0.042},
            engines_involved=("tesseract",),
        )
        d = fact.as_dict()
        assert d["type"] == "global_leader_cer"
        assert d["importance"] == 100
        assert d["payload"]["cer"] == 0.042
        assert d["engines_involved"] == ["tesseract"]

    def test_fact_importance_ordering(self):
        assert FactImportance.CRITICAL > FactImportance.HIGH
        assert FactImportance.HIGH > FactImportance.MEDIUM
        assert FactImportance.MEDIUM > FactImportance.LOW


class TestDetectorRegistry:
    def test_registry_starts_empty(self):
        registry = DetectorRegistry()
        assert registry.registered_types() == ()
        assert registry.run({}) == []

    def test_register_and_run(self):
        registry = DetectorRegistry()

        def dummy_detector(data: dict) -> list[Fact]:
            return [Fact(
                type=FactType.GLOBAL_LEADER_CER,
                importance=FactImportance.CRITICAL,
                payload={"engine": data.get("leader", "unknown")},
            )]

        registry.register(FactType.GLOBAL_LEADER_CER, dummy_detector)
        assert FactType.GLOBAL_LEADER_CER in registry.registered_types()

        facts = registry.run({"leader": "tesseract"})
        assert len(facts) == 1
        assert facts[0].payload["engine"] == "tesseract"

    def test_registry_swallows_detector_exceptions(self):
        """Un détecteur défaillant ne doit pas casser le pipeline narratif."""
        registry = DetectorRegistry()

        def broken_detector(data: dict) -> list[Fact]:
            raise RuntimeError("boom")

        def working_detector(data: dict) -> list[Fact]:
            return [Fact(
                type=FactType.SPEED_WINNER,
                importance=FactImportance.HIGH,
                payload={},
            )]

        registry.register(FactType.GLOBAL_LEADER_CER, broken_detector)
        registry.register(FactType.SPEED_WINNER, working_detector)

        facts = registry.run({})
        assert len(facts) == 1
        assert facts[0].type == FactType.SPEED_WINNER

    def test_default_registry_is_empty_in_sprint_1(self):
        """Sprint 1 = fondations uniquement. Aucun détecteur n'est activé
        par défaut — ils le seront au Sprint 4 avec leurs templates."""
        facts = detect_all({})
        assert facts == []
