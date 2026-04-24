"""Moteur narratif factuel — génération de synthèse déterministe.

Le module extrait des faits saillants d'un ``BenchmarkResult`` et les rend en
phrases courtes via des templates externes. Aucun LLM n'est appelé : chaque
nombre ou nom apparaissant dans la synthèse est traçable à un champ du JSON de
résultats en entrée.

Sprint 1 : fondations — modèle ``Fact`` et registre de détecteurs.
Sprint 4 : intégration complète avec templates Jinja2 et rendu HTML.
"""

from picarones.core.narrative.facts import (
    Fact,
    FactType,
    FactImportance,
    DetectorRegistry,
    detect_all,
)

__all__ = [
    "Fact",
    "FactType",
    "FactImportance",
    "DetectorRegistry",
    "detect_all",
]
