"""Moteur narratif factuel — génération de synthèse déterministe.

Extrait des faits saillants d'un ``BenchmarkResult`` et les rend en phrases
courtes via des templates externes YAML. Aucun LLM : chaque nombre ou nom
apparaissant dans la synthèse est traçable au JSON de résultats en entrée.

API publique
------------
- ``Fact``, ``FactType``, ``FactImportance`` : modèle de données
- ``DetectorRegistry``                        : registre des détecteurs
- ``detect_all(data)``                        : applique le registre par défaut
- ``select_facts(facts, max_facts=5)``        : arbitre de sélection
- ``render_synthesis(facts, lang="fr")``      : rend en liste de phrases
- ``build_synthesis(data, lang="fr")``        : pipeline complet (Sprint 4)
"""

from picarones.measurements.narrative.facts import (
    Fact,
    FactType,
    FactImportance,
    DetectorRegistry,
    detect_all,
    _DEFAULT_REGISTRY,
)
from picarones.measurements.narrative.arbiter import select_facts
from picarones.measurements.narrative.renderer import (
    render_fact,
    render_synthesis,
    extract_numbers,
)
from picarones.measurements.narrative.detectors import (
    register_default_detectors,
    DETECTORS_BY_TYPE,
)


# Activer le registre par défaut — Sprint 4
register_default_detectors(_DEFAULT_REGISTRY)


def build_synthesis(
    benchmark_data: dict,
    lang: str = "fr",
    max_facts: int = 5,
) -> dict:
    """Pipeline complet : détection → arbitre → rendu.

    Returns
    -------
    dict avec :
      - ``sentences`` : liste de phrases prêtes à l'affichage
      - ``facts``     : liste de dicts ``Fact.as_dict()`` pour traçabilité
      - ``lang``      : langue utilisée
    """
    all_facts = detect_all(benchmark_data)
    selected = select_facts(all_facts, max_facts=max_facts)
    sentences = render_synthesis(selected, lang=lang)
    return {
        "sentences": sentences,
        "facts": [f.as_dict() for f in selected],
        "lang": lang,
    }


__all__ = [
    "Fact",
    "FactType",
    "FactImportance",
    "DetectorRegistry",
    "detect_all",
    "select_facts",
    "render_fact",
    "render_synthesis",
    "extract_numbers",
    "build_synthesis",
    "register_default_detectors",
    "DETECTORS_BY_TYPE",
]
