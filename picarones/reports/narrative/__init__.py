"""Moteur narratif factuel — génération de synthèse déterministe.

Extrait des faits saillants d'un ``BenchmarkResult`` et les rend en phrases
courtes via des templates externes YAML.

Garantie d'intégrité (audit scientifique F7) — formulation précise
----------------------------------------------------------------
**Aucun LLM, aucune valeur aléatoire ou fabriquée.**  Le rendu est un
``str.format_map`` déterministe (même entrée → mêmes phrases).
Concernant les valeurs affichées :

- les **noms d'entités** (moteurs, strates, documents) sont repris
  *verbatim* du JSON de résultats en entrée ;
- les **nombres** sont soit repris verbatim du JSON d'entrée, soit une
  **fonction déterministe et auditable** de valeurs d'entrée calculée
  dans la couche rapport (p. ex. écart relatif médiane↔moyenne,
  accélération = durée_médiane/durée, largeur d'IC).  Statistiques
  dérivées traçables, jamais inventées.

L'ancienne formulation « chaque nombre provient du JSON d'entrée »
était trop forte (elle passait sous silence les dérivations) et le
test associé était circulaire (il validait les nombres rendus contre
le *payload* du Fact, lui-même rempli par le détecteur).  La
traçabilité est désormais testée vis-à-vis de la **source**.

API publique
------------
- ``Fact``, ``FactType``, ``FactImportance`` : modèle de données
- ``DetectorRegistry``                        : registre des détecteurs
- ``detect_all(data)``                        : applique le registre par défaut
- ``select_facts(facts, max_facts=5)``        : arbitre de sélection
- ``render_synthesis(facts, lang="fr")``      : rend en liste de phrases
``build_synthesis(data, lang="fr")``        : pipeline complet
"""

from picarones.domain.facts import (
    Fact,
    FactType,
    FactImportance,
    DetectorRegistry,
    detect_all,
    _DEFAULT_REGISTRY,
)
from picarones.reports.narrative.arbiter import select_facts
from picarones.reports.narrative.renderer import (
    render_fact,
    render_synthesis,
    extract_numbers,
)
from picarones.reports.narrative.detectors import (
    register_default_detectors,
    DETECTORS_BY_TYPE,
)


# Activer le registre par défaut
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
