# Étendre le moteur narratif

Ce guide explique comment ajouter un nouveau type de **fait détecté** à
la synthèse factuelle en tête du rapport.

## Architecture

```
picarones/domain/narrative/
├── __init__.py              # API publique + pipeline build_synthesis
├── facts.py                 # Modèle Fact, FactType, FactImportance, DetectorRegistry
├── detectors.py             # 12 détecteurs (un par FactType)
├── arbiter.py               # Tri par importance, non-redondance, anti-contradiction
├── renderer.py              # Rendu str.format_map sur templates YAML
└── templates/
    ├── fr.yaml              # Templates français (1 par FactType)
    └── en.yaml              # Templates anglais
```

## Ajouter un détecteur

> **Sprint 29** : un nouveau détecteur ne demande plus que **deux**
> fichiers à toucher (au lieu de quatre avant le sprint). Le décorateur
> `@register_detector` se charge de l'enregistrement, du tri par
> priorité, et de l'alimentation de `arbiter.DEFAULT_TYPE_ORDER`.

### 1. Déclarer le type de fait

Dans `facts.py`, ajoutez une valeur à `FactType` :

```python
class FactType(str, Enum):
    ...
    NEW_THING = "new_thing"
```

### 2. Implémenter et enregistrer le détecteur

Dans `detectors.py`, écrivez une fonction pure qui prend le dict
`benchmark_data` et retourne une liste de `Fact`, puis décorez-la avec
`@register_detector` :

```python
from picarones.domain.narrative.facts import Fact, FactImportance, FactType
from picarones.domain.narrative.registry import register_detector


@register_detector(
    FactType.NEW_THING,
    priority=55,                          # entre STRATUM_COLLAPSE (50) et ERROR_PROFILE_OUTLIER (60)
    importance=FactImportance.HIGH,
)
def detect_new_thing(benchmark_data: dict) -> list[Fact]:
    ...
```

Le décorateur :
- enregistre la fonction dans le registre central trié par `priority` ;
- alimente automatiquement `arbiter.DEFAULT_TYPE_ORDER` (plus besoin
  d'éditer `arbiter.py`) ;
- vérifie qu'aucun autre détecteur n'est déjà enregistré sur le même
  `FactType` (sinon `ValueError`) ;
- laisse la fonction utilisable telle quelle (pour les tests unitaires
  qui l'appellent directement).

### Conventions de priorité

Plus la valeur est petite, plus le fait remonte tôt en synthèse à
importance égale. Les détecteurs builtin utilisent un pas de **10**
pour laisser de la place :

| Priority | Type | Question éditoriale |
|---:|---|---|
| 10 | `GLOBAL_LEADER_CER`        | Qui gagne globalement ? |
| 20 | `STATISTICAL_TIE`          | Y a-t-il un ex-aequo ? |
| 30 | `SIGNIFICANT_GAP`          | À quel point l'écart est solide ? |
| 40 | `STRATUM_WINNER`           | Qui domine sur quel sous-corpus ? |
| 50 | `STRATUM_COLLAPSE`         | Qui s'effondre sur quoi ? |
| 60 | `ERROR_PROFILE_OUTLIER`    | Qui se trompe différemment ? |
| 70 | `LLM_HALLUCINATION_FLAG`   | Hallucinations VLM ? |
| 80 | `ROBUSTNESS_FRAGILE`       | Sensibilité aux dégradations ? |
| 90 | `PARETO_ALTERNATIVE`       | Y a-t-il un compromis coût/qualité ? |
| 100 | `SPEED_WINNER`            | Vitesse ? |
| 110 | `COST_OUTLIER`            | Coût aberrant ? |
| 120 | `CONFIDENCE_WARNING`      | Mise en garde sur la fiabilité. |

### Détails techniques

Le détecteur ne doit **jamais lever d'exception** — le
`DetectorRegistry` capte les erreurs en `logger.warning` mais c'est
une protection, pas une excuse.

```python
def detect_new_thing(benchmark_data: dict) -> list[Fact]:
    """Doc explicite : qu'est-ce qui déclenche ce fait ?"""
    # Exemple : flag les moteurs où une métrique X dépasse un seuil
    facts: list[Fact] = []
    for engine in benchmark_data.get("engines") or []:
        if (engine.get("some_metric") or 0) > 0.5:
            facts.append(Fact(
                type=FactType.NEW_THING,
                importance=FactImportance.HIGH,
                payload={
                    "engine": engine["name"],
                    "value": round(engine["some_metric"], 4),
                    "value_pct": round(engine["some_metric"] * 100, 1),
                },
                engines_involved=(engine["name"],),
            ))
    return facts
```

**Règle d'or anti-hallucination** : chaque champ que vous mettez dans
`payload` doit être **calculé à partir de** valeurs présentes dans
`benchmark_data`. Pas de constante ni de calcul invraisemblable.

### 3. Enregistrer dans la table

Toujours dans `detectors.py`, ajoutez au dict `DETECTORS_BY_TYPE` :

```python
DETECTORS_BY_TYPE = {
    ...
    FactType.NEW_THING: detect_new_thing,
}
```

`register_default_detectors(registry)` parcourt ce dict et l'enregistre
automatiquement. Aucune action supplémentaire requise.

### 4. Ajouter les templates FR/EN

Dans `templates/fr.yaml` et `templates/en.yaml`, ajoutez une entrée par
type, avec le nom de la valeur enum (ici `new_thing`) :

```yaml
new_thing: >-
  Le moteur {engine} dépasse le seuil de la métrique X
  ({value_pct} %).
```

Les placeholders `{engine}`, `{value_pct}` etc. doivent **exactement**
correspondre aux clés du `payload` du détecteur. Si vous oubliez un
champ, le rendu utilisera `?` (et logguera un warning) plutôt que de
crasher — mais les tests doivent attraper ça.

### 5. Ajuster l'arbitre si besoin

Dans `arbiter.py`, deux choses à considérer :

- **Ordre canonique** : ajoutez votre type dans `_TYPE_ORDER` à la
  position appropriée. Cet ordre départage les ex-aequo à importance
  égale et garantit le déterminisme.
- **Paires complémentaires** : par défaut, l'arbitre supprime les
  doublons sur le même moteur. Si votre nouveau type est complémentaire
  d'un autre type pour le même moteur (ex. leader + speed), ajoutez la
  paire dans `_COMPLEMENTARY_PAIRS`.
- **Règles anti-contradiction** : si votre fait peut contredire un autre
  (ex. Nemenyi vs Wilcoxon), implémentez la règle dans
  `_remove_contradictions`.

### 6. Tests

Ajoutez au minimum :

- Un test unitaire dans `tests/test_sprint19_narrative_engine.py` (ou
  un nouveau fichier) :

```python
class TestNewThingDetector:
    def test_emits_when_threshold_crossed(self):
        data = _minimal_data(engines=[
            {"name": "X", "some_metric": 0.7},
        ])
        facts = detect_new_thing(data)
        assert len(facts) == 1
        assert facts[0].payload["engine"] == "X"

    def test_empty_when_under_threshold(self):
        data = _minimal_data(engines=[
            {"name": "X", "some_metric": 0.3},
        ])
        assert detect_new_thing(data) == []
```

- Le test global de traçabilité
  (`test_every_number_in_synthesis_is_traceable`) couvrira automatiquement
  votre détecteur dès que vous l'ajoutez à la synthèse.

## Ajouter une langue

Pour ajouter une nouvelle langue (ex. allemand) :

1. Créez `templates/de.yaml` en copiant la structure de `fr.yaml` et en
   traduisant chaque entrée.
2. Ajoutez `de.json` dans `picarones/reports/html/i18n/` pour les libellés
   d'interface.
3. Ajoutez `de.yaml` dans `picarones/reports/html/glossary/` pour le glossaire.
4. Le code détecte automatiquement la langue via `load_glossary("de")`,
   `get_labels("de")`, et `_load_templates("de")` — aucun code à modifier.

## Tester votre changement

```bash
pytest tests/ -q --tb=short
picarones demo --output /tmp/demo.html --docs 8
# Ouvrir /tmp/demo.html et vérifier que la synthèse contient votre fait
```

Si la synthèse ne contient pas votre fait, vérifiez :
1. Que votre détecteur retourne bien quelque chose sur les données de
   démo (`grep -A 20 "def generate_sample_benchmark" picarones/evaluation/synthetic.py`).
2. Que l'importance est suffisante (> `MEDIUM`) pour passer le filtre
   par défaut de l'arbitre.
3. Que votre type n'est pas en collision avec un autre déjà retenu pour
   le même moteur (cf. `_is_redundant`).

---

## Politique éditoriale (Sprint 23)

L'arbitre départage les faits d'**égale importance** par un ordre canonique
des types : c'est un choix éditorial qui répond à la question *« quand A et
B sont aussi importants l'un que l'autre, lequel parle en premier ? »*.

L'ordre par défaut est défini dans `arbiter.py` sous le nom
`DEFAULT_TYPE_ORDER` :

```python
DEFAULT_TYPE_ORDER = (
    FactType.GLOBAL_LEADER_CER,      # 1. Qui gagne globalement
    FactType.STATISTICAL_TIE,        # 2. Y a-t-il un ex-aequo
    FactType.SIGNIFICANT_GAP,        # 3. À quel point l'écart est solide
    FactType.STRATUM_WINNER,         # 4. Qui domine sur quel sous-corpus
    FactType.STRATUM_COLLAPSE,       # 5. Qui s'effondre sur quoi
    FactType.ERROR_PROFILE_OUTLIER,  # 6. Qui se trompe différemment
    FactType.LLM_HALLUCINATION_FLAG, # 7. Hallucinations VLM
    FactType.ROBUSTNESS_FRAGILE,     # 8. Sensibilité aux dégradations
    FactType.PARETO_ALTERNATIVE,     # 9. Y a-t-il un compromis coût/qualité
    FactType.SPEED_WINNER,           # 10. Vitesse
    FactType.COST_OUTLIER,           # 11. Coût aberrant
    FactType.CONFIDENCE_WARNING,     # 12. Mise en garde sur la fiabilité
)
```

**Hypothèse implicite** : un lecteur d'institution patrimoniale veut
d'abord savoir *qui gagne* puis *à quel point cette victoire est solide*,
avant de découvrir des considérations de coût ou de vitesse. Une équipe
DevOps cherchant à industrialiser une chaîne aurait probablement l'ordre
inverse — vitesse et coût d'abord, qualité ensuite.

### Surcharger l'ordre sans patcher le code

Depuis le Sprint 23, `select_facts` accepte un argument optionnel
`type_order` :

```python
from picarones.domain.narrative import build_synthesis
from picarones.domain.narrative.arbiter import select_facts, DEFAULT_TYPE_ORDER
from picarones.domain.narrative.facts import FactType

# Réordonnancement : on remonte vitesse et coût avant qualité.
custom = (
    FactType.SPEED_WINNER,
    FactType.COST_OUTLIER,
    FactType.PARETO_ALTERNATIVE,
    FactType.GLOBAL_LEADER_CER,
    # ... compléter avec les autres types ; ceux qui manquent sont
    #     relégués à la fin sans crash.
)

facts = detect_all(benchmark_data)
selected = select_facts(facts, max_facts=5, type_order=custom)
```

Cas d'usage typiques :

- **Atelier MOOC** : promouvoir `STRATUM_COLLAPSE` et
  `ERROR_PROFILE_OUTLIER` en tête pour mettre l'accent sur la lecture
  diagnostique des erreurs.
- **Comité technique** : promouvoir `CONFIDENCE_WARNING` en tête pour
  forcer la discussion sur la fiabilité avant les classements.
- **Évaluation budgétaire** : promouvoir `COST_OUTLIER` et
  `PARETO_ALTERNATIVE` en tête.

### Règle anti-hallucination renforcée (Sprint 23)

Avant le Sprint 23, le test de traçabilité des nombres tolérait deux
littéraux non-traçables au payload (`95` pour le seuil de l'IC, `100`
comme tolérance numérique). Cette whitelist est désormais vide :

- Le seuil de confiance est propagé via `confidence_level` dans le
  payload des `Fact` de type `CONFIDENCE_WARNING`.
- L'unité du coût (`/1000 pages`) est propagée via `cost_unit_pages`
  dans `PARETO_ALTERNATIVE` et `COST_OUTLIER`.

**Si vous ajoutez un détecteur dont le template référence un nombre
constant** (ex. *« seuil α = 0,05 »*), vous devez **systématiquement**
le mettre dans le `payload`. Le test
`test_sprint19_narrative_engine.py::test_every_number_in_synthesis_is_traceable`
plus le test
`test_sprint23_anti_hallucination.py::TestTemplatesNoHardcodedLiterals`
échoueront sinon.
