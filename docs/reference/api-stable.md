# API publique stable de Picarones

> **Statut** : ce document décrivait l'API publique du Cercle 1
> historique (`picarones.core/`).  Le projet est en cours de
> retrait du legacy vers une **architecture 8 couches**
> (`domain → formats → evaluation → pipeline → adapters → app
> → reports_v2 → interfaces`, cf.
> [`docs/explanation/architecture.md`](../explanation/architecture.md)).
>
> **Pendant la migration** (jusqu'à la version 2.0), l'API
> publique est en cours de refonte.  Tous les chemins legacy
> (`picarones.core.X`, `picarones.measurements.X`, etc.) sont
> des shims `DeprecationWarning` qui ré-exportent depuis le
> canonique.  Les nouveaux imports doivent utiliser les chemins
> canoniques (`picarones.domain.*`, `picarones.evaluation.*`).
>
> Le tableau de parité legacy ↔ canonique vit dans
> [`tests/architecture/test_legacy_canonical_parity.py`](../../tests/architecture/test_legacy_canonical_parity.py).

## Définition

L'API publique stable de Picarones est constituée des classes,
fonctions, constantes et types listés ci-dessous, désormais
exportés depuis l'arborescence canonique.

Ce qui n'est pas dans cette liste peut évoluer à tout moment
sans bump majeur.

Les imports historiques restent fonctionnels via shims pendant
la migration ; ils ne font **pas** partie de l'API publique
stable et émettent un `DeprecationWarning`.

## Test automatique

Le test `tests/test_public_api.py` vérifie que tous les noms listés
ici existent et restent accessibles. Il échoue si un nom disparaît
ou change de forme.

## Liste exhaustive

### `picarones.core.corpus`

```python
class GTLevel(str, Enum):
    TEXT, ALTO, PAGE, ENTITIES, READING_ORDER

class TextGT:           # GT texte plat
class AltoGT:           # GT ALTO XML
class PageGT:           # GT PAGE XML
class EntitiesGT:       # GT entités nommées (NER)
class ReadingOrderGT:   # GT ordre de lecture des régions
GTPayload = Union[...]  # type alias

class Document:         # un document du corpus (image + GT multi-niveaux)
class Corpus:           # collection de Documents

GT_SUFFIXES: dict[GTLevel, str]   # mapping niveau → suffixe fichier

def load_corpus_from_directory(path) -> Corpus
```

### `picarones.domain.artifacts`

```python
class ArtifactType(str, Enum):
    IMAGE, RAW_TEXT, CORRECTED_TEXT, ALTO_XML, PAGE_XML,
    CANONICAL_DOCUMENT, ENTITIES, READING_ORDER, ALIGNMENT, CONFIDENCES
    # Aliases legacy pour rétrocompat : TEXT, ALTO, PAGE
```

### `picarones.domain.module_protocol`

```python
class BaseModule(ABC):
    input_types: tuple[ArtifactType, ...]
    output_types: tuple[ArtifactType, ...]
    execution_mode: "io" | "cpu"

    @property name
    @abstractmethod process(inputs)
    metadata() -> dict
    validate_inputs(inputs)
    validate_outputs(outputs)

ExecutionMode = Literal["io", "cpu"]
```

### `picarones.core.results`

```python
class DocumentResult:    # résultat moteur sur un doc (CER, métriques, taxonomy…)
class EngineReport:      # agrégat moteur sur tout le corpus
class BenchmarkResult:   # résultat global multi-moteurs
```

### `picarones.measurements.metrics`

```python
class MetricsResult:     # CER, WER, MER, WIL + variantes diplomatique/caseless
def compute_metrics(reference, hypothesis, char_exclude=None) -> MetricsResult
def aggregate_metrics(results: list) -> dict
```

### `picarones.measurements.runner`

```python
def run_benchmark(
    corpus, engines,
    output_json=None,
    show_progress=True,
    progress_callback=None,
    char_exclude=None,
    max_workers=4,
    timeout_seconds=60.0,
    partial_dir=None,
    cancel_event=None,
    entity_extractor=None,
    profile="standard",
) -> BenchmarkResult
```

### `picarones.core.pipeline`

```python
class PipelineStep:
class PipelineSpec:
class StepResult:
class PipelineResult:
class PipelineRunner:
```

### `picarones.measurements.pipeline_benchmark`

```python
class StepAggregate:
class PipelineBenchmarkResult:

def default_initial_inputs(doc) -> dict
def run_pipeline_benchmark(spec, corpus, factory=...) -> PipelineBenchmarkResult
```

### `picarones.measurements.pipeline_comparison`

```python
class PipelineComparisonResult:

def compare_pipelines(specs, corpus, factories=None) -> PipelineComparisonResult
```

### `picarones.measurements.pipeline_spec_loader`

```python
class PipelineSpecLoadError(ValueError):

def load_pipeline_spec_from_yaml(path) -> PipelineSpec
def load_pipeline_spec_from_dict(data: dict) -> PipelineSpec
def load_comparison_specs_from_yaml(path) -> tuple[list[PipelineSpec], dict]
def load_comparison_specs_from_dict(data: dict) -> tuple[list[PipelineSpec], dict]
```

### `picarones.evaluation.metric_registry`

```python
class MetricSpec:    # frozen dataclass : name, func, input_types, ...

def register_metric(*, name, input_types, ...) -> Callable
def get_metric(name) -> MetricSpec
def all_metrics() -> list[MetricSpec]
def select_metrics(input_types) -> list[MetricSpec]
def compute_at_junction(reference, hypothesis, input_types, *, skip_on_error=True) -> dict
```

### `picarones.evaluation.metric_hooks`

```python
# Profils — constantes
PROFILE_MINIMAL = "minimal"
PROFILE_STANDARD = "standard"
PROFILE_PHILOLOGICAL = "philological"
PROFILE_DIAGNOSTICS = "diagnostics"
PROFILE_ECONOMICS = "economics"
PROFILE_PIPELINE = "pipeline"
PROFILE_FULL = "full"
KNOWN_PROFILES: frozenset[str]

# Modèles
class DocumentMetricHook:    # frozen dataclass
class CorpusMetricAggregator:

# API
def validate_profile(profile)
def register_document_metric(*, name, attribute, profiles, ...) -> Callable
def register_corpus_aggregator(*, name, attribute, profiles) -> Callable
def select_document_hooks(profile) -> list[DocumentMetricHook]
def select_corpus_aggregators(profile) -> list[CorpusMetricAggregator]
def run_document_hooks(profile, *, ground_truth, hypothesis, image_path, corpus_lang, ocr_result) -> dict
def run_corpus_aggregators(profile, document_results) -> dict
```

### `picarones.measurements.builtin_metrics`

Métriques scalaires natives, enregistrées dans le registre typé :

```python
def cer(reference, hypothesis) -> float
def wer(reference, hypothesis) -> float
def mer(reference, hypothesis) -> float
def wil(reference, hypothesis) -> float

# Stub démonstrateur
def text_preservation_after_reconstruction(reference_text, hypothesis_alto) -> float
```

### `picarones.measurements.alto_metrics`

Métriques (ALTO, ALTO) + helper :

```python
def extract_text_from_alto(payload) -> str

def alto_text_cer(reference_alto, hypothesis_alto) -> float
def alto_text_wer(reference_alto, hypothesis_alto) -> float
def alto_text_mer(reference_alto, hypothesis_alto) -> float
def alto_text_wil(reference_alto, hypothesis_alto) -> float
```

### `picarones.web.jobs`

Persistance des jobs benchmark (utilisé par l'interface web) :

```python
class JobStore:
def get_default_store() -> JobStore
def reset_default_store(...)
```

## Politique de stabilité

### Ce que nous garantissons

- **Existence** : aucun nom listé ne disparaît entre `1.x.0` et
  `1.y.0` (pour `y > x`).
- **Signatures** : aucun argument requis ajouté à une fonction
  publique. Les nouveaux arguments sont keyword avec valeur par
  défaut.
- **Types de retour** : compatibles entre versions mineures (un
  `dict` peut gagner des clés mais pas en perdre).
- **Sémantique** : un nom listé garde le même comportement
  fonctionnel. Les corrections de bug sont permises.

### Ce que nous ne garantissons pas

- **Modules `picarones.measurements/`** : peuvent évoluer librement.
  Quand ils changent, les shims rétrocompat dans `picarones.core/`
  reflètent ces changements.
- **Modules `picarones.extras/`** : statut variable selon le
  sous-package (academic / governance / historical / importers).
  Voir `docs/explanation/architecture.md`.
- **Comportement des renderers HTML** : la structure des fichiers HTML
  peut évoluer entre versions mineures. Nous gardons les noms des
  vues principales.
- **Internes des modules canoniques** : les noms commençant par `_`
  ne font pas partie de l'API publique. Les tests Sprints
  historiques qui les importent (Sprint 13/42) sont préservés mais
  par effort, pas par contrat.

### Bump majeur (`2.0.0`)

Un bump majeur sera nécessaire pour :

- Supprimer un nom de cette liste.
- Changer la signature d'une fonction publique de manière non
  rétrocompatible.
- Casser le format de sérialisation du `BenchmarkResult.to_json()`.
- Renommer un module de l'arborescence canonique.

## Modules historiques rétrocompat (non canoniques)

Les imports suivants continuent à fonctionner mais ne font pas partie
de l'API publique stable. Ils peuvent évoluer ou être retirés en
version mineure si une RFC le justifie.

```python
# Mesures (déplacées vers picarones.measurements/)
from picarones.measurements.confusion import build_confusion_matrix
from picarones.measurements.taxonomy import classify_errors
from picarones.measurements.calibration import compute_calibration_metrics
# ... ~40 modules métriques ...

# Moteur narratif (déplacé vers picarones.measurements.narrative/)
from picarones.measurements.narrative import build_synthesis
from picarones.domain.facts import Fact, FactType, FactImportance
from picarones.measurements.narrative.detectors import detect_global_leader_cer

# Plugins (déplacés vers picarones.extras/)
from picarones.core.taxonomy_intra_doc import compute_taxonomy_position_heatmap
from picarones.core.unicode_blocks import compute_unicode_block_accuracy
from picarones.core.module_policy import ModuleManifest
from picarones.importers.iiif import IIIFImporter
```

Pour les **nouvelles** intégrations, préférer les chemins canoniques :

- `picarones.measurements.X` pour les mesures.
- `picarones.measurements.narrative.X` pour le moteur narratif.
- `picarones.extras.historical.X` pour les modules philologiques.
- `picarones.extras.importers.X` pour les importers.
- `picarones.extras.academic.X` / `picarones.extras.governance.X` pour
  les plugins niche / gouvernance.

## Voir aussi

- [`docs/explanation/architecture.md`](architecture.md) — cartographie
  des 3 cercles + critères d'assignation.
- [`docs/explanation/architecture.md`](architecture.md) — vue d'ensemble post-chantiers.
- [`tests/test_public_api.py`](../tests/test_public_api.py) — test
  automatique qui échoue si un nom listé ici disparaît.
