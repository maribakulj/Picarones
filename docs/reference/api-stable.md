# API publique stable de Picarones

> **Statut v2.0 (mai 2026)** : la migration vers l'architecture
> 8 couches canoniques est terminée.  Tous les paquets legacy
> top-level (`picarones.core`, `picarones.measurements`,
> `picarones.engines`, `picarones.modules`, `picarones.report`,
> `picarones.llm`, `picarones.pipelines`, `picarones.cli`,
> `picarones.web`, `picarones.extras`) ainsi que les sous-paquets
> transitoires (`adapters/legacy_engines/`, `adapters/legacy_pipelines/`,
> `interfaces/{cli,web}/_legacy/`) ont été **supprimés**.  Plus aucun
> shim, plus aucun `DeprecationWarning` rétrocompat.
>
> **Architecture canonique (cf.
> [`docs/explanation/architecture.md`](../explanation/architecture.md))** :
> `domain → formats → evaluation → pipeline → adapters → app →
> reports → interfaces`.
>
> **Chantier post-rewrite (mai 2026, branche
> `claude/fix-module-rewiring-MHssX`)** : réconciliation des
> contrats UI/API/runner après la migration.  Ruptures API
> visibles côté consommateur :
>
> - `CompetitorConfig` → `PipelineConfig`.
> - `PipelineConfig.ocr_engine` → `PipelineConfig.engine_name`
>   (le field accepte aussi `corpus` et des VLMs zero-shot — le
>   préfixe `ocr_` était trompeur).
> - `PipelineConfig.pipeline_mode` typé `Literal["text_only",
>   "text_and_image", "zero_shot"]` ; toute autre valeur (y compris
>   les anciens alias `post_correction_text` /
>   `post_correction_image`) est rejetée en 422.

## Définition

L'API publique stable de Picarones est constituée des classes,
fonctions, constantes et types listés ci-dessous, exportés depuis
l'arborescence canonique 8 couches.

Ce qui n'est pas dans cette liste peut évoluer à tout moment
sans bump majeur — utiliser ces points d'entrée pour une intégration
durable.

## Test automatique

Le test `tests/test_public_api.py` vérifie que tous les noms listés
ici existent et restent accessibles. Il échoue si un nom disparaît
ou change de forme.

## Liste exhaustive

### `picarones.evaluation.corpus`

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

### `picarones.evaluation.benchmark_result`

```python
class DocumentResult:    # résultat moteur sur un doc (CER, métriques, taxonomy…)
class EngineReport:      # agrégat moteur sur tout le corpus
class BenchmarkResult:   # résultat global multi-moteurs
```

### `picarones.evaluation.metrics.text_metrics`

```python
class MetricsResult:     # CER, WER, MER, WIL + variantes diplomatique/caseless
def compute_metrics(reference, hypothesis, char_exclude=None) -> MetricsResult
def aggregate_metrics(results: list) -> dict
```

### `picarones.app.services` — RunOrchestrator & helpers

```python
from picarones import RunOrchestrator, RunSpec, load_run_spec_from_yaml
from picarones.app.services import (
    prepare_preset_args,
    run_result_to_benchmark_result,
    PresetArgs,
    OrchestrationResult,
)

class RunOrchestrator:
    def __init__(self, output_dir: str | Path) -> None: ...
    def execute(
        self, spec: RunSpec, *,
        report_renderer=None, progress_callback=None, cancel_event=None,
    ) -> OrchestrationResult: ...
    def execute_preset(
        self, spec: RunSpec, *,
        corpus_spec, extracted_dir, pipeline_specs, adapter_resolver,
        adapter_kwargs=None, report_renderer=None,
        progress_callback=None, cancel_event=None,
    ) -> OrchestrationResult: ...

def prepare_preset_args(
    corpus, engines, *,
    workspace_dir: Path,
    views: tuple[str, ...] = ("text_final",),
    char_exclude=None, normalization_profile=None,
    partial_dir=None, entity_extractor=None,
    profile="standard", output_json=None,
    timeout_seconds_per_doc=60.0, code_version=None,
    output_dir=None,
) -> PresetArgs

def run_result_to_benchmark_result(
    run_result, *, corpus, engines,
    char_exclude=None, normalization_profile=None, profile="standard",
) -> BenchmarkResult
```

Phase B3-final (migration Option B, mai 2026) — entry point
canonique pour lancer un benchmark.  Deux modes :

- **YAML/déclaratif** : ``RunOrchestrator.execute(spec)`` avec un
  ``RunSpec`` chargé depuis YAML.  Cible long-terme pour la
  reproductibilité.
- **Python/instances** : ``prepare_preset_args(corpus, engines, ...)``
  → ``execute_preset(**args)`` → ``run_result_to_benchmark_result()``.
  Pour les callers qui instancient leurs adapters en Python.

Le module legacy ``benchmark_runner.py`` (entry point
``run_benchmark_via_service``) a été supprimé.  Voir
``docs/archive/2026-migration/option-b-user-guide.md`` pour le mapping.

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

### `picarones.evaluation.metrics.builtin_metrics`

Métriques scalaires natives, enregistrées dans le registre typé :

```python
def cer(reference, hypothesis) -> float
def wer(reference, hypothesis) -> float
def mer(reference, hypothesis) -> float
def wil(reference, hypothesis) -> float

# Stub démonstrateur
def text_preservation_after_reconstruction(reference_text, hypothesis_alto) -> float
```

### `picarones.evaluation.metrics.alto_metrics`

Métriques (ALTO, ALTO) + helper :

```python
def extract_text_from_alto(payload) -> str

def alto_text_cer(reference_alto, hypothesis_alto) -> float
def alto_text_wer(reference_alto, hypothesis_alto) -> float
def alto_text_mer(reference_alto, hypothesis_alto) -> float
def alto_text_wil(reference_alto, hypothesis_alto) -> float
```

### `picarones.interfaces.web.jobs`

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

- **Modules `picarones.evaluation.metrics/`** : peuvent évoluer librement.
  Quand ils changent, les shims rétrocompat dans `picarones.domain/`
  reflètent ces changements.
- **Modules `picarones.evaluation.metrics/`** : statut variable selon le
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

## Chemins canoniques par couche

L'arborescence v2.0 expose des points d'entrée stables organisés par
couche.  Toutes les intégrations doivent passer par ces chemins —
plus de path legacy disponible.

### Couche 3 — `picarones.evaluation`

```python
# Métriques (CER/WER + métriques avancées)
from picarones.evaluation.metrics.confusion import build_confusion_matrix
from picarones.evaluation.metrics.taxonomy import classify_errors
from picarones.evaluation.metrics.calibration import compute_calibration_metrics

# Moteur narratif (Cercle 7 → reports/, mais le contrat est en
# couche 3 pour rester accessible aux consommateurs externes)
from picarones.reports.narrative import build_synthesis
from picarones.domain.facts import Fact, FactType, FactImportance

# Modules philologiques (Sprints 55-60)
from picarones.evaluation.metrics.unicode_blocks import compute_unicode_block_accuracy
from picarones.evaluation.metrics.module_policy import ModuleManifest
```

### Couche 5 — `picarones.adapters`

```python
# OCR (factory canonique)
from picarones.adapters.ocr import ocr_adapter_from_name
from picarones.adapters.ocr import (
    TesseractAdapter, PeroOCRAdapter, KrakenAdapter, CalamariAdapter,
    MistralOCRAdapter, GoogleVisionAdapter, AzureDocIntelAdapter,
    PrecomputedTextAdapter,
)

# LLM
from picarones.adapters.llm.openai_adapter import OpenAIAdapter
from picarones.adapters.llm.anthropic_adapter import AnthropicAdapter
from picarones.adapters.llm.mistral_adapter import MistralAdapter
from picarones.adapters.llm.ollama_adapter import OllamaAdapter

# Importers de corpus distants
from picarones.adapters.corpus.iiif import IIIFImporter
from picarones.adapters.corpus.htr_united import HTRUnitedCatalogue
from picarones.adapters.corpus.huggingface import HuggingFaceImporter
```

### Couche 6 — `picarones.app.services`

```python
# Orchestration benchmark (Phase B3-final, migration Option B)
from picarones import RunOrchestrator, RunSpec, load_run_spec_from_yaml
from picarones.app.services import (
    prepare_preset_args, run_result_to_benchmark_result,
)
from picarones.app.services.corpus_service import CorpusService
from picarones.app.services.path_security import (
    WorkspaceManager,
    validated_path,
    safe_report_name,
    validated_prompt_filename,
)
from picarones.app.services.partial_store import (
    compute_run_fingerprint,
    partial_path_for_engine,
)
```

### Couche 7 — `picarones.reports.html`

```python
from picarones.reports.html.generator import ReportGenerator
```

### Couche 8 — `picarones.interfaces`

```python
# CLI : exposée comme entry point ``picarones`` (cf. pyproject.toml).
# Pas d'API Python stable — l'invocation est ``picarones run/diagnose/…``.

# Web : FastAPI app (intégration via ASGI).
from picarones.interfaces.web.app import app
from picarones.interfaces.web.models import (
    PipelineConfig, PipelineMode,
    BenchmarkRequest, BenchmarkRunRequest,
    NormalizationProfileId, TesseractLang, ReportLang,
)
```

## Voir aussi

- [`docs/explanation/architecture.md`](architecture.md) — cartographie
  des 3 cercles + critères d'assignation.
- [`docs/explanation/architecture.md`](architecture.md) — vue d'ensemble post-chantiers.
- [`tests/test_public_api.py`](../tests/test_public_api.py) — test
  automatique qui échoue si un nom listé ici disparaît.
