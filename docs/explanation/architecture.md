# Architecture Picarones — manifeste

> **Audience** : développeurs et mainteneurs.  Ce document explique
> *pourquoi* le code est organisé comme il l'est, pas seulement *où
> sont les fichiers*.  Pour la liste exhaustive des modules, lire
> directement le code — il est typé et documenté.

## Deux arborescences cohabitent par design

Le projet est en transition entre une arborescence **legacy** (héritée
de la fondation 2025) et une arborescence **post-rewrite** (refondation
ciblée S27-S46, 2026).  Cette cohabitation est explicite et finie dans
le temps :

| Arbo | Statut | Utilisation |
|------|--------|-------------|
| **Post-rewrite** | Canonique | **Tout nouveau code va ici.** |
| **Legacy** | Transitionnel | Reste exécutable le temps que les callers externes (HuggingFace Space, scripts BnF, notebooks de chercheurs) migrent. |

Le retrait du legacy est calendrier dans le CHANGELOG ; cf. aussi
`docs/migration/rewrite-status-s46.md`.

## Arbo canonique — 8 cercles concentriques

```
domain → formats → evaluation → pipeline → adapters → app → reports_v2 → interfaces
```

**Règle de dépendance stricte** : les flèches d'import vont uniquement
de l'extérieur vers l'intérieur.  Vérifié par
`tests/architecture/test_layer_dependencies.py`.  Aucun shim — un
module a un seul emplacement canonique.

### `picarones/domain/` — types purs

Couche 1 (la plus interne).  Aucune dépendance d'exécution,
aucun I/O, aucun framework.  Pydantic et stdlib uniquement.

| Module | Contenu |
|---|---|
| `artifacts.py` | `Artifact`, `ArtifactType` (10 types : IMAGE, RAW_TEXT, ALTO_XML, PAGE_XML, ENTITIES, READING_ORDER, ALIGNMENT, CORRECTED_TEXT, CANONICAL_DOCUMENT, CONFIDENCES) |
| `artifact_key.py` | `ArtifactKey` — clé canonique multi-paramètres pour la reprise par hash |
| `corpus.py` | `CorpusSpec`, métadonnées de corpus |
| `documents.py` | `DocumentRef`, `GroundTruthRef` |
| `evaluation_spec.py` | `MetricSpec`, `EvaluationView`, `EvaluationSpec` |
| `pipeline_spec.py` | `PipelineSpec`, `PipelineStep`, `INITIAL_STEP_ID` |
| `projection_spec.py` | `ProjectionSpec` (transformation candidate avant évaluation) |
| `provenance.py` | `ProvenanceRecord` |
| `run_manifest.py` | `RunManifest` — empreinte immuable d'un run, sérialisée en `run_manifest.json` |
| `errors.py` | Hiérarchie d'exceptions (`PicaronesError`, `AdapterStepError`, `ArtifactValidationError`, …) |

### `picarones/formats/` — parsers et sérialiseurs

Lecture/écriture des formats externes : ALTO XML, PAGE XML, texte
normalisé.  Dépend du domain ; aucune logique d'évaluation.

### `picarones/evaluation/` — moteurs d'évaluation

| Sous-package | Rôle |
|---|---|
| `metrics/` | Métriques (CER/WER, philologiques, calibration, NER, layout…). Enregistrées via `@register_metric` au registre typé |
| `projectors/` | Projections inter-types (ALTO → texte, canonical → texte) avec `ProjectionReport` |
| `views/` | Vues d'évaluation : `TextView`, `AltoView`, `SearchView`.  L'`EvaluationViewExecutor` aligne candidate + GT, applique normalisation + projection, calcule les métriques |
| `evaluation_engine.py` | Moteur central qui exécute une `EvaluationView` |
| `projection_engine.py` | Moteur de projection |
| `registry/` | `MetricRegistry` — découverte typée par signature `(input_type, output_type)` |

### `picarones/pipeline/` — DAG d'étapes

Orchestration mono-document d'une pipeline composée :

| Module | Rôle |
|---|---|
| `executor.py` | `PipelineExecutor` — exécute un `PipelineSpec` step par step, capture `StepResult`, filtre outputs sur `step.output_types` |
| `planner.py` | `PipelinePlanner` — résout les `inputs_from`, valide la spec, calcule les métriques aux jonctions |
| `validation.py` | Validation statique d'une `PipelineSpec` (types s'enchaînent, pas de cycle) |
| `runner.py` | `CorpusRunner` — orchestration corpus-wide avec ProcessPool/ThreadPool, backpressure, timeout, cancellation |
| `cache.py`, `cache_helpers.py`, `cache_protocol.py` | Reprise par hash via `ArtifactCachePort` |
| `yaml_io.py` | Sérialisation YAML déterministe d'une `PipelineSpec` |

### `picarones/adapters/` — implémentations concrètes

C'est ici que vivent les **dépendances externes** (pytesseract, pero,
mistralai, openai, anthropic, google-cloud-vision, …).

| Sous-package | Adapters |
|---|---|
| `ocr/` | TesseractAdapter, PeroOCRAdapter, MistralOCRAdapter, GoogleVisionAdapter, AzureDocIntelAdapter, PrecomputedTextAdapter |
| `llm/` | AnthropicLLMAdapter, OpenAILLMAdapter, MistralLLMAdapter, OllamaLLMAdapter |
| `vlm/` | AnthropicVLMAdapter, OpenAIVLMAdapter, MistralVLMAdapter, OllamaVLMAdapter (héritage multiple `BaseVLMAdapter + BaseLLMAdapter`, MRO guard) |
| `corpus/` | local folder, IIIF, Gallica, HTR-United, HuggingFace Datasets, eScriptorium |
| `storage/` | `InMemoryArtifactStore`, `FilesystemArtifactStore`, `JobStore` (SQLite) |
| `output_paths.py` | Helper partagé `resolve_output_path` (workspace-aware, read-only-mount-safe) |
| `_retry.py` | Helper partagé `call_with_retry` (3 retries, backoff 2/4/8s, sur 429+5xx+timeout réseau) |

**Règle** : un adapter peut importer le domain et ses libs externes.
Il ne doit **jamais** importer `app/` ou `interfaces/`.  Il n'a aucune
logique d'évaluation (un OCR adapter ne calcule pas le CER — il
produit un artefact texte que `evaluation/` consommera).

### `picarones/app/` — services applicatifs

Orchestration entre adapters et evaluation.

| Module | Rôle |
|---|---|
| `services/run_orchestrator.py` | `RunOrchestrator.execute(RunSpec)` — point d'entrée d'un run complet |
| `services/benchmark_service.py` | `BenchmarkService.run` — exécute pipelines × vues × corpus, produit `RunResult` |
| `services/job_runner.py` | `JobRunner` — soumission asynchrone (thread daemon) avec persistance `JobStore` |
| `services/corpus_service.py` | Loading + sandboxing + extraction ZIP avec zip-slip protection |
| `services/dependencies.py` | `capture_dependencies_lock()` via `importlib.metadata` pour le `RunManifest` |
| `services/path_security.py` | `WorkspaceManager` — sandboxe par session |
| `services/registry_service.py` | Découverte des adapters et vues canoniques |
| `schemas/run_spec.py` | `RunSpec`, `StepSpec` — modèles YAML user-facing |
| `results.py` | `RunResult`, `RunDocumentResult`, `ReportRenderer` (alias type unique) |

### `picarones/reports_v2/` — rendu déterministe

| Sous-package | Rôle |
|---|---|
| `csv/render.py` | `CsvReportRenderer` — un CSV plat (`run_id, doc, pipeline, view, metric, value, status`) |
| `json/render.py` | `JsonReportRenderer` — manifest + documents en JSON déterministe |
| `html/render.py` | `HtmlReportRenderer` — rapport autonome (TextView, AltoView, SearchView) |

Le rendu est strict : pas de JS dynamique, pas d'I/O, déterministe
bit-for-bit à entrée constante.  Permet à un relecteur 5 ans plus tard
de hasher un rapport et de le citer.

### `picarones/interfaces/` — points d'entrée user-facing

| Sous-package | Rôle |
|---|---|
| `cli/` | Click — `picarones-rewrite run`, `import_corpus`, `report` |
| `web/` | FastAPI — skeleton, routers (corpus, benchmark, jobs), middlewares de sécurité |

## Arbo legacy — `picarones/{cli,web,engines,llm,pipelines,report,measurements,extras,modules,core}/`

Reste exécutable.  Ne pas y ajouter de nouveau code.  Une partie est
re-exportée depuis l'arbo canonique via des shims dépréciés (cf.
`picarones/pipeline/spec.py`, alias `DEFAULT_*_PROMPT` singuliers
dans `BaseLLMAdapter`/`BaseVLMAdapter`) qui émettent
`DeprecationWarning` à l'usage.  Suppression effective prévue en 2.0.

## Principes architecturaux

### Pas de shim hors deprecation period

Un module a un seul emplacement canonique.  Quand un module migre,
on choisit explicitement entre :

- **Suppression dure** (pour la dette interne, pas de caller externe).
- **Shim avec `DeprecationWarning`** (pour la stabilité d'API publique).
  Le shim a une date de retrait inscrite dans le CHANGELOG.

### Pas d'`except Exception: pass`

Toute fonctionnalité optionnelle qui échoue émet un
`logger.warning("[module] feature dégradée : %s", exc)` avec contexte.
Vérifié par `tests/architecture/test_no_side_effect_imports.py`.

### Tests architecturaux comme garde-fous

Plusieurs tests verrouillent des invariants structurels que la revue
de code humaine raterait :

- `test_layer_dependencies.py` — circles strictement orientés
- `test_file_budgets.py` — pas de god-modules
- `test_doc_paths.py` — chemins cités dans la doc existent
- `test_output_paths_uniformity.py` — tous les adapters passent par `resolve_output_path`
- `test_storage_keys_filesystem_safe.py` — clés du store filesystem-safe (Windows)
- `test_manifest_reproducibility.py` — `RunManifest` capture tout pour rejouer
- `test_module_coverage.py` — chaque module a un test associé

### Reproductibilité bit-for-bit

Le `RunManifest` capture systématiquement : `code_version`,
`pipeline_specs` complets, `adapter_kwargs`, `dependencies_lock`
(via `importlib.metadata`), `view_specs`, timestamps.  La
sérialisation est déterministe (Pydantic ordered fields, JSON
sorted keys).  Le hash du manifest peut être cité dans une
publication scientifique.

## Évolution

L'évolution de l'architecture est documentée :

- Plans : [`docs/roadmap/evolution-2026.md`](../roadmap/evolution-2026.md)
- État du rewrite : [`docs/migration/rewrite-status-s46.md`](../migration/rewrite-status-s46.md)
- Audits institutionnels : [`docs/audits/`](../audits/)
- Politique d'API publique : [`docs/reference/api-stable.md`](../reference/api-stable.md)
