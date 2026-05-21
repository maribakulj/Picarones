# Architecture Picarones — manifeste

> **Audience** : développeurs et mainteneurs.  Ce document explique
> *pourquoi* le code est organisé comme il l'est, pas seulement *où
> sont les fichiers*.  Pour la liste exhaustive des modules, lire
> directement le code — il est typé et documenté.

## Statut v2.0 — une seule arborescence canonique

À v2.0 (mai 2026), Picarones a **une seule arborescence**.  Tous
les paquets pré-rewrite ainsi que leurs sous-paquets transitoires
ont été supprimés au cours des sprints A-H.  Pour le détail
historique, voir le [CHANGELOG section 2.0.0](../../CHANGELOG.md)
et [`docs/archive/2026-migration/`](../archive/2026-migration/).

Toute documentation, tout commentaire qui mentionne « deux
arborescences » ou « legacy en cours de retrait » est obsolète.
La seule cohabitation acceptable à v2.0+ est celle entre
**modules canoniques** : par exemple `evaluation/metric_registry`
(module-level, side-effect d'import) et `evaluation/registry/registry`
(instance-based) — deux patterns volontairement coexistants pour
deux usages distincts (auto-discovery vs DI explicite).

Le test `tests/architecture/test_no_legacy_imports_in_rewrite.py`
verrouille cet invariant via `LEGACY_PACKAGES = ()`.

## Architecture — 8 couches concentriques

```
domain → formats → evaluation → pipeline → adapters → app → reports → interfaces
```

**Règle de dépendance stricte** : les flèches d'import vont uniquement
de l'extérieur vers l'intérieur (couche N peut importer 1..N-1, pas
N+1..8).  Vérifié par
`tests/architecture/test_layer_dependencies.py`.

| # | Couche | Rôle |
|---|---|---|
| 1 | `domain/` | Types purs (Pydantic + stdlib) |
| 2 | `formats/` | Parsers ALTO, PAGE XML, normalisation texte |
| 3 | `evaluation/` | Métriques, statistiques, vues d'évaluation |
| 4 | `pipeline/` | DAG d'étapes, cache, runner corpus-wide |
| 5 | `adapters/` | OCR, LLM, VLM, corpus importers, storage |
| 6 | `app/` | Services applicatifs (orchestration) |
| 7 | `reports/` | Rendu HTML / JSON / CSV |
| 8 | `interfaces/` | CLI Click + Web FastAPI |

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

Le parser XML interne (`_xml_utils.safe_parse_xml`) délègue à
`defusedxml` avec `forbid_dtd=True`, bloquant XXE, Billion Laughs
et déclarations `<!DOCTYPE>`.  Les défenses sont verrouillées par
`tests/security/test_xxe_attack.py` (Sprint S1.4).

### `picarones/evaluation/` — moteurs d'évaluation

| Sous-package | Rôle |
|---|---|
| `metrics/` | ~37 métriques (CER/WER, philologiques, calibration, NER, layout…). Enregistrées via `@register_metric` au registre typé |
| `projectors/` | Projections inter-types (ALTO → texte, canonical → texte) avec `ProjectionReport` |
| `views/` | Vues d'évaluation : `TextView`, `AltoView`, `SearchView`.  L'`EvaluationViewExecutor` aligne candidate + GT, applique normalisation + projection, calcule les métriques |
| `evaluation_engine.py` | Moteur central qui exécute une `EvaluationView` |
| `projection_engine.py` | Moteur de projection |
| `registry/` | `MetricRegistry` — découverte typée par signature `(input_type, output_type)` |
| `statistics/` | Wilcoxon, Friedman/Nemenyi, bootstrap, Pareto, CDD |
| `synthetic.py` | `generate_sample_benchmark` (utilisé par `picarones demo`) |

**Whitelist d'imports externes** : `PIL, annotated_types, jiwer,
numpy, pydantic, rapidfuzz, scipy, spacy, typing_extensions,
yaml`.  **Pas** `pytesseract, mistralai, azure, google,
pero_ocr` — ceux-là vivent en couche 5 (`adapters/`).

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
| `llm_pipeline_builder.py` | `make_ocr_llm_pipeline_spec` (3 modes : text_only, text_and_image, zero_shot) |
| `llm_pipeline_config.py` | `OCRLLMPipelineConfig` (container OCR+LLM) |

### `picarones/adapters/` — implémentations concrètes

C'est ici que vivent les **dépendances externes** (pytesseract,
pero, mistralai, openai, anthropic, google-cloud-vision, …).

| Sous-package | Adapters |
|---|---|
| `ocr/` | TesseractAdapter, PeroOCRAdapter, MistralOCRAdapter, GoogleVisionAdapter, AzureDocIntelAdapter, PrecomputedTextAdapter + factory `ocr_adapter_from_name` |
| `llm/` | AnthropicLLMAdapter, OpenAILLMAdapter, MistralLLMAdapter, OllamaLLMAdapter |
| `vlm/` | AnthropicVLMAdapter, OpenAIVLMAdapter, MistralVLMAdapter, OllamaVLMAdapter (héritage multiple `BaseVLMAdapter + BaseLLMAdapter`, MRO guard) |
| `corpus/` | local folder, IIIF, Gallica, HTR-United, HuggingFace Datasets, eScriptorium |
| `storage/` | `InMemoryArtifactStore`, `FilesystemArtifactStore`, `JobStore` (SQLite avec schema versioning) |
| `output_paths.py` | Helper partagé `resolve_output_path` (workspace-aware, read-only-mount-safe) |
| `_retry.py` | Helper partagé `call_with_retry` (3 retries, backoff 2/4/8s, sur 429+5xx+timeout réseau) |

**Règle** : un adapter peut importer le domain et ses libs externes.
Il ne doit **jamais** importer `app/` ou `interfaces/`.  Il n'a aucune
logique d'évaluation (un OCR adapter ne calcule pas le CER — il
produit un artefact texte que `evaluation/` consommera).

**Anti-SSRF** : `corpus/_http.py:validate_http_url` refuse
loopback, lien-local, RFC 1918, métadonnées cloud (AWS
`169.254.169.254`, GCP `metadata.google.internal`).  Verrouillé par
`tests/security/test_ssrf_attack.py` (Sprint S1.6).

### `picarones/app/` — services applicatifs

Orchestration entre adapters et evaluation.

| Module | Rôle |
|---|---|
| `services/run_orchestrator.py` | `RunOrchestrator.execute(RunSpec)` — point d'entrée d'un run complet |
| `services/benchmark_service.py` | `BenchmarkService.run` — exécute pipelines × vues × corpus, produit `RunResult` |
| `services/benchmark_runner.py` | Façade `run_benchmark_via_service` consommée par CLI/web |
| `services/job_runner.py` | `JobRunner` — soumission asynchrone (thread daemon) avec persistance `JobStore` |
| `services/corpus_service.py` | Loading + sandboxing + extraction ZIP avec ZIP slip protection |
| `services/dependencies.py` | `capture_dependencies_lock()` via `importlib.metadata` pour le `RunManifest` |
| `services/path_security.py` | `WorkspaceManager` — sandboxe par session |
| `services/registry_service.py` | Découverte des adapters et vues canoniques |
| `services/partial_store.py` | Persistance NDJSON des résultats partiels (reprise sur interruption) |
| `schemas/run_spec.py` | `RunSpec`, `StepSpec` — modèles YAML user-facing |
| `results.py` | `RunResult`, `RunDocumentResult`, `ReportRenderer` (alias type unique) |

**Anti ZIP slip** : `corpus_service._extract_safely` rejette les
chemins absolus, `..`, octets nuls, symlinks ZIP entries
(mode UNIX 0xA000), avec garde-fou final `target.resolve().relative_to(extract_dir)`.
Verrouillé par `tests/security/test_zip_slip_attack.py`.

### `picarones/reports/` — rendu déterministe

| Sous-package | Rôle |
|---|---|
| `csv/render.py` | `CsvReportRenderer` — un CSV plat (`run_id, doc, pipeline, view, metric, value, status`) |
| `json/render.py` | `JsonReportRenderer` — manifest + documents en JSON déterministe |
| `html/render.py` | `HtmlReportRenderer` — rapport autonome (TextView, AltoView, SearchView) — minimaliste |
| `html/generator.py` | `ReportGenerator` — rapport interactif riche (28 renderers + 5 vues) consommé par CLI/web |
| `narrative/` | Moteur narratif (20 détecteurs) — synthèse factuelle déterministe |
| `glossary/`, `i18n/` | Glossaire + i18n FR/EN |

Le rendu est strict : pas de JS dynamique côté serveur, pas d'I/O
hors écriture finale, déterministe bit-for-bit à entrée constante.
Permet à un relecteur 5 ans plus tard de hasher un rapport et de le
citer.

**Anti-XSS** : `html/generator.py` utilise
`autoescape=select_autoescape(['html', 'j2', 'xml'])` (Jinja2) +
helper `_safe_json_for_script_tag` qui encode `<>&` en
`<>&` pour le JSON injecté dans
`<script type="application/json">`.  Verrouillé par
`tests/security/test_xss_in_reports.py` (Sprint S1.1).

### `picarones/interfaces/` — points d'entrée user-facing

| Sous-package | Rôle |
|---|---|
| `cli/` | Click — 16+ commandes : `run`, `diagnose`, `economics`, `edition`, `compare`, `robustness`, `history`, `serve`, `metrics`, `engines`, `info`, `demo`, `report`, `import` (group) |
| `web/` | FastAPI — UI Jinja2 + SSE benchmark + ZIP upload + 11 routers (corpus, benchmark, jobs, reports, history, engines, normalization, importers, synthesis, system, home) |

**Anti-CSRF** : middleware `csrf_middleware` actif si
`PICARONES_CSRF_REQUIRED=1`.  Pattern double-submit cookie + HMAC
signature.  Verrouillé par `tests/security/test_csrf_required.py`.

## Principes architecturaux

### Pas de shim hors deprecation period

Un module a un seul emplacement canonique.  Quand un module migre,
on choisit explicitement entre :

- **Suppression dure** (pour la dette interne, pas de caller externe).
- **Shim avec `DeprecationWarning`** (pour la stabilité d'API publique).
  Le shim a une date de retrait inscrite dans le CHANGELOG.

À v2.0 il reste **un seul shim** documenté :
`picarones/pipeline/spec.py` (réexporte `picarones.domain.pipeline_spec`),
dont la deprecation period expire en v2.1.

### Pas d'`except Exception: pass`

Toute fonctionnalité optionnelle qui échoue émet un
`logger.warning("[module] feature dégradée : %s", exc)` avec contexte.
Vérifié par `tests/architecture/test_no_side_effect_imports.py`.

### Tests architecturaux comme garde-fous

Plusieurs tests verrouillent des invariants structurels que la revue
de code humaine raterait :

- `test_layer_dependencies.py` — couches strictement orientées
- `test_no_legacy_imports_in_rewrite.py` — `LEGACY_PACKAGES = ()`
- `test_file_budgets.py` — pas de god-modules
- `test_doc_paths.py` — chemins cités dans la doc existent
- `test_output_paths_uniformity.py` — tous les adapters passent par `resolve_output_path`
- `test_storage_keys_filesystem_safe.py` — clés du store filesystem-safe (Windows)
- `test_manifest_reproducibility.py` — `RunManifest` capture tout pour rejouer
- `test_module_coverage.py` — chaque module a un test associé

### Tests de sécurité comme verrous de défense

Sprint S1 a ajouté 63 tests d'attaque qui verrouillent les
défenses revendiquées :

- `tests/security/test_xss_in_reports.py` (5) — autoescape Jinja2 + escape JSON.
- `tests/security/test_xxe_attack.py` (9) — XXE / Billion Laughs / DTD.
- `tests/security/test_zip_slip_attack.py` (9) — ZIP slip + symlinks.
- `tests/security/test_ssrf_attack.py` (26) — loopback, RFC 1918, métadonnées cloud.
- `tests/security/test_csrf_required.py` (14) — double-submit + HMAC.

### Reproductibilité bit-for-bit

Le `RunManifest` capture systématiquement : `code_version`,
`pipeline_specs` complets, `adapter_kwargs`, `dependencies_lock`
(via `importlib.metadata`), `view_specs`, timestamps.  La
sérialisation est déterministe (Pydantic ordered fields, JSON
sorted keys).  Le hash du manifest peut être cité dans une
publication scientifique.

## Évolution

L'évolution de l'architecture est documentée :

- Backlog vivant : [`docs/roadmap/backlog.md`](../roadmap/backlog.md)
- Plans archivés (migration legacy → rewrite, terminée à v2.0) :
  [`docs/archive/2026-migration/`](../archive/2026-migration/)
- Roadmap historique pré-v2.0 :
  [`docs/archive/2026-roadmap/`](../archive/2026-roadmap/)
- Audits institutionnels : [`docs/archive/2026-audits/`](../archive/2026-audits/)
- Politique d'API publique : [`docs/reference/api-stable.md`](../reference/api-stable.md)
