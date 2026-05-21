# Changelog — Picarones

Tous les changements notables de ce projet sont documentés dans ce fichier.

Le format suit [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/).
La numérotation de version suit [Semantic Versioning](https://semver.org/lang/fr/).

---

## [Unreleased] — Migration Option B vers RunOrchestrator (mai 2026)

Branche `claude/test-alto-pipelines-qyFsL` — chantier de migration
complète vers `RunOrchestrator` comme entry-point canonique pour
lancer un benchmark, avec livrable métier ALTO documentaire.  20+
commits sur ~13 jours d'effort, 4830 → 4897 tests passants (+67).

### Nouveautés — valeur métier

- **`picarones.RunOrchestrator`** exposé au niveau racine.  Consomme
  un `RunSpec` Pydantic validé et expose 4 fichiers JSONL natifs
  (`run_manifest.json`, `pipeline_results.jsonl`,
  `artifacts_index.jsonl`, `view_results.jsonl`) en plus du
  `BenchmarkResult` legacy (via `spec.output_json`).
- **`RunSpec` étendu** avec 7 nouveaux champs : `char_exclude`,
  `normalization_profile`, `partial_dir`, `entity_extractor`,
  `profile`, `output_json`, `timeout_seconds_per_doc`.  Tous validés
  par Pydantic (`_validate_profile_is_known`,
  `_validate_entity_extractor_format`).
- **3 vues canoniques natives** dans le `RunResult` : `text_final`,
  `alto_documentary`, `searchability`.  Chacune produit ses propres
  `ViewResult` typés avec `metric_values`, `failed_metrics`,
  `projection_report`, `warnings`.
- **TesseractAdapter expose ALTO natif** via le flag `expose_alto`
  (off par défaut, compat ascendante).  Premier adapter du repo à
  produire un `Artifact ALTO_XML`.  Valide structurellement la
  sortie avant promotion (résistance XML mal formé).
- **AltoView étendu** : 7 métriques par défaut au lieu de 3.  Les
  4 nouvelles (`alto_text_cer/wer/mer/wil`) opèrent sur le texte
  plat extrait de l'ALTO via `extract_text_from_alto` — permettent
  de détecter une régression textuelle même quand la structure est
  préservée.
- **Rapport HTML multi-vues** : nouvelle section
  `view-results-section` rendue par
  `picarones/reports/html/renderers/view_results.py`.  Affiche un
  tableau `Métrique × engine` par vue avec moyennes, et liste
  explicitement les **pipelines OMIS** de chaque vue (critique pour
  AltoView : un OCR sans ALTO ne doit pas être omis silencieusement).
  Adaptive : section absente si `benchmark.view_results` vide
  (chemin legacy intact).
- **`BenchmarkResult.view_results`** : nouveau champ optionnel
  `{view: {engine: {doc: {metric: value}}}}` peuplé par le converter
  depuis le `RunResult.document_results[*].view_results`.  Consommé
  par le rapport HTML et accessible aux clients pour analyses
  ad-hoc.

### BREAKING — Retrait du legacy (Phase B3-final, migration nette)

Suppression complète de l'entry point legacy et de ses modules
helpers internes.  Les call sites CLI/Web/tests ont été migrés vers
le **pattern 3 étapes explicite** (Option 10 du chantier) :

```python
# Pattern moderne — 3 étapes visibles, pas de shim caché
args = prepare_preset_args(corpus, engines, workspace_dir=...)
orch_result = RunOrchestrator(out).execute_preset(**dataclass_fields)
result = run_result_to_benchmark_result(orch_result.run_result, ...)
```

- **`run_benchmark_via_service` supprimée** — utilisez
  `RunOrchestrator` + `prepare_preset_args` (Python) ou
  `RunOrchestrator.execute(RunSpec)` (YAML).
- **Modules supprimés en Phase B3-final** (~1700 LOC nettes pour
  cette phase isolée ; la branche complète cumule aussi l'audit
  code-quality qui ajoute massivement — voir entrée suivante) :
    * `benchmark_runner` (entry point legacy)
    * `_benchmark_execution` (helper interne orchestration)
    * `_benchmark_orchestration` (run_benchmark_unified /
      run_benchmark_with_partial)
    * `legacy_runner_compat` (shim intermédiaire B3 introduit puis
      supprimé dans le même chantier — voir entrée Option 10)
- **Tests d'invariance supprimés** — leur rôle (garde-fou pendant
  la migration) est rempli, la migration est terminée :
    * `tests/integration/test_migration_invariance.py`
    * `tests/integration/snapshots/migration_invariance.json`

### Modifié — Audit B3-final correctif (mai 2026)

L'audit implacable de la branche post-migration a identifié 4
demi-chantiers critiques : les features livrées en B5/B6/B2
étaient implémentées en interne mais **inaccessibles aux
utilisateurs CLI/Web**.  Corrections appliquées :

- **CLI** : 5 nouvelles options ajoutées à `picarones run` :
    * `--views VIEW1,VIEW2,…` (B6 multi-vues)
    * `--expose-alto` (B5 Tesseract ALTO XML)
    * `--char-exclude CHARS` (B2.5)
    * `--partial-dir PATH` (B2.3 resume)
    * `--entity-extractor DOTTED_PATH` (B2.4 NER)
  `_engine_from_name` propage `expose_alto` à Tesseract.
- **Web** : `BenchmarkRunRequest` étendu avec `views: list[ViewName]`,
  `profile`, `partial_dir`, `entity_extractor`, `output_json`.
  `PipelineConfig.expose_alto` activable par concurrent.  Le worker
  `run_benchmark_thread_v2` propage les nouveaux champs au pattern
  3 étapes.
- **Helper test** : `tests/_migration_helpers.run_via_orchestrator`
  reçoit un kwarg `views` (corrige la divergence test↔prod identifiée
  par l'audit — aucun test B4 ne couvrait précédemment le multi-vues
  via le helper).

Impact utilisateur :
```bash
picarones run -c ./corpus -e tesseract --expose-alto \
    --views text_final,alto_documentary,searchability \
    --profile standard
```
Génère désormais un rapport HTML avec 3 sections (TextView, AltoView,
SearchView) + ALTO XML natif de Tesseract.  La valeur métier C3
est enfin accessible aux utilisateurs.

### Modifié

- **`picarones.interfaces.cli._workflows`** : 6 commandes (`run`,
  `diagnose`, `economics`, `edition`, `compare`, `robustness`)
  utilisent désormais un helper local `_run_orchestrator_for_cli`
  qui mutualise le pattern 3 étapes.  Comportement utilisateur
  identique.
- **`picarones.interfaces.web.benchmark_utils.run_benchmark_thread_v2`** :
  pattern 3 étapes inline.  L'API REST `POST /api/benchmark/run`
  est inchangée pour les clients.
- **`picarones.app.services.__init__`** expose désormais
  `PresetArgs`, `prepare_preset_args`, `run_result_to_benchmark_result`
  comme API publique.
- **`TesseractAdapter.output_types`** : étendu de
  `{RAW_TEXT, CONFIDENCES}` à `{RAW_TEXT, CONFIDENCES, ALTO_XML}`
  (set maximal).  Les pipelines existants restent inchangés tant
  que `expose_alto` n'est pas activé.
- **`build_text_view` / `build_search_view`** : nouveau kwarg
  `char_exclude` propagé jusqu'au `DefaultEvaluationViewExecutor`
  qui filtre les caractères avant calcul des métriques.

### Migration utilisateur

Cf. `docs/archive/2026-migration/option-b-user-guide.md` pour le mapping complet
des paramètres legacy → `RunSpec`, 4 cas concrets (corpus mémoire,
partial_dir resume, NER attach, cancellation), et calendrier de
retrait phasé.

### Phases du chantier (référence interne)

- **B0** préparation (snapshot d'invariance + squelette feature
  parity + inventaire tests)
- **B1** `RunSpec` étendu (7 nouveaux champs + validators)
- **B2** porting des 7 features dans `RunOrchestrator`
  (progress_callback, cancel_event, partial_dir, entity_extractor,
  char_exclude + normalization_profile, profile hooks, output_json)
  → **Checkpoint C1**
- **B3** exports publics + `DeprecationWarning` + migration concrète
  des call sites CLI/Web via `legacy_runner_compat` → **Checkpoint C2**
- **B4** migration des 6 fichiers de tests catégorie A (71 appels)
- **B5** `TesseractAdapter.expose_alto` (premier adapter ALTO natif)
- **B6** rapport HTML multi-vues + extension `DEFAULT_ALTO_METRICS`
  → **Checkpoint C3** (valeur métier livrée)
- **B7** deprecation finale (bannières + CHANGELOG initial)
- **B3-final** (Option 10) — migration nette : helper
  `prepare_preset_args` + pattern 3 étapes inline dans CLI/Web/tests,
  puis suppression complète du shim `legacy_runner_compat` et des
  3 modules purement legacy.  **-1700 LOC nettes**.

---

## [Unreleased] — Audit code-quality (mai 2026)

Branche `claude/code-quality-audit-EeY0r` — audit implacable du repo
suite à la migration v2.0, suivi de 12 sprints correctifs (Phases 0
à 12 du plan d'audit).

### BREAKING — ruptures API v2.0

Toutes ces ruptures sont **immédiates**, sans calendrier de
dépréciation (règle de l'audit : « soit on supprime, soit on garde
et on développe »).  Migration directe documentée ci-dessous.

- **`POST /api/benchmark/start` retiré** au profit de
  `POST /api/benchmark/run`.  Le modèle Pydantic
  `BenchmarkRequest` (liste de moteurs plats) est remplacé par
  `BenchmarkRunRequest` (liste de `PipelineConfig`).  Helpers
  associés supprimés : `_legacy_request_to_run_request`,
  `run_benchmark_thread` (v1 ; `run_benchmark_thread_v2` reste).
  Migration :

  ```python
  # avant (v1.x)
  POST /api/benchmark/start
  {"corpus_path": "...", "engines": ["tesseract", "pero_ocr"]}

  # après (v2.0)
  POST /api/benchmark/run
  {
      "corpus_path": "...",
      "competitors": [
          {"name": "tesseract", "engine_name": "tesseract"},
          {"name": "pero_ocr", "engine_name": "pero_ocr"},
      ],
  }
  ```

- **`run_benchmark_via_service(..., max_workers=4)` retiré** —
  paramètre absorbé sans effet via `# noqa: ARG001`.  Le rewrite
  passe par `CorpusRunner.max_in_flight` directement.

- **`expand_legacy_keys()` retiré** de `picarones.domain.artifacts` —
  0 caller en production.  Le dict `LEGACY_VALUE_ALIASES` reste
  vivant pour le canonicalisation des manifests legacy.

- **JSON `BenchmarkResult` pré-v2.0 plus relisibles** — le retrait
  de `expand_legacy_keys` interdit le round-trip depuis des sorties
  v1.x.  Régénérer les benchmarks de référence.

- **Paramètre `text_hint` retiré** de
  `picarones.evaluation.synthetic._make_placeholder_png` (était
  jamais lu).

### Added — features inachevées débloquées

- **Agrégation sur-normalisation LLM corpus-wide** :
  `aggregate_over_normalization` câblée via
  `@register_corpus_aggregator(name="over_normalization", ...)`
  (profils `philological`, `diagnostics`, `full`).  Le hook
  extrait depuis `DocumentResult.pipeline_metadata["over_normalization"]`
  et alimente `EngineReport.aggregated_over_normalization`.
  Round-trip JSON préservé.

- **Journal de fallbacks importer end-to-end** : la chaîne
  `record_fallback → consume_fallback_log → BenchmarkResult.metadata
  → build_report_data → narrative.detect_importer_fallback` est
  désormais branchée.  Un fallback HTR-United mode démo apparaît
  dans la synthèse narrative avec traçabilité (URL, raison).

- **Profils de normalisation YAML versionnables** :
  - CLI : `picarones run --normalization-profile <ID-OR-PATH>` —
    accepte un identifiant builtin ou un fichier `.yaml`
    versionné dans git.
  - API : `POST /api/normalization/profiles/preview` — valide un
    YAML utilisateur et retourne le profil sérialisé (preview, pas
    de persistance).  Limite 64 KiB côté Pydantic.

- **`register_default_metrics()` exposée publiquement** —
  remplace le side-effect `import picarones.evaluation.metrics`
  opaque en tête de `picarones/__init__.py`.  Idempotente
  (`sys.modules` cache).  Auto-déclenchement préservé pour
  rétrocompat.

- **`RunResult` accessible depuis `picarones.pipeline.run_result`** —
  déplacé de `app.results` (compat shim conservé) vers la couche 4
  pour respecter l'orientation des couches (`reports/` ne peut
  plus importer depuis `app/`).

### Changed — architecture & lisibilité

- **8 `__init__.py` de couche** : renumérotation `Cercle N`
  (incohérent, max 5, doublons) → `Couche N` (1 à 8, ordre du
  manifeste).
- **`benchmark_runner.py`** : 1 700 → 1 584 LOC.  Extractions :
  - `_benchmark_ner.py` (NER aggregation, ~100 LOC).
  - `_benchmark_persistence.py` (sérialisation JSON, ~15 LOC).
  Budget `test_file_budgets` resserré de 1 750 à 1 620.
- **`robustness.py`** : 850 → 578 LOC.  Suppression des 5 helpers
  pure-Python `_apply_*` et du stub `_degrade_pure_python` (Pillow
  est dep obligatoire, fallback sans valeur).
- **`PipelineMode` unifié** : source unique
  `picarones.domain.pipeline_spec.PipelineMode`.  Les 3 alias
  historiques (`OCRLLMMode`, `OCRLLMPipelineMode`, `PipelineMode`)
  deviennent des re-exports.
- **Validation chemin web factorisée** : helpers
  `validated_user_path` / `validated_user_output_dir` dans
  `interfaces/web/_path_helpers.py` ; 2 routers migrés.
- **eScriptorium anti-SSRF** : `_get`, `_post` et le téléchargement
  d'image utilisent désormais `validate_http_url` et
  `download_url` (cohérence avec IIIF/Gallica/HTR-United).
- **Tesseract `lang` validation** : regex
  `^[a-zA-Z]{3,}(\+[a-zA-Z]{3,})*$` rejette les injections CLI
  (`fra --user-words /etc/passwd`).
- **CI : sync compteurs bloquant** — nouveau job ``sync-counters``
  exécute `scripts/gen_readme_tables.py --check`.
- **CI : 7 nouveaux tests d'architecture** verrouillent les
  invariants pour bloquer la régression :
  - `test_no_zombie_skips.py` (interdit `pytest.skip` sur dep
    obligatoire).
  - `test_no_legacy_imports_in_rewrite.py` refondu — test actif
    contre la résurrection des paquets legacy supprimés (au lieu
    d'un `LEGACY_PACKAGES = ()` vacuement vrai).
  - `test_no_broad_pytest_raises.py` — refuse les
    `pytest.raises(Exception)` (ratchet, baseline 24).
  - `test_logger_prefix.py` — refuse les logs sans préfixe
    `[<module>]` (ratchet, baseline 46).
  - `test_live_test_markers.py` — chaque fonction dans
    `tests/integration/live/` porte `@pytest.mark.live`.
  - `test_reports_layer_strict.py` — interdit les imports
    `reports/ → {adapters, app, interfaces}`.
  - `test_pipeline_mode_single_source.py` — refuse toute nouvelle
    redéfinition de `Literal["text_only", "text_and_image", "zero_shot"]`.
  - `test_api_stable_modules_exist.py` — chaque module cité dans
    `api-stable.md` doit s'importer.

### Removed

- **`POST /api/benchmark/start`** + helpers (cf. BREAKING).
- **`BenchmarkRequest`** modèle Pydantic v1 (cf. BREAKING).
- **`max_workers`**, **`text_hint`** paramètres morts.
- **`expand_legacy_keys()`** (cf. BREAKING).
- **5 helpers `_apply_*`** + **`_degrade_pure_python`** dans
  robustness.py (~300 LOC).
- **7 `pytest.skip("click non installé")`** zombies dans
  `tests/integration/test_chantier{4,5}.py` (click est dep
  obligatoire — skip vacuement vrai).
- **4 modules fantômes** retirés de `docs/reference/api-stable.md`
  (`pipeline.legacy_runner`, `pipeline.legacy_pipeline_benchmark`,
  `pipeline.legacy_pipeline_comparison`,
  `evaluation.metrics.pipeline_spec_loader`).

### Fixed

- **`app.py`** : entry point HuggingFace cassé
  (`picarones.web.app:app` → `picarones.interfaces.web.app:app`).
- **8 `except: pass` silencieux** : tous remplacés par
  `logger.warning("[<module>] ...")` ou `logger.debug(...)` selon
  criticité (friedman_nemenyi, image_quality, iiif, clustering,
  path_security, benchmark_runner, job_store, robustness).
- **10 `pytest.raises(Exception)` trop larges** : précisés en
  `FrozenInstanceError` ou `pydantic.ValidationError`.
- **README.md** : retrait du paragraphe « Legacy paths still
  present as shims » (faux depuis v2.0) ; `mypy picarones/core/`
  → `mypy picarones/domain/`.
- **CLAUDE.md** : compteur tests synchronisé (4 700 → réel),
  auto-contradiction « 12 vs 9 skipped » résolue, 18 → 20
  détecteurs, 22 → 28 renderers.
- **Faux positifs bandit B608** documentés avec `# nosec` et
  commentaire de justification (sites SQL où les `fields`
  interpolés sont des littéraux internes, valeurs via `?`).
- **eScriptorium** : urlretrieve sans validation → `download_url`
  avec anti-SSRF.

### Security

- **SSRF résiduel eScriptorium** fermé (`_get`, `_post`,
  téléchargement images).
- **Injection CLI Tesseract** bloquée (regex sur `lang`).
- **Aucune CVE** introduite (pip-audit vert).

### Stats

- **+158 nouveaux tests** (4 686 → 4 784 passing).  Ruff propre,
  bandit propre (1 LOW résiduel inoffensif), 0 régression.
- **12 nouveaux modules** créés (helpers extraits + tests
  d'invariant).
- **~600 LOC mortes supprimées**.

---

## [Unreleased] — Chantier post-rewrite (mai 2026)

Branche `claude/fix-module-rewiring-MHssX` — réconciliation des chemins
UI / API / runner / JSON / rapport après audit révélant des options
ignorées, moteurs annoncés sans backend, surfaces filesystem ouvertes
et round-trip JSON appauvri.

### Added

- **Adapters `KrakenAdapter` et `CalamariAdapter`** (couche 5) avec
  lazy imports.  Auparavant annoncés par `/api/engines` mais sans
  factory branchée — benchmark web échouait silencieusement.
- **Extras pyproject `[calamari]`** (`calamari-ocr>=2.0.0`) ;
  l'extra `[kraken]` préexistant pointe désormais sur un adapter réel.
- **`BenchmarkResult.from_dict` / `from_json_object`** restaurent
  fidèlement toutes les analyses avancées (taxonomy, confusion_matrix,
  NER, calibration, philological, searchability, hallucination,
  numerical_sequence, readability) ; le rapport régénéré depuis JSON
  est désormais indistinguable du in-memory.
- **`partial_store.compute_run_fingerprint` + `partial_path_for_engine`** :
  hash SHA-256 stable (engine_config + normalization_profile +
  char_exclude + corpus mtime/size + code_version) suffixé au nom du
  fichier partiel.  Deux runs avec configs différentes ne se
  contaminent plus.
- **Workflows CLI `diagnose`/`economics`/`edition`** génèrent
  désormais le rapport HTML automatiquement à côté du JSON
  (`--no-html` pour skipper).  Les docstrings vendaient déjà les vues
  HTML correspondantes.
- **UI** : boutons "💾 Sauvegarder config" / "📂 Charger config" dans
  l'écran benchmark (binding sur `/api/config/save` + `/api/config/load`
  qui existaient mais n'étaient appelés par personne).
- **UI** : bandeau "Mode démo" sous le titre HTR-United quand le
  catalogue distant est inaccessible (champ `is_demo` exposé par le
  router).
- **Tests** : `tests/security/test_phase1_post_rewrite_wiring.py`
  (~50 tests couvrant les 5 phases) + extensions de tests existants.

### Changed

- **`HTRUnitedCatalogue`** : router utilise `from_remote(timeout=5)`
  avec fallback automatique sur démo (au lieu de `from_demo()`
  exclusif).  Variable d'env `PICARONES_HTR_UNITED_OFFLINE=1` pour
  forcer démo (CI / déploiements offline).
- **`upload_purge_task` (RGPD)** : démarrée par le lifespan FastAPI
  (auparavant définie mais jamais lancée).  `JobStore.create_job`
  reçoit désormais un payload `{"corpus": req.corpus_path}` pour que
  la purge identifie les corpus actifs.
- **`/api/benchmark/start`** : worker unifié — délègue à
  `run_benchmark_thread_v2` après conversion
  `BenchmarkRequest → BenchmarkRunRequest`.  Marqué deprecated dans
  les logs ; un seul chemin à patcher.
- **CLI `engines`** : source de vérité unique avec
  `adapters/ocr/factory._SUPPORTED`.  Plus de hardcode local
  `[tesseract, pero_ocr]` qui divergeait du web.
- **`ReportGenerator.from_json`** : délègue à
  `BenchmarkResult.from_json_object` (simplification + fidélité).

### Breaking Changes

- **`CompetitorConfig` → `PipelineConfig`** (renommage de classe).
- **`PipelineConfig.ocr_engine` → `engine_name`** : le field accepte
  aussi `corpus` (OCR pré-calculé) et des VLMs en zero-shot — le
  préfixe `ocr_` était trompeur.  Les clients qui envoyaient
  `{"ocr_engine": "…"}` doivent migrer vers `{"engine_name": "…"}`
  (Pydantic v2 ignore silencieusement l'extra → benchmark refuse).
- **`PipelineConfig.pipeline_mode`** typé `Literal["text_only",
  "text_and_image", "zero_shot"]`.  Toute autre valeur (y compris
  les anciens alias `post_correction_text` /
  `post_correction_image`) est rejetée en 422 par Pydantic — fini
  le fallback silencieux vers `text_only` qui masquait les configs
  invalides.

### Security

- **`output_dir` validé** (`validated_path` contre
  `compute_workspace_roots`) dans `api_htr_united_import` et
  `api_huggingface_import` — plus de path traversal écriture.
- **`db_path` validé** dans `/api/history/regressions` — plus de
  lecture SQLite arbitraire.  Pour pointer une base hors workspace,
  exporter `PICARONES_HISTORY_DB`.
- **ZIP collision de basename** : `a/img.png` + `b/img.png` ne
  s'écrasent plus silencieusement — second renommé avec préfixe slug
  du dirname source.
- **ZIP image extraite validée** : `validate_image_safe` (Pillow.verify,
  anti-bombe, limite taille) appelé sur chaque image lors de
  l'extraction — auparavant les images extraites passaient sans
  vérif (zip bomb jusqu'à 500 Mo brut).

### Fixed

- Round-trip `BenchmarkResult.to_json` ↔ `from_json` préserve désormais
  l'intégralité des analyses (reproductibilité scientifique).
- Partial store fingerprint évite la réutilisation illégale de
  résultats entre runs avec configs différentes.

---

## [2.0.0] — Legacy retirement complete (mai 2026)

**Breaking changes** majeurs : suppression complète des paquets
legacy.  L'architecture canonique 8 couches (`domain → formats →
evaluation → pipeline → adapters → app → reports → interfaces`)
est la seule arborescence du code.  Aucun shim, aucun `_legacy/`,
aucun `legacy_*` subdir.

### Suppressions (toutes Sprints A-H, mai 2026)

**Top-level** :
- `picarones/core/` — Lots A-G : domain, formats, evaluation
- `picarones/measurements/` — Lot D + E : evaluation/metrics, statistics
- `picarones/engines/`, `picarones/modules/` — Lot E : adapters/legacy_*
- `picarones/report/` — Lot F : reports/html (Sprint H.3 : `reports_v2/` → `reports/`)
- `picarones/llm/`, `picarones/pipelines/`, `picarones/cli/`,
  `picarones/web/`, `picarones/extras/`, `picarones/fixtures.py`
  — Sprints F, G, H.1

**Adapters legacy** :
- `picarones/adapters/legacy_engines/` (Sprint H.2.d) :
  `BaseOCREngine`, `EngineResult`, `LegacyOCREngineExecutor`,
  `engine_from_name`, et les 5 adapters Tesseract/Pero/Mistral
  OCR/Google Vision/Azure DI legacy.  Remplacés par
  `picarones.adapters.ocr.*` (`BaseOCRAdapter` natif, factory
  `ocr_adapter_from_name`).
- `picarones/adapters/legacy_pipelines/` (Sprint H.2.c) :
  `OCRLLMPipeline`, `PipelineMode`.  Remplacés par
  `picarones.pipeline.llm_pipeline_config.OCRLLMPipelineConfig`
  + `picarones.pipeline.llm_pipeline_builder.make_ocr_llm_pipeline_spec`.
- `picarones/adapters/legacy_modules/` (Sprint H.2.a) :
  `TextToAltoMonoRegion`.

**Interfaces legacy** :
- `picarones/interfaces/cli/_legacy/` + stubs canoniques
  inachevés `interfaces/cli/{run,report,import_corpus}.py`
  (Sprint H.4) : consolidé en `interfaces/cli/` (16+ commandes
  Click).
- `picarones/interfaces/web/_legacy/` + stubs canoniques
  inachevés `interfaces/web/{__init__,app,security}.py` +
  `routers/`, `templates/`, `static/`, `i18n/`
  (Sprint H.4) : consolidé en `interfaces/web/` (FastAPI + UI
  Jinja2 + SSE benchmark + ZIP upload).

### Renames

- `picarones/reports_v2/` → `picarones/reports/` (Sprint H.3).
- `picarones/app/services/_legacy_runner_adapter.py` →
  `picarones/app/services/benchmark_runner.py` (Sprint H.4) :
  drop le préfixe `_legacy_` ; c'est l'entry point public des
  interfaces vers `BenchmarkService`.
- `picarones/app/services/_legacy_partial_store.py` →
  `picarones/app/services/partial_store.py` (Sprint H.4).

### Features ajoutées (Sprints D)

- `partial_dir` : reprise sur interruption (NDJSON per-engine,
  Sprint D.2.b) — un benchmark crashé peut reprendre sans
  perdre le travail déjà fait.
- `entity_extractor` : NER attach post-bench (Sprint D.2.e) —
  metrics NER calculées + agrégées sur les documents avec GT
  `ENTITIES`.
- `over_normalization` : détection automatique pour les
  pipelines OCR+LLM avec OCR amont (Sprint D.2.d).
- `validate_profile()` au démarrage du benchmark (Sprint D.2.f) :
  un profil inconnu lève `ValueError` avant tout calcul.

### Architecture

- `LEGACY_PACKAGES = ()` dans
  `tests/architecture/test_no_legacy_imports_in_rewrite.py` :
  plus aucun paquet legacy.
- `test_legacy_canonical_parity.py` supprimé (Sprint H.5) — la
  table de parité est sans objet à v2.0.
- `test_layer_imports_are_legal[layer-X]` passe pour toutes les
  couches.

### Migration depuis 1.x

```python
# AVANT (1.x)
from picarones.cli import cli
from picarones.engines.tesseract import TesseractEngine
from picarones.measurements.runner import run_benchmark
from picarones.pipelines import OCRLLMPipeline, PipelineMode
from picarones.report.generator import ReportGenerator

# APRÈS (2.0)
from picarones.interfaces.cli import cli
from picarones.adapters.ocr.tesseract import TesseractAdapter
from picarones.app.services.benchmark_runner import run_benchmark_via_service
from picarones.pipeline.llm_pipeline_config import OCRLLMPipelineConfig
from picarones.reports.html.generator import ReportGenerator
```

### Statistiques

- 4126 tests passing, 0 failed.
- ~10 paquets legacy supprimés.
- ~50000 LOC de legacy retirées.
- Architecture 8 couches respectée (vérifiée par 4 tests
  architecturaux : layer_imports, no_legacy_imports,
  layer_dependencies, file_budgets).

---

---

> **Historique pré-v2.0** : les versions antérieures à 2.0 (janvier 2025 → avril 2026)
> sont archivées dans
> [`docs/archive/changelog-pre-v2.md`](docs/archive/changelog-pre-v2.md).
