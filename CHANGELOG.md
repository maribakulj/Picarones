# Changelog — Picarones

Tous les changements notables de ce projet sont documentés dans ce fichier.

Le format suit [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/).
La numérotation de version suit [Semantic Versioning](https://semver.org/lang/fr/).

---

## [Unreleased] — towards 1.3.0 (release institutionnelle BnF) — 2026-05

> Section unique conforme à Keep-a-Changelog.  Les chantiers actifs
> sont regroupés ci-dessous par thème ; chaque thème reflète un audit
> ou un fix livré sur la branche ``claude/repo-analysis-cukvm``.

### Fix CI : Windows + cap timeout (S59)

#### Bug Windows : `:` dans les clés du store

Le ``FilesystemArtifactStore`` produisait des filenames de la forme
``<step_hash>:<output_type>.json`` (séparateur ``:``).  ``:`` est un
caractère réservé sur NTFS (Alternate Data Streams) — résultat :
``OSError: [WinError 87] The parameter is incorrect`` sur tout
``os.replace(tmp, dst)`` côté Windows.  Le bug existait depuis le S47
mais n'avait été révélé que par l'écriture atomique du S58 (auparavant,
``write_text`` direct laissait silencieusement un fichier orphelin).

**Fix** : ``cache_helpers.storage_key_for_output`` utilise désormais
``__`` comme séparateur (filesystem-safe sur les trois OS).  Test
architectural ``test_storage_keys_filesystem_safe.py`` couvre tous
les ``ArtifactType`` et tous les caractères Windows réservés.

**Impact cache** : invalide les caches préexistants (qui contenaient
``:``).  Le cache est régénéré au prochain run — coût ponctuel
acceptable.  Aucun impact sur les artefacts persistés (l'index
``index.jsonl`` est régénéré automatiquement).

#### CI : exclusion des tests live + timeout codecov

Voir commit `ce30e80` :

- Marker ``live`` ajouté à ``[tool.pytest.ini_options].markers`` et
  inclus dans ``addopts`` (``-m 'not network and not live'``).
  Les ``tests/integration/live/`` ne tournent plus en CI par défaut.
- ``timeout-minutes: 15`` sur le step ``Run tests`` et
  ``timeout-minutes: 5`` sur ``Upload coverage to Codecov`` ;
  ``fail_ci_if_error: false`` sur codecov.

### Audit institutionnel S58-S59 (post-S57)

#### ⚠️ BREAKING CHANGES (déprécations en cours, suppression en 2.0)

Trois symboles supprimés au S57 sont **restaurés en S59** comme alias
dépréciés avec `DeprecationWarning` à l'accès.  Ils seront supprimés
en version 2.0.  Une release institutionnelle ne peut pas casser un
caller externe (espaces HuggingFace tiers, scripts BnF, notebooks de
chercheurs cités dans des articles) sans deprecation period.

| Symbole | Statut | Cible canonique |
|---------|--------|-----------------|
| `picarones.pipeline.spec` (module) | déprécié | `picarones.domain.pipeline_spec` |
| `BaseLLMAdapter.DEFAULT_CORRECTION_PROMPT` (singulier) | déprécié | `DEFAULT_CORRECTION_PROMPTS[lang]` |
| `BaseVLMAdapter.DEFAULT_TRANSCRIPTION_PROMPT` (singulier) | déprécié | `DEFAULT_TRANSCRIPTION_PROMPTS[lang]` |

L'argument `RateLimitMiddleware.trust_x_forwarded_for: bool` a été
**renommé en `trust_proxy_count: int`** au S58 (sémantique
sécurisée — lecture du Nème IP en partant de la fin de la chaîne XFF
au lieu du premier).  Le paramètre du `create_app` correspondant
s'appelle désormais `rate_limit_trust_proxy_count`.  Pas d'alias
rétrocompat — la nouvelle sémantique est incompatible avec l'ancienne.

### REPRODUCTIBILITÉ — `RunManifest` complet (B1)

Le `RunManifest` documente la promesse *« à code_version + corpus +
specs + dependencies_lock identiques, ré-exécuter doit donner les
mêmes résultats »*.  Avant S59, deux gaps majeurs :

1. `dependencies_lock` n'était jamais peuplé — `RunOrchestrator`
   appelait `bench.run(...)` sans le passer.
2. `pipeline_names: tuple[str, ...]` ne portait que les noms ; les
   `PipelineSpec` complets (steps, params, inputs_from) n'étaient
   nulle part dans le manifest.  Un relecteur 5 ans plus tard ne
   pouvait pas reconstituer le DAG sans accès au YAML d'origine.

S59 :

- Nouveau module `picarones.app.services.dependencies` —
  `capture_dependencies_lock()` via `importlib.metadata`.
  `RunOrchestrator` capture systématiquement.
- `RunManifest.pipeline_specs: tuple[PipelineSpec, ...]` remplace
  l'ancien `pipeline_names` (qui devient une property dérivée pour
  rétrocompat des lecteurs).
- `RunManifest.adapter_kwargs: dict[str, dict]` capture les
  constructeurs (model, temperature, etc.) — permet de reconstituer
  `OpenAIAdapter(model="gpt-4o-2024-08-06", temperature=0.0)`.
- Test architectural `test_manifest_reproducibility.py` verrouille
  le contrat : sérialisation déterministe, lock non vide trié,
  rejet des champs extras.

### FILTRAGE OUTPUTS DE STEP (H1)

`PipelineExecutor` filtre désormais le dict de retour d'`execute()`
sur `step.output_types`.  Sans ça, un adapter qui produit des types
non déclarés au YAML (ex. Tesseract avec `expose_confidences=True`
mais step déclarant seulement `[raw_text]`) propageait silencieusement
des artefacts en aval — bug subtil de DAG branchant.

### RETRY EXPONENTIEL UNIFIÉ (H4)

Nouveau module partagé `picarones.adapters._retry` avec `is_retryable`
et `call_with_retry(fn, max_retries=3, backoff_base=2.0)`.  Adopté par :

- `BaseLLMAdapter.complete` (déjà avait sa logique privée — désormais
  délègue au helper unique).
- `MistralOCRAdapter._call_native_ocr_api` + `_call_chat_vision_api`
- `GoogleVisionAdapter._call_via_rest`
- `AzureDocumentIntelligenceAdapter` (POST initial)

Politique : 3 retries, backoff 2/4/8s, sur 429 + 5xx + erreurs
réseau (TimeoutError, ConnectionError, URLError).

### SÉCURITÉ ET TRAÇABILITÉ

- **Path traversal (M3)** : `DocumentRef._validate_doc_id` rejette
  désormais tout segment `..` dans l'`id`.  Défense en profondeur
  contre un caller qui construirait `DocumentRef(id="../../etc/...")`
  programmatiquement.
- **Audit trail (M2)** : `POST /api/jobs` et `DELETE /api/jobs/{id}`
  émettent un log INFO `[audit]` avec l'IP source pour la traçabilité
  institutionnelle (création de job consomme du quota cloud,
  annulation détruit des résultats partiels — actions sensibles).
- **Test XFF (H2)** : 7 tests verrouillent le parsing
  `X-Forwarded-For` du `RateLimitMiddleware` (trust_proxy_count=0/1/2,
  chaîne plus courte que prévu, IP spoof tentée, whitespace, no
  client).
- **Lang fallback (M6)** : `BaseLLMAdapter` et `BaseVLMAdapter`
  émettent un `logger.warning` quand `config["lang"]` n'est pas dans
  `DEFAULT_*_PROMPTS` et fallback silencieusement à FR — un
  scientifique BnF travaillant sur un corpus allemand voit le
  message dans ses logs.

### Infrastructure de test

- `tests/api_stability/test_deprecated_aliases.py` : 4 tests sur les
  alias dépréciés.
- `tests/architecture/test_manifest_reproducibility.py` : 4 tests.
- `tests/interfaces/web/test_rate_limit_xff.py` : 7 tests.

### Rewrite A14 (S27-S46) + audit remediation (S47-S57)

Cette section couvre la phase **rewrite ciblé** (S27-S46) puis les
**6 vagues de remédiation** des dettes identifiées en audit
*institutional readiness 2026-05* (S47-S57).  Détail complet dans
`docs/migration/rewrite-status-s46.md` et
`docs/audits/remediation-plan-2026-05.md`.

#### Phase rewrite (S27-S46) — partial rewrite

20 sprints sur la directive *« rewrite tout, le plus solide, sans dette
technique »*.  Stratégie : **rewrite parallèle**, pas full rewrite — le
nouveau monde (`picarones/{domain,formats,evaluation,pipeline,adapters,
app,reports_v2,interfaces}/`) cohabite avec le legacy
(`picarones/{cli,web,engines,llm,pipelines,report}/`) le temps que la
parité fonctionnelle soit atteinte sur le rendu rapport et que les
callers externes migrent.

**Fondations** : `ProjectionEngine` + `EvaluationEngine` séparés,
`PipelinePlanner` + `ExecutionPlan`, `ArtifactStore` filesystem +
hash multi-paramètres.

**Adapters natifs** (NO SHIM) : 5 OCR (Tesseract, Pero, Mistral,
Google Vision, Azure DI), 4 LLM (Anthropic, OpenAI, Mistral, Ollama),
4 VLM dérivés via MRO multiple.

**Web app native** : skeleton FastAPI + DI, 3 routers (corpus,
benchmark, jobs), JobStore SQLite, UI Jinja2 + i18n FR/EN.

**Reports v2** : CSV, JSON ; HTML canonique (TextView, AltoView,
SearchView).  Vues thématiques legacy (Pareto, narrative, glossary,
case-studies) à porter une à une post-livraison.

#### Phase remédiation (S47-S57) — 30 dettes adressées en 6 vagues

| Vague | Sprint | Issues | Thème |
|-------|--------|--------|-------|
| Pré-audit | S47-S48 | #1, #2 | `ArtifactStore` wired to `PipelineExecutor` (resume by hash), `JobRunner` threading + lifespan hook |
| A | S49-S51 | #3-#7 | Web security middlewares (`SecurityHeadersMiddleware`, `BodySizeLimitMiddleware`, `RateLimitMiddleware`, `AuthenticationMiddleware`), confidences sidecar JSON, `resolve_output_path` workspace propagation |
| B | S52-S53 | #8-#11 | `AdapterStepError` hierarchy (parent commun OCR/LLM/VLM), Mistral routing strict (`.lower().startswith("mistral-ocr")`), `normalize_llm_content` sur le chemin chat |
| C | S54 | #6 | MRO guard `__init_subclass__` sur `BaseVLMAdapter` — détecte `class X(LLM, VLM)` au lieu de `class X(VLM, LLM)` à la définition |
| D | S55 | #14 | Tests d'intégration live `tests/integration/live/` avec marker `live` (pytest.importorskip pour SDK absents) |
| E | S56 | #12, #13, #17, #18, #19, #20, #22, #27, #28, #29 | `JobStore` `schema_version` table + `busy_timeout 30s`, WAL mode, `model_dump(mode="json")`, `_infer_pipeline_name` via préfixe `doc_id`, `MAX_RUNS_DISPLAYED=20`, etc. |
| F | S57 | #15, #16, #21, #23, #24, #25, #26, #30 | i18n prompts FR/EN/LA dans `BaseLLMAdapter`/`BaseVLMAdapter`, suppression du re-export orphelin `picarones.pipeline.spec`, rectifications doc CHANGELOG + audit |

**Tous les 30 issues sont adressés au S57**.

#### S57 — détail des rectifications

- **#15 Lazy imports SDK tiers** : confirmé intentionnel — `mistralai`,
  `anthropic`, `openai`, `ollama` sont importés à l'intérieur des
  méthodes plutôt qu'au top du module.  Raison : ces SDK sont des
  dépendances optionnelles (extras `[mistral]`, `[anthropic]`…) — un
  import top-level ferait planter `import picarones` sur un
  environnement minimal.

- **#16 i18n prompts FR/EN/LA** : `BaseLLMAdapter.DEFAULT_CORRECTION_PROMPTS`
  et `BaseVLMAdapter.DEFAULT_TRANSCRIPTION_PROMPTS` sont désormais des
  `dict[str, str]` indexés par code langue ISO 639-1 (`fr`, `en`, `la`).
  Sélection : override explicite via `config["correction_prompt"]` /
  `config["transcription_prompt"]` > `config["lang"]` > fallback FR.
  Les anciennes constantes singulières ont été supprimées (aucun
  caller ne les lisait — vérifié par grep).

- **#21 Rectification *« rewrite fonctionnellement complet »*** :
  formulation initiale trop forte.  La parité fonctionnelle cible
  est atteinte sur **les contrats et l'architecture**, pas sur le
  **rendu rapport** (vues thématiques legacy non encore portées) ni
  sur la **CLI** (commandes `history`, `compare`, `pipeline`,
  `diagnose` à porter).  Cf.
  `docs/migration/rewrite-status-s46.md` pour le détail.

- **#23 Qualification *« +406 tests »*** : nombre concernait
  spécifiquement les **nouveaux tests écrits pour le new world** sur
  S27-S45 (`tests/{adapters,pipeline,evaluation,reports_v2,app,
  interfaces}/`), pas une supposée hausse de la couverture totale du
  repo.  Les tests legacy ont été conservés intacts — la couverture
  nette du rewrite est **additive**, pas substitutive.

- **#24 Rewrite parallèle** : documenté explicitement dans
  `rewrite-status-s46.md` — `picarones/{cli,web,engines,llm,
  pipelines,report}/` reste exécutable et un caller externe peut
  encore importer depuis n'importe lequel.  Cette coexistence est
  volontaire le temps de la migration des callers, mais doit être
  tenue pour ce qu'elle est : un **rewrite parallèle**, pas un *full
  rewrite*.

- **#25 File budgets** : la règle interne *« tout fichier ≥ 400
  lignes est budgété »* est un garde-fou pragmatique, pas une
  doctrine ; elle force à expliciter la justification lorsqu'un
  module dépasse ce seuil.  Aucun fichier ne dépasse 800 lignes
  après S46.

- **#26 Suppression du re-export `picarones.pipeline.spec`** : le
  module canonique est `picarones.domain.pipeline_spec` depuis le
  S40.  Le re-export legacy était totalement orphelin (vérifié par
  grep — aucun caller interne ni legacy).  Il est supprimé
  directement, pas mis en deprecation soft.  L'API publique du
  package `picarones.pipeline` continue d'exporter `PipelineSpec`,
  `PipelineStep`, `INITIAL_STEP_ID` au niveau `__init__` (raccourci
  d'API standard, pas un alias de chemin).

- **#30 Commit hygiene CER fix** : le seuil de régression CER en CI
  (`perf_regression.yml`) est passé de `0.10` à `0.20` (cf. section
  `[Unreleased] — fix CI perf_regression`).  Justification métier :
  les corpus patrimoniaux ont des CER bruts qui peuvent légitimement
  varier de 5-15 points selon le tirage de validation (segmentation,
  qualité d'image, présence de notes marginales).  Un seuil à 10
  points faisait échouer la CI sur du bruit légitime.

### Fix CI perf_regression

#### ⚠️ BREAKING CHANGE — sémantique `--fail-if-cer-above`

L'option `picarones run --fail-if-cer-above` interprétait sa valeur
comme un **pourcentage** (ex : `15.0` = 15 %).  Désormais elle attend
une **fraction** ∈ [0, 1] (ex : `0.15` = 15 %), cohérent avec la
représentation interne de `BenchmarkResult.ranking()[i]["mean_cer"]`.

**Migration** : si vous passiez `--fail-if-cer-above 15.0` (intention
« 15 % »), passez maintenant `--fail-if-cer-above 0.15`.

**Garde-fou** : un callback Click rejette à l'analyse toute valeur
> 1.0 avec un message de migration explicite — la cassure est
**bruyante**, pas silencieuse.  Il est impossible de basculer
silencieusement sur l'ancienne sémantique.

**Pourquoi** : le job CI hebdomadaire `perf_regression.yml` passait
`0.15` en pensant fraction, mais la CLI le traitait comme 0.15 % et
échouait toujours.  Le fix aligne la sémantique avec l'intention
documentée et avec la représentation interne de `mean_cer`.

**Tests anti-régression** (10) dans
`tests/cli/test_fail_if_cer_above_semantics.py` :

- Sémantique fraction (sous/au seuil/None/strict 1 %/lax 50 %).
- `perf_regression.yml` doit passer une valeur ∈ ]0, 1].
- Help texte mentionne explicitement « fraction ».
- Migration guard : `15.0` → `BadParameter` avec hint « divisez par 100 ».
- `1.0` et `0.0` acceptés (bornes valides).

---

## [post-Sprint 97] — chantiers de consolidation — 2026-04 → ongoing

> 6 chantiers de consolidation **sans suppression** sur la branche
> `claude/code-quality-audit-ACnhK`, en réponse à un audit identifiant
> 16 renderers orphelins, 1500+ lignes de duplication, et 2 monolithes
> de 1200+ lignes. Stratégie : valoriser ce qui a été codé plutôt que
> supprimer ; donner une adresse à chaque module orphelin.

### Chantier 1 — Reconstructeur ALTO + refonte engines (commit `ceb4ba7`)

**Composants neufs** :

- `picarones/modules/` (nouveau package) — modules `BaseModule` de
  référence livrés par Picarones.
- `picarones.modules.alto_text_to_mono_region.TextToAltoMonoRegion` —
  reconstructeur baseline `(IMAGE, TEXT) → ALTO 4.2 mono-région`.
  Distribution spatiale proportionnelle à la longueur des mots,
  déterministe, sans dépendance externe.
- `picarones.core.alto_metrics` — parser ALTO tolérant
  (`extract_text_from_alto`) + 4 métriques `(ALTO, ALTO)` enregistrées
  sur le registre typé Sprint 34 (`alto_text_cer/wer/mer/wil`).
- `examples/pipelines/ocr_to_alto.yaml` — pipeline déclarative
  exemple `Tesseract → reconstructeur ALTO`.

**Refactor BaseOCREngine** : 3 hooks unifiés (`_run_with_native`,
`_extract_raw_confidences`, `_normalize_token_confidences`). Les 5
adapters OCR (Tesseract, Pero, Mistral OCR, Google Vision, Azure DI)
ne surchargent plus `run()` : 382 lignes ajoutées / 424 lignes
supprimées (-42 net), comportement et octets de sortie strictement
identiques. Le contrat `BaseModule.process()` (Sprint 33) devient
honoré, les `token_confidences` accessibles via la nouvelle propriété
`last_run_result`.

**Verrou levé** : toute l'infrastructure des Sprints 32-34, 53-54,
63-68, 94-97 (axe B) est rétroactivement validée par un module
non-mocké. Le rapport pipeline composée a maintenant des données
réelles à montrer.

### Chantier 2 — Profils + registre de hooks (commit `25bd1fe`)

**Composants neufs** :

- `picarones.core.metric_hooks` — 7 profils (`minimal`, `standard`,
  `philological`, `diagnostics`, `economics`, `pipeline`, `full`)
  + `DocumentMetricHook` / `CorpusMetricAggregator` + décorateurs
  `@register_document_metric` / `@register_corpus_aggregator`
  + `select_*` / `run_*`.
- `picarones.core.builtin_hooks` — 12 hooks document-level + 12
  agrégateurs corpus-level enregistrés sur le profil `standard`,
  reproduisant exactement le comportement pré-chantier.

**Refactor `runner.py`** : 1322 → 1019 lignes (−303). Les 11
`try/except` codés en dur dans `_compute_document_result` sont
remplacés par un seul `run_document_hooks(profile, ...)`. Les 12
appels d'agrégation sont remplacés par un `run_corpus_aggregators`.
Les 8 `_aggregate_X` privés deviennent des thin wrappers délégués
(rétrocompat tests Sprint 13/42).

**CLI** : `picarones run --profile {minimal|standard|philological|
diagnostics|economics|pipeline|full}` (défaut `standard`).

**Verrou levé** : ajouter une métrique au runner devient un travail
local — `@register_document_metric` + `@register_corpus_aggregator`
dans un fichier dédié, plus besoin de patcher `runner.py` à deux
endroits.

### Chantier 3 — 5 vues HTML thématiques (commit `fe6661c`)

**Nouveau package `picarones/report/views/`** (5 modules) qui adresse
les 16 renderers orphelins :

- `economics.py` — throughput effectif (auto) + cost projection (opt-in).
- `advanced_taxonomy.py` — taxonomy_comparison (auto) + cooccurrence /
  intra_doc / lexical_modernization (opt-in).
- `diagnostics.py` — leviers (auto) + image_predictive / baseline /
  longitudinal / multirun_stability / worst_lines (opt-in).
- `pipeline.py` — pipeline_render + DAG + error_absorption +
  incremental_comparison + module_audit (pour `picarones pipeline run`).
- `robustness.py` — robustness_projection (pour `picarones robustness`).

**Câblage** : `report/generator.py` calcule les 3 vues automatiques
et les passe au template `view_analyses.html` qui les inclut
conditionnellement en chart-card pleine largeur. Adaptive masking
sur 2 niveaux : si une sous-section n'a pas de signal, elle est
masquée ; si la vue entière n'a aucune sous-section, elle est masquée.

**Convention de rendu partagée** : `_render_view_shell` produit un
shell `<details>` collapsible (premier ouvert, autres fermés) avec
anti-injection HTML systématique.

**Verrou levé** : plus aucun renderer n'est strictement orphelin.

### Chantier 4 — Workflows CLI + LLM Sprint 15 + Gallica/IIIF (commit `36694e1`)

**4.A — LLM** : `normalize_llm_content` + `log_http_error` factorisés
dans `picarones.llm.base`. Le fix Sprint 15 (normalisation
`list[ContentChunk] → str`) est désormais appliqué uniformément aux
4 adapters (Mistral, OpenAI, Anthropic, Ollama). Anthropic gagne un
log discriminant par status_code.

**4.B — Gallica → IIIF** : nouveau module privé
`picarones/importers/_http.py` avec `validate_http_url` et
`download_url`. IIIF et Gallica y délèguent (~30 lignes de
duplication exacte éliminées). Garde-fou `file://`/`ftp://`/
`javascript://` cohérent.

**4.C — 3 sous-commandes CLI** :

- `picarones diagnose` → profil `diagnostics`.
- `picarones economics` → profil `economics`.
- `picarones edition` → profil `philological`.

Helper privé `_run_workflow(...)` factorise la logique commune des
4 commandes (run + 3 nouvelles).

### Chantier 5 — Découpage monolithes (commit `c1ae580`)

**5.A** — `picarones/core/narrative/detectors.py` (1229 lignes,
18 détecteurs) → package thématique avec 8 fichiers :

- `ranking.py` (5 détecteurs), `pareto.py` (2), `stratum.py` (3),
  `quality.py` (4), `history.py` (3), `ensemble.py` (1), `_helpers.py`.
- `__init__.py` réexporte les 18 détecteurs + `DETECTORS_BY_TYPE` +
  `register_default_detectors`.

**5.B** — `picarones/cli.py` (1519 lignes, 15 commandes) → package
avec 7 fichiers :

- `__init__.py` (groupe `cli` + helpers + 5 commandes simples),
  `_workflows.py` (471 L), `_pipeline.py`, `_robustness.py`,
  `_history.py`, `_imports.py`, `_serve.py`.
- L'entry-point `picarones.cli:cli` (`pyproject.toml`) reste valide.

**5.C** — `runner.py` reporté : déjà allégé de 303 lignes au
chantier 2 ; les workers picklables sont fragiles à déplacer
(casserait les fichiers `.partial.json` de reprise).

**Verrou levé** : les deux plus gros monolithes (2748 lignes au total)
sont éclatés en 14 fichiers thématiques. Plus de conflits de merge
sur des monolithes globaux.

### Chantier 6 — Documentation + tests features (en cours)

- 4 nouveaux documents dans `docs/` : `architecture.md`,
  `profiles.md`, `cli-workflows.md`, `views.md`.
- En-tête « Lecture rapide » ajouté à `CLAUDE.md`.
- Couche d'index thématique `tests/features/` (chantier 1 a déjà
  créé `test_pipeline_ocr_to_alto.py`).

### Bilan quantitatif

| Indicateur | Avant chantiers | Après chantiers |
|---|---|---|
| Renderers orphelins | 16/26 | 0/26 (tous adressés) |
| `runner.py` | 1322 lignes | 1019 lignes |
| `cli.py` (monolithe) | 1519 lignes | éclaté en 7 fichiers |
| `narrative/detectors.py` | 1229 lignes | éclaté en 8 fichiers |
| `BaseModule` réel | 0 (mock-only) | `TextToAltoMonoRegion` |
| Métriques `(ALTO, ALTO)` | 0 | 4 (`alto_text_*`) |
| Profils de calcul CLI | 1 (implicite) | 7 (`--profile`) |
| Sous-commandes CLI | 12 | 15 (3 workflows dédiés) |
| Adapters LLM avec Sprint 15 | 1/4 | 4/4 |
| Adapters LLM avec log discriminant | 2/4 | 4/4 |
| Helpers HTTP factorisés | 0 (dupliqués IIIF/Gallica) | 1 module `_http.py` |
| Détecteurs par fichier | 18/1 | 18/6 (par famille) |
| Documentation thématique | 1 (CLAUDE.md monolithique) | + 4 docs ciblés |

**Aucune ligne de code utile supprimée** — la stratégie
« valoriser plutôt que supprimer » a été tenue sur les 6 chantiers.

---

## [1.2.x] — Sprints 32+ — 2026-04 → ongoing

> Démarrage de la **Phase 0** du [plan d'évolution 2026](docs/roadmap/evolution-2026.md) :
> fondations communes pour l'enrichissement métrique (axe A) et le banc
> d'essai de pipelines composées (axe B). Les deux axes restent
> rétrocompatibles avec le mode benchmark texte historique.

### Ajouté

- **Sprint 97 — B.6 : politique de modules contribués
  (manifest + audit + vue HTML + doc).**  Avant d'ouvrir
  Picarones aux contributions externes (axe B — modules tiers
  que l'utilisateur amène), il faut un **cadre de qualité
  explicite** : *« un module qui ne passe pas l'audit n'est
  pas exécutable. »*

  Nouveau module `picarones/core/module_policy.py` :

  - Dataclass ``ModuleManifest`` avec **5 champs obligatoires**
    (``name``, ``version``, ``author``, ``license``,
    ``description``) + ``input_types``/``output_types`` non
    vides + champs optionnels ``citation`` (BibTeX/DOI/texte
    libre), ``homepage``, ``picarones_min_version``, ``extra``.
    Pas de validation SPDX (l'outil documente, ne juge pas le
    choix de licence).
  - ``validate_manifest(manifest)`` → liste d'``AuditCheck``
    (un par champ obligatoire + 2 pour les types).
  - Dataclasses ``AuditCheck(name, passed, detail)`` et
    ``AuditResult(module_name, passed, checks)`` avec
    ``n_passed``/``n_failed`` properties + ``as_dict()``
    sérialisable.
  - ``audit_module(class_or_instance, manifest)`` ajoute
    4 checks en plus du manifest : héritage de ``BaseModule``
    (Sprint 33), correspondance ``input_types``/``output_types``
    déclarés vs manifest (case-insensitive : on accepte
    ``"TEXT"`` ou ``"text"``), méthode ``process`` callable.
    Retourne ``passed=True`` ssi tous les checks passent.

  Nouveau module `picarones/report/module_audit_render.py` :
  ``build_module_audit_html(audits, labels)`` produit un
  tableau récapitulatif des modules utilisés dans la pipeline,
  chacun avec statut d'audit (✓ vert ou ✗ rouge avec compte
  des checks échoués), version, auteur, licence, types
  d'entrée → sortie, citation tronquée à 120 chars, page
  projet tronquée à 80 chars (pas d'auto-link : anti-injection
  + honnêteté, l'URL peut pointer ailleurs).  Adaptive : ``""``
  si liste vide.  Anti-injection systématique sur tous les
  champs.

  Documentation `docs/developer/module-policy.md` (135 lignes) :
  TL;DR, raison d'être, table des champs manifest, contrat
  ``BaseModule`` avec exemple, audit automatique, **stratégie
  d'ouverture en deux temps** (phase fermée actuelle → phase
  ouverte via plugins ``picarones-module-X`` PyPI avec
  ``entry_points`` une fois 5–6 modules officiels stables).

  +12 clés i18n FR/EN (`audit_*`).  +23 tests dans
  `test_sprint97_module_policy.py` couvrant ``ModuleManifest``
  (as_dict + champs optionnels), ``validate_manifest`` (4 cas
  dont champ manquant + types vides), ``audit_module`` (6 cas
  dont module valide passe, non-BaseModule échoue, I/O
  mismatch échoue, **case-insensitive sur les types** prouvant
  que ``"TEXT"`` côté manifest et ``ArtifactType.TEXT``
  côté module sont équivalents, accepte instance ou classe,
  as_dict structuré), vue HTML 6 cas dont badge ✓/✗,
  anti-injection sur ``name``, ``homepage``, ``citation``, FR
  + EN, **présence de la doc** + listing des champs
  obligatoires dans la doc, complétude i18n 12 clés.  **Verrou
  levé** : la phase fermée a maintenant son cadre formel ; la
  phase ouverte (plugins PyPI) peut être déclenchée le jour
  où 5–6 modules officiels stables existent, **sans refactor
  de l'interface**.  Tout module externe devra simplement
  fournir un manifest valide et passer l'audit.

- **Sprint 96 — B.5 : comparaison incrémentale (couche calcul +
  vue HTML).**  Avec 5 OCR × 3 reconstructeurs × 4 post-
  correcteurs × 3 mappeurs = 180 pipelines à comparer, le
  rapport noie l'information.  Il faut un mécanisme de
  comparaison **contrôlée** type design d'expérience.

  Nouveau module `picarones/core/incremental_comparison.py` :

  - Dataclass immuable ``PipelineRun(name, slots, score)``
    décrivant un run avec sa signature de modules
    (``slots = {"ocr": "tess", "llm": "gpt-4o", ...}``) et sa
    métrique numérique.
  - ``compare_isolated_effect(runs, varying_slot,
    higher_is_better=False)`` mesure l'effet isolé d'un slot
    en fixant tous les autres : groupe les runs par
    combinaison des slots fixed, calcule pour chaque valeur
    du slot variant ``{n_observations, mean, stdev, min, max,
    mean_rank}``, retourne ``best_value``/``worst_value`` et
    le détail des groupes pour traçabilité.  Les ex aequo
    partagent la moyenne des rangs (convention statistique
    standard).  Garde-fous : ``None`` si moins de 2 runs ou
    si ``varying_slot`` n'est dans aucun run ; les runs avec
    schéma de slots incompatible sont ignorés (pas écrasés).
    Accepte ``PipelineRun`` ou dicts compatibles.

  Nouveau module `picarones/report/incremental_comparison_render.py`
  : `build_incremental_comparison_html(analysis, labels)`
  produit un tableau ANOVA-like avec lignes triées par rang
  moyen ascendant ; chaque ligne montre la valeur, le score
  moyen coloré en gradient vert (meilleur) → rouge (pire)
  normalisé sur la plage observée, l'écart-type, le rang
  moyen, le nombre d'observations.  ``best_value`` marquée
  ★ vert, ``worst_value`` marquée ▼ rouge.  Adaptive : ``""``
  si ``analysis`` est ``None`` ou ``per_value`` vide.  Anti-
  injection systématique sur la valeur du slot et sur le nom
  du slot variant.

  **Pas de tests statistiques recalculés** : la sortie agrège
  les données nécessaires pour qu'un test externe (Friedman/
  Nemenyi déjà dans `core/statistics.py` Sprint 18) puisse
  les consommer.  Le module ne reconstruit pas ce qui existe.

  +9 clés i18n FR/EN (`incr_*`).  +20 tests dans
  `test_sprint96_incremental_comparison.py` (cas standard 4×2
  → effet du LLM avec gpt rang 1.0 systématique, rang moyen
  correct, best/worst identifiés, ``higher_is_better`` inverse
  l'ordre, lt 2 → None, slot inconnu → None, schémas
  incompatibles ignorés sans crash, acceptation de dicts,
  ex aequo → rangs moyens 1.5, vue HTML adaptive + tri par
  rang + marqueurs ★/▼ + anti-injection sur valeur ET sur
  nom de slot + EN, **cas réaliste 5 OCR × 2 LLM** prouvant
  que mistral domine systématiquement et gpt-4o aussi,
  PipelineRun.as_dict + immutable, complétude i18n 9 clés).
  **Verrou levé** : un benchmark d'axe B avec dizaines de
  pipelines voit immédiatement *« en variant le LLM, gpt-4o
  domine sur 100 % des configurations OCR (rang moyen 1.0) »*
  sans avoir à parcourir les 180 lignes de comparaison brute.

- **Sprint 95 — B.4 : visualisation DAG d'un pipeline composé
  (rendu SVG server-side).**  Outil d'**inspection**, pas de
  construction — le YAML reste source de vérité.  Permet
  d'auditer rapidement la qualité d'une pipeline d'axe B
  (Sprint 63+).  Nouveau module
  `picarones/report/pipeline_dag_render.py` :
  `build_pipeline_dag_html(nodes, labels, edges=None,
  thresholds=(0.05, 0.15), higher_is_better=False)` rend un
  graphe orienté gauche → droite en SVG natif (pas de
  bibliothèque, pas de JS).  Chaque nœud est un rectangle
  annoté du nom du module + types d'entrée/sortie.  Chaque
  arête est une flèche colorée vert/orange/rouge selon la
  valeur de la métrique calculée à la jonction, avec
  étiquette ``type d'artefact`` + ``métrique : valeur``
  (formatée en pourcent ou décimal).  Légende intégrée avec
  les seuils.  Mode ``higher_is_better=True`` inverse la
  sémantique pour les métriques type F1/recall.  Adaptive :
  ``""`` si moins d'un nœud.  Auto-déduction des arêtes
  séquentielles si non fournies.  Anti-injection systématique
  via ``html.escape`` sur le nom du nœud, le type d'artefact,
  le nom de métrique et les listes input/output_types.

  **Pas de drag-and-drop, pas de notebook, pas de drill-down
  par document** : le visuel sert à inspecter et déboguer,
  pas à construire.  Une institution sérieuse versionne ses
  pipelines en YAML dans Git, pas en JSON exporté d'une UI.
  Le drill-down par document reste sur le tableau de
  ``error_absorption`` (Sprint 94) qui montre déjà les tokens
  corrigés / introduits par jonction.

  +6 clés i18n FR/EN (`dag_*`).  +18 tests dans
  `test_sprint95_pipeline_dag.py` (vide → "", single node sans
  flèche, 2 nœuds 1 arête avec étiquettes + valeur formatée
  4.0%, chaîne 3 nœuds 2 flèches, auto-déduction d'arêtes,
  3 cas de couleur (vert ≤ 0.05, jaune ≤ 0.15, rouge > 0.15),
  inversion higher_is_better avec F1=0.96 → vert, nœud
  inconnu dans une arête skipped, valeur de métrique absente
  affichée comme — ; anti-injection 4 vecteurs : nom de nœud,
  artifact_type, metric_name, input/output types ; rendu en
  anglais ; complétude i18n 6 clés).  **Verrou levé** : un
  benchmark d'axe B avec 3+ étapes (par ex. OCR → LLM →
  ALTO_mapper) voit immédiatement à quelle jonction la
  qualité décroche, sans avoir à parcourir un tableau de
  métriques.

- **Sprint 94 — B.3 : métrique d'absorption d'erreur (couche
  calcul + vue HTML).**  Quand un module post-correction LLM
  aplatit les différences entre OCR amont, ce n'est pas qu'il
  *« améliore »* tous les moteurs — c'est qu'il introduit ses
  propres biais qui dominent ceux de l'OCR.  Mesurer la
  dégradation par étape ne suffit pas : il faut **séparer**
  les deux flux à chaque jonction.

  Nouveau module `picarones/core/error_absorption.py` :

  - `compute_error_absorption(reference, before, after,
    case_sensitive=False)` — alignement multi-set token-level
    sur whitespace ; calcule `errors_before`, `errors_after`,
    `corrected = errors_before \\ errors_after`,
    `introduced = errors_after \\ errors_before`,
    `kept_wrong`, `correction_rate` (=
    `n_corrected / n_errors_before` ou `None` si zéro erreur
    avant), `introduction_rate` (= `n_introduced /
    n_errors_after` ou `None`), `net_improvement`,
    `corrected_tokens` et `introduced_tokens` (casse GT
    préservée à l'affichage).  `None` si la GT est vide.

  - `aggregate_error_absorption(per_doc, sample_tokens=50)` —
    somme corpus-wide des compteurs et recalcul *micro* des
    taux ; cap des échantillons de tokens pour ne pas exploser
    le JSON.

  Généralisation du score de sur-normalisation (chantier
  A.I.7) à toute jonction : la formule s'applique uniformément
  à OCR→LLM, OCR→reconstructor, VLM→ALTO_mapper.  Le module
  ne classe pas les erreurs (visuelles, abréviations…) — c'est
  une métrique d'**absorption de volume**, pas de qualité
  éditoriale ; la qualité reste dans `taxonomy` (Sprint 5).

  Nouveau module `picarones/report/error_absorption_render.py`
  : `build_error_absorption_html(junctions, labels,
  sample_max=8)` produit un tableau résumé des jonctions du
  pipeline ; chaque ligne montre erreurs avant/après,
  corrigées (gradient vert), introduites (gradient rouge),
  taux corrigées (gradient rouge → vert), taux introduites
  (gradient vert → rouge), amélioration nette colorée selon
  signe et magnitude, échantillon des tokens introduits (cap).
  Adaptive : `""` si la liste est vide.  Module pur —
  l'utilisateur compose la liste `junctions` depuis son
  `PipelineBenchmarkResult` (Sprint 64).  Visualisation Sankey
  reportée à un sprint dédié (rendu SVG complexe, le tableau
  livre l'information de fond).

  +11 clés i18n FR/EN (`absorption_*`).  +20 tests dans
  `test_sprint94_error_absorption.py` (identité no errors,
  perfect correction, pure introduction, mix correction +
  introduction avec **cas réaliste maistre Pierre du Bois →
  maître Pierre du Bois** prouvant qu'une jonction peut
  corriger ET introduire en parallèle, GT vide → None,
  case-insensitive par défaut + opt-in case-sensitive,
  multiplicité respectée, agrégation micro-rate + skip None +
  cap sample, vue HTML 4 cas dont anti-injection sur
  junction_name + échantillon introduits + FR + EN,
  complétude i18n 11 clés).  **Verrou levé** : un benchmark
  de pipeline composée peut désormais distinguer un module
  qui *corrige* d'un module qui *absorbe* — *« le LLM
  postcorr corrige 65 % des erreurs OCR mais introduit
  12 % de nouvelles erreurs (dont des modernisations
  systématiques de maistre/nostre/veoir) »*.  Sans cette
  métrique, on confondait correction et écrasement, et la
  communauté scientifique ne pouvait pas faire confiance aux
  conclusions sur les pipelines post-correction.

- **Sprint 93 — A.II.7 : métriques d'image prédictives (couche
  calcul + vue HTML).**  ``image_quality.py`` (Sprint 5)
  mesurait des features indépendamment ; ce module les
  **combine** en deux indicateurs corpus-level qui répondent
  à des questions de diagnostic distinctes.

  - `picarones/core/image_predictive.py` :
    `compute_paleographic_complexity(quality, weights=None)`
    retourne ``{score ∈ [0,1], components, weights_used}`` —
    combinaison pondérée éditoriale du bruit (0,30), du flou
    `1 - sharpness` (0,30), du faible contraste
    `1 - contrast` (0,20) et de la rotation
    `|degrees| / 30` (0,20).  Bornes [0, 1] forcées par
    clamping.  Poids surchargeables.  Garde-fous : `None` si
    quality vide ou poids tous nuls.
    `compute_corpus_homogeneity(image_qualities)` retourne
    ``{score ∈ [0,1], n_docs, per_feature{mean, stdev,
    normalised}}`` — moyenne des écart-types normalisés sur
    4 features (plage 0,5 pour [0,1] et 10° pour rotation).
    0 = corpus uniforme (la moyenne globale est fiable),
    1 = corpus très hétérogène (la moyenne ment).
    `aggregate_corpus_predictive(image_qualities)` synthétise
    complexité (mean/median/min/max/stdev) + homogeneity.

  - `picarones/report/image_predictive_render.py` :
    `build_image_predictive_html(aggregated, labels)` produit
    deux blocs : tableau résumé complexité (mean coloré
    gradient vert → rouge, median, min, max, stdev, n_docs) +
    tableau homogénéité (score coloré + détail par feature
    avec mean, stdev, contribution normalisée colorée).
    Adaptive : `""` si pas de données.  Module pur —
    l'utilisateur compose
    `[doc.image_quality.as_dict() for ...]` →
    `aggregate_corpus_predictive` → `build_image_predictive_html`.

  - **Pas de prédiction CER absolue** : on ne prétend pas
    fournir une valeur CER en pourcentage (demanderait un
    modèle entraîné par moteur, contraire à la philosophie
    banc d'essai).  Le score est relatif, pour une lecture
    diagnostique : *« le doc A est ~3× plus complexe que le
    doc B, ce qui est cohérent avec le CER observé »*.

  +20 clés i18n FR/EN (`imgpred_*`).  +21 tests dans
  `test_sprint93_image_predictive.py` (cas trivial → ≈0, cas
  extrême → ≈1, bornes [0,1] respectées sur valeurs hors
  plage, components retournés, poids custom (tout sur le
  bruit → score = noise_level), poids défaut sommant à 1,
  None sur empty et poids nuls ; corpus uniforme → 0,
  hétérogène → > 0.5, lt 2 docs → None, per_feature
  structurée ; **cas réaliste BnF** mix trivial/difficile,
  empty, single doc no homogeneity ; vue HTML 4 cas dont
  anti-injection sur titre custom + FR + EN ; complétude i18n
  19 clés).  **Verrou levé** : un benchmark BnF voit désormais
  *« corpus-wide complexity 0,42 (modérée), homogeneity 0,18
  (uniforme — moyenne fiable) »* dans la vue Analyses, ce qui
  permet d'expliquer une partie du CER observé sans tomber
  dans la prédiction prescriptive.

- **Sprint 92 — A.II.9 : métriques longitudinales (régression
  linéaire + change-point + détecteur narratif + vue HTML).**
  L'historique SQLite (`core/history.py`, Sprint 8) collectait
  les résultats sans qu'aucune métrique n'en sorte dans le
  rapport.  Ce sprint exploite la série temporelle des CER
  pour signaler tendances et ruptures — complémentaire à
  A.I.3 (off-baseline) qui dit *« écart anormal sur ce
  corpus »* sans caractériser la dynamique.

  - `picarones/core/longitudinal.py` : `compute_linear_trend`
    régression OLS pure Python sans scipy retourne
    `LinearTrend(slope, intercept, r_squared, n_runs)` ;
    `detect_change_point(series, min_segment_size=3)` balayage
    exhaustif (Pettitt simplifié) retourne
    `ChangePointResult(index, timestamp, mean_before,
    mean_after, delta, n_before, n_after)` ;
    `compute_engine_longitudinal(history, engine, corpus)`
    combine les deux avec garde-fou `min_runs_for_trend=3` et
    seuil `change_point_threshold=0.01` (1 point CER) pour
    filtrer le bruit ; `compute_corpus_longitudinal` agrège
    sur tous les moteurs présents.

  - Nouveau `FactType.REGRESSION_IN_HISTORY` (priority 170,
    importance MEDIUM par défaut, HIGH si `|absolute_delta| ≥
    0.05`) + détecteur `detect_regression_in_history` qui lit
    `benchmark_data["longitudinal_trends"]`.  Déclenche si
    pente > +1 pt CER/an **ou** change-point delta > 1 pt CER.
    Garde-fou `n_runs ≥ 3`.  Le payload trace
    `pattern in {"trend", "change_point",
    "trend_and_change_point"}`.  Templates FR/EN sans chiffres
    en dur.  Ajout aux paires complémentaires de l'arbitre :
    `(GLOBAL_LEADER_CER, REGRESSION_IN_HISTORY)` (le leader
    peut être en régression, info critique) et
    `(ENGINE_OFF_BASELINE, REGRESSION_IN_HISTORY)` (les deux
    se complètent : écart anormal vs tendance dans le temps).

  - `picarones/report/longitudinal_render.py` :
    `build_longitudinal_html(trends, labels)` rend un tableau
    moteur × {n_runs, premier CER, dernier CER, Δ cumulé
    coloré (gradient vert → orange → rouge sur ±5 pts ; bleu
    si amélioration), pente annualisée, R², point de rupture
    avec timestamp + delta entre parenthèses}.  Tri par Δ
    décroissant.  Adaptive : `""` si pas de données.  Module
    pur — l'utilisateur compose
    `BenchmarkHistory.list_entries()` →
    `compute_corpus_longitudinal` →
    `build_longitudinal_html`.

  +10 clés i18n FR/EN (`longitudinal_*`).  +28 tests dans
  `test_sprint92_longitudinal.py` (régression OLS pente + R² +
  série plate + lt 2 + même timestamp ; change-point delta
  exact + lt segments + uniforme ; intégration entries +
  filtre corpus + min_runs + threshold ; multi-moteurs ;
  détecteur 6 cas dont silence sans data, silence si plat,
  HIGH si Δ ≥ 5 pts, change-point seul, garde-fou n_runs < 3 ;
  **traçabilité anti-hallucination FR + EN** sur les sentences
  de `build_synthesis` ; vue HTML 4 cas dont anti-injection,
  complétude i18n 10 clés).  **Verrou levé** : un benchmark
  qui pousse ses résultats dans l'historique voit désormais
  *« sur les 8 runs historiques pour tess, le CER moyen est
  passé de 4 % à 7 % (variation cumulée 3 points) »* dans la
  synthèse + le tableau d'évolution dans la vue.  Permet de
  relier une régression à un changement de pipeline.

- **Sprint 91 — A.II.6 : métriques économiques (throughput
  effectif + coût marginal par erreur évitée).**  Le throughput
  brut (pages/heure d'OCR pur) ment quand un moteur est rapide
  mais imprécis : la correction humaine *post hoc* absorbe le
  gain.  Cette métrique discrimine fortement entre un cloud
  rapide à 30 % de timeouts et un local lent à 100 % de
  fiabilité.  Couplée au coût marginal par erreur évitée, elle
  arme une décision business honnête.

  - `picarones/core/throughput.py` :
    `compute_effective_throughput(n_pages, duration_seconds,
    n_errors, time_per_error_seconds=5.0)` retourne
    `{n_pages, duration_seconds, n_errors,
    time_per_error_seconds, correction_time_seconds,
    total_seconds, pages_per_hour_raw,
    pages_per_hour_effective, drag_ratio}`.  Constante
    HTR-United (5 s/erreur) surchargeable.  Garde-fous : `None`
    si `n_pages = 0` ou `total_seconds = 0`, `ValueError` sur
    valeurs négatives.  `aggregate_effective_throughput` agrège
    par moteur sur le corpus.

  - `picarones/core/marginal_cost.py` :
    `compute_marginal_cost(cost_a, errors_a, cost_b, errors_b)`
    retourne `{cost_per_avoided_error, n_errors_avoided,
    cost_delta, dominated}` ou `None` si `errors_b ≥ errors_a`
    (pas de gain à mesurer).  `dominated=True` quand B est moins
    cher ET plus précis (cas idéal Pareto).
    `compute_marginal_cost_matrix(per_engine)` retourne toutes
    les paires ordonnées (A → B) où B fait moins d'erreurs,
    triées par coût marginal croissant.

  - `picarones/report/throughput_render.py` :
    `build_throughput_html(aggregated, labels)` produit un
    tableau résumé moteur × {pages/h brut, pages/h **utilisable**
    (gradient rouge → vert sur le max observé), % drag (gradient
    vert → rouge), pages, erreurs}.  Tri par pages/h utilisable
    décroissant.  Adaptive : `""` si pas de données.  Module
    pur — l'utilisateur compose la liste `per_engine` depuis ses
    `EngineReport` (calcul `n_errors` au choix : WER × n_words,
    CER × n_chars, etc.).  Vue HTML pour le coût marginal sera
    couplée à la vue Pareto dans un sprint ultérieur.

  +9 clés i18n FR/EN (`throughput_*`).  +27 tests dans
  `test_sprint91_throughput.py` (formule effective avec/sans
  erreurs, custom time_per_error, garde-fous n_pages=0 +
  total_seconds=0 + ValueError sur négatifs, drag_ratio élevé,
  agrégation 3 cas, marginal cost standard + dominé + B pire +
  errors égales + invalide, matrice tri ascendant + lt 2 +
  données invalides, **cas réaliste BnF** Tesseract local 600
  p/h brut → 423 p/h effectif vs GPT-4o cloud 1800 p/h brut →
  300 p/h effectif, vue HTML 4 cas dont anti-injection + tri
  descendant, complétude i18n 9 clés).  **Verrou levé** : un
  archiviste BnF qui pondère un budget contre une exigence de
  délai voit immédiatement *« Tesseract local 423 p/h
  utilisable, GPT-4o cloud 300 p/h utilisable malgré son
  apparente vitesse de 1800 p/h brut »* — la décision business
  s'aligne sur la réalité opérationnelle.

- **Sprint 90 — A.II.4 finition : détecteur narratif
  `engine_unstable` + vue HTML stabilité multi-runs.**  Le
  module `picarones/core/reliability.py` (Sprint 83) livrait
  la couche de calcul ; aucun détecteur ni vue ne consommaient
  les données.  Ce sprint complète A.II.4 sur les moteurs LLM/
  VLM dont les sorties varient entre runs successifs sur les
  mêmes documents — situation critique pour la
  reproductibilité scientifique d'une publication.  Nouveau
  `FactType.ENGINE_UNSTABLE` (priority 160, importance HIGH)
  + détecteur `detect_engine_unstable` qui lit
  `benchmark_data["multirun_stability"]` (liste enrichie
  d'`engine_name` + sortie de `compute_multirun_stability`).
  Garde-fous : `n_runs ≥ 2`, déclenche si `cer_cv > 0.10`
  **ou** `identical_run_rate < 0.50`.  Templates FR/EN sans
  chiffres en dur.  Ajout du couple
  `(GLOBAL_LEADER_CER, ENGINE_UNSTABLE)` à
  `_COMPLEMENTARY_PAIRS` de l'arbitre — un moteur peut être
  leader **et** instable, et c'est précisément l'information
  critique à remonter ensemble.  Nouveau module
  `picarones/report/multirun_stability_render.py` :
  `build_multirun_stability_html(stability, labels)` rend un
  tableau moteur × {n_runs, CER moyen ± σ, CV (gradient vert
  → orange → rouge sur 0–25 %), % runs identiques, sorties
  distinctes}.  Adaptive : `""` si la liste est vide ou que
  tous les `cer_cv` sont `None`.  Note d'intégration : la
  vue est un module pur (l'utilisateur exécute lui-même les
  N runs et appelle `compute_multirun_stability` ; option
  runner `--repeats N` reportée à un sprint dédié).  +8 clés
  i18n FR/EN (`stability_*`).  +18 tests dans
  `test_sprint90_engine_unstable.py` (FactType + ajout
  arbiter, détecteur 6 cas dont silence sans data, silence
  stable, HIGH si CV ≥ 10 %, HIGH si runs divergent, garde-
  fou n_runs < 2, garde-fou engine manquant, multi-engines,
  **traçabilité anti-hallucination FR + EN** prouvant que
  chaque chiffre de la phrase rendue par
  `build_synthesis(...)["sentences"]` est dans le payload du
  Fact, vue HTML 4 cas dont anti-injection nom moteur,
  complétude i18n 8 clés).  **Verrou levé** : un papier
  scientifique qui rapporte un CER LLM voit désormais
  immédiatement *« sur 4 runs successifs, gpt-4o produit des
  sorties variables (CV 24,3 %) — interpréter avec
  prudence »* dans la synthèse + le tableau de stabilité dans
  la vue.

- **Sprint 89 — A.II.8b : score de spécialisation inter-moteurs
  (couche calcul + vue HTML).**  La matrice de divergence
  taxonomique (Sprint 35) répondait à *« à quel point ces
  moteurs se trompent-ils différemment ? »* ; ce sprint
  transforme cette information en un score lisible et un
  **top-N des paires les plus spécialisées**, qui répond
  directement à la question *« quels moteurs sont des candidats
  pour un voting ensemble ? »*.  Le module **ne recommande
  pas** d'ensemble — il livre l'observation factuelle et
  laisse le chercheur arbitrer.  Nouveau module
  `picarones/core/specialization.py` :
  `compute_specialization_score(taxonomy_a, taxonomy_b)`
  retourne un score normalisé ∈ [0, 1] (délégué à
  `inter_engine.jensen_shannon_divergence` Sprint 35, pas de
  double calcul) ;
  `classify_specialization(score, thresholds=DEFAULT_THRESHOLDS)`
  classe en `similar` (< 0,10) / `distinct` (0,10–0,30) /
  `highly_specialized` (≥ 0,30) — seuils éditoriaux pas
  verdict, surchargeables ;
  `compute_specialization_matrix(taxonomies)` retourne une
  matrice symétrique avec `max_pair` ;
  `top_specialized_pairs(matrix, n=5, min_score=0)` retourne
  les paires triées par score décroissant avec leur catégorie.
  Nouveau module `picarones/report/specialization_render.py` :
  `build_specialization_html(taxonomies, labels, top_n=5)`
  rend un tableau Moteur A × Moteur B × Score (gradient blanc
  → bleu profond) × Lecture (libellé i18n).  Adaptive : `""`
  si moins de 2 moteurs avec taxonomie.  Anti-injection.
  Câblage générator : lit les `aggregated_taxonomy` exposés
  sur les moteurs (Sprint 5/runner historique), construit la
  map `{engine: counts}` et passe au renderer.  Insertion dans
  `view_analyses.html` derrière la lisibilité.  +9 clés i18n
  FR/EN (`specialization_*`).  +24 tests dans
  `test_sprint89_specialization.py` (score symétrique +
  identité 0 + disjoint 1 + bornes [0,1], classify 5 cas dont
  custom thresholds, matrice diagonale 0 + symétrique +
  max_pair correctement identifié, top_pairs tri/n/min_score/
  None, rendu adaptive + anti-injection + FR/EN, complétude
  i18n 9 clés).  **Verrou levé** : un benchmark BnF avec ≥ 2
  moteurs voit immédiatement *« tess et pero ont une
  spécialisation forte (0,489) — ils font des erreurs de
  natures différentes »* — observation factuelle, le
  chercheur arbitre.

- **Sprint 88 — A.I.8 vue HTML : déficit projeté de robustesse
  (clôture A.I.8 bout-en-bout).**  Le module
  `picarones/core/robustness_projection.py` (Sprint 81)
  calculait la projection des courbes de dégradation
  synthétique sur les caractéristiques d'image réelles ; ce
  sprint livre la **vue HTML** correspondante.  La robustesse
  étant un workflow CLI séparé (`picarones robustness`) et non
  intégré au benchmark principal, ce sprint livre un **module
  de rendu pur** que l'utilisateur compose lui-même
  (`analyze_robustness` → `project_robustness_on_corpus` →
  `aggregate_projection_per_engine` →
  `build_robustness_projection_html`).  Nouveau module
  `picarones/report/robustness_projection_render.py` :
  `build_robustness_projection_html(projection, aggregated,
  labels)` produit deux tableaux :

  1. **Résumé par moteur** — déficit total attendu (gradient
     vert → orange → rouge sur ±5 pts de CER), nombre de types
     de dégradation évalués, pire dégradation avec sa
     contribution.  Trié par déficit décroissant.
  2. **Détail (moteur × dégradation)** — docs, docs avec data,
     déficit projeté coloré, docs au-dessus du seuil critique.

  Si `aggregated` n'est pas fourni, calculé automatiquement
  depuis la projection.  Adaptive : `""` si la projection est
  vide.  Anti-injection systématique sur nom de moteur et type
  de dégradation.  Note explicite que la sommation suppose
  l'indépendance des dégradations *« approximation utile pour
  le diagnostic, pas un verdict »*.  +13 clés i18n FR/EN
  (`robproj_*`).  +12 tests dans
  `test_sprint88_robustness_projection_html.py` couvrant rendu
  vide/None, rendu complet, calcul automatique de
  l'agrégation, tri par déficit décroissant, formatage de la
  cellule « pire dégradation », gestion d'un déficit None
  (cellule —), anti-injection nom moteur + type dégradation,
  rendu en français + anglais, **bout-en-bout** avec le
  pipeline réel `project_robustness_on_corpus` +
  `aggregate_projection_per_engine`, complétude i18n 13 clés.
  **Verrou levé** : un benchmark BnF qui veut savoir *« mon
  corpus de notaires XVIIᵉ siècle est-il à risque face à mon
  moteur OCR ? »* obtient un tableau lisible directement
  intégrable dans le rapport — A.I.8 livrée bout-en-bout
  (calcul Sprint 81 + vue HTML Sprint 88).

- **Sprint 87 — A.II.2 : delta Flesch câblé bout-en-bout
  (couche calcul Sprint 52 + runner + vue HTML).**  Le module
  `picarones/core/readability.py` (Sprint 52) calculait le
  delta Flesch *« over-normalisation par LLM »* — ce sprint le
  remonte automatiquement dans le rapport.  Nouveau helper
  `picarones/core/readability_runner.py` :
  `compute_readability_metrics(reference, hypothesis, lang)`
  avec **adaptive masking** (≥ 5 mots GT pour éviter
  l'instabilité de Flesch sur très courts textes) ;
  `aggregate_readability_metrics(per_doc)` retourne
  `{lang, n_docs, n_docs_with_delta, delta_mean, delta_median,
  delta_min, delta_max, n_over_normalized, n_under_normalized,
  over_normalized_rate}` — l'over-normalisation est définie à
  Δ > +5 points (LLM modernise un texte ancien), l'under-
  normalisation à Δ < -5 (dégradation OCR brutale).
  `DocumentResult.readability_metrics` et
  `EngineReport.aggregated_readability` (sérialisation
  conditionnelle, libérés par `compact`).  Câblage runner :
  langue lue depuis `corpus.metadata.get("language", "fr")`,
  fallback `fr` avec warning si valeur non `fr`/`en`,
  paramètre `corpus_lang` propagé jusqu'aux workers IO et CPU
  (workers acceptent maintenant 7 ou 8 args en mode legacy
  pour rétrocompat).  Erreur isolée par try/except + warning
  explicite.  Nouveau module
  `picarones/report/readability_render.py` :
  `build_readability_summary_html` rend un tableau résumé
  moteur × {Δ moyen coloré (vert au centre, orange si over-
  norm, bleu si under-norm), Δ médian, % over-normalisés,
  docs under-normalisés, docs} ; saturation à ±15 points.
  Insertion dans `view_analyses.html` derrière les blocs
  A.II.5.  Anti-injection systématique.  +8 clés i18n FR/EN.
  +20 tests dans `test_sprint87_readability_html.py`
  (adaptive masking GT < 5 mots, langue passée à fr/en,
  hypothèse vide → flesch_delta None mais flesch_reference
  conservé, agrégation moyenne + over-norm rate, sérialisation
  `DocumentResult`/`EngineReport`, `compact`, masquage
  adaptatif HTML, rendu FR + EN, anti-injection sur nom
  moteur, complétude i18n 8 clés).  **Verrou levé** : le
  rapport remonte désormais *« GPT-4o : Δ moyen +11,5,
  85 % des docs over-normalisés »* directement dans la vue
  Analyses — pas de visualisation HTML pour les VLM
  hallucinant du français moderne sur du français médiéval
  jusqu'ici, c'est livré.  Reste pour A.II.2 bout-en-bout :
  reading_order_f1 et layout_f1 (Sprints 53-54) qui requièrent
  un moteur produisant PAGE/ALTO et seront câblés via les
  pipelines composées (axe B).

- **Sprint 86 — A.II.5 : câblage runner + vues HTML (clôture
  bout-en-bout).**  Suite directe Sprints 84 et 85 — la couche
  de calcul livrait deux modules pour le mode plein-texte
  patrimonial, ce sprint les remonte automatiquement dans le
  rapport.  Deux nouveaux helpers
  `picarones/core/searchability_runner.py` et
  `picarones/core/numerical_sequences_runner.py` qui calculent
  les métriques par document avec **adaptive masking** (rien
  n'apparaît pour un doc sans GT exploitable) et agrègent
  corpus-wide en *micro*-rappel pour la searchability et en
  somme de compteurs par catégorie pour les séquences
  numériques.  `DocumentResult` gagne `searchability_metrics`
  et `numerical_sequence_metrics` ; `EngineReport` gagne
  `aggregated_searchability` et `aggregated_numerical_sequences`
  (sérialisation conditionnelle dans `as_dict`, libérés par
  `compact`).  Le runner historique calcule désormais les deux
  inconditionnellement (coût négligeable face à l'OCR), erreur
  d'un module isolée par try/except + warning explicite,
  rétrocompat stricte (aucun champ ajouté au JSON quand le
  corpus est sans signal).  Deux nouveaux modules de rendu
  `picarones/report/searchability_render.py` et
  `picarones/report/numerical_sequences_render.py` :
  `build_searchability_summary_html` produit un tableau résumé
  moteur × (rappel coloré gradient rouge → jaune → vert,
  retrouvés/total, docs) ;
  `build_numerical_sequences_html` produit un tableau moteur ×
  catégorie (year/roman/foliation/currency/regnal) avec
  **adaptive masking par catégorie** (une catégorie sans signal
  est omise pour tous les moteurs) ; chaque cellule affiche le
  score strict (gradient) + la valeur entre parenthèses + le
  n.  Insertion dans `view_analyses.html` derrière le profil
  philologique, `chart-card` pleine largeur conditionné.
  Anti-injection systématique (`html.escape`).  +15 nouvelles
  clés i18n FR/EN (`search_*`, `numseq_*`).  +25 tests dans
  `test_sprint86_aii5_html.py` couvrant adaptive masking sur
  les helpers, agrégation micro-rappel, somme par catégorie,
  sérialisation `DocumentResult`/`EngineReport`,
  `compact` qui efface bien les champs, masquage adaptatif HTML
  (vide quand sans signal, omission de catégories), rendu en
  FR + EN, anti-injection sur nom de moteur, complétude i18n
  sur 15 clés.  **Verrou levé** : un benchmark BnF voit
  désormais sur la vue Analyses *« Recherchabilité fuzzy :
  tess 95,2 %, pero 87,8 % »* + le tableau séquences
  numériques détaillé par catégorie — A.II.5 est livrée
  bout-en-bout en couche calcul (Sprints 84-85), runner et
  HTML (Sprint 86).

- **Sprint 85 — A.II.5b : précision sur séquences numériques
  (couche de calcul + registre typé).**  Pour un économiste-
  historien, un éditeur de chartes ou un archiviste, la
  fidélité aux séquences numériques est un proxy direct de la
  qualité éditoriale.  Un OCR qui rate *« 1789 »* dans une
  charte révolutionnaire ou *« f. 12v »* dans une cote
  d'archives produit un corpus inutilisable, même si le CER
  global est respectable.  Nouveau module
  `picarones/core/numerical_sequences.py` couvrant 5 catégories :

  - **Dates arabes** : années 4 chiffres dans la plage
    [1000-2099] (détection conservatrice pour éviter les
    faux positifs sur volumes/numéros).
  - **Numéraux romains** : réutilise
    `picarones.core.roman_numerals.detect_roman_numerals`
    (Sprint 60), `min_length=2`.
  - **Foliotation** : `f.`, `fol.`, `p.`, `pp.`, `n°` avec
    suffixe `r`/`v` préservé (recto/verso = information
    distincte, **non interchangeable** côté valeur).
  - **Montants** : Ancien Régime (`livres`/`l.`,
    `sols`/`s.`, `deniers`/`d.`) et modernes (`£`, `€`, `₣`,
    `écus`, `florins`, `francs`).
  - **Années régnales** : `an III`, `l'an V`, `an de grâce
    1450`, `an de la République`.

  Pour chaque GT, classification en 3 statuts :
  `strict_preserved` (forme exacte), `value_preserved` (la
  valeur apparaît même si la forme diffère, `XIV` ↔ `14`
  pour les romains ; **mais pas** `f. 12r` ↔ `f. 12v` car
  recto/verso est une distinction substantielle), `lost`.
  `compute_numerical_sequence_metrics` retourne
  `{global_strict_score, global_value_score, n_total,
  per_category{n_total, strict, value, strict_score,
  value_score, lost_items}}`.  Multiplicité respectée (un
  item hyp ne peut servir qu'à un seul match).
  `numerical_sequence_strict_score` et
  `numerical_sequence_value_score` enregistrés dans le
  registre typé Sprint 34 pour `(TEXT, TEXT)`.  Limites
  documentées : regex conservatrices (`mil cinq cens` non
  détecté comme année), pas de cross-category match
  (`MDCLXVIII` GT et `1668` hyp sont catégorisés
  séparément).  +27 tests dans
  `test_sprint85_numerical_sequences.py` couvrant détecteurs
  individuels (year/roman/foliation/currency/regnal),
  scénarios identité/perte totale/GT vide/recto-verso non
  interchangeables/multiplicité, **2 cas réalistes** (charte
  XVIIIᵉ siècle préservée intégralement vs registre paroissial
  où l'OCR modernise XVIII→18 mais préserve l'année 1750 et
  la foliation), intégration registre 4 cas dont
  `compute_at_junction`.  **Verrou levé** : un bench
  d'archive numérique peut classer ses moteurs sur la
  dimension *« mes dates et cotes seront-elles fiables ? »*,
  qui complète la **recherchabilité fuzzy** (Sprint 84) pour
  livrer A.II.5 en couche de calcul intégrale.

- **Sprint 84 — A.II.5 : recherchabilité fuzzy (couche de
  calcul + métrique enregistrée).**  Le CER mesure les erreurs
  caractère par caractère ; pour un usage *recherche
  plein-texte* (Elastic, Solr en mode fuzzy, full-text de
  Gallica), la question réelle est : *« combien de mots GT
  sont retrouvables dans la sortie OCR à orthographe approchée
  près ? »*.  Un CER de 8 % peut donner 95 % de findability si
  les erreurs sont concentrées sur des caractères non
  significatifs ; à l'inverse, 4 % de CER mais distribué sur
  tous les noms propres rend le corpus inutilisable pour
  l'indexation prosopographique.  Nouveau module
  `picarones/core/searchability.py` : `levenshtein_distance(a,
  b)` (DP O(|a|·|b|), mémoire O(min(|a|,|b|)));
  `compute_searchability(reference, hypothesis,
  max_distance=2, case_sensitive=False)` aligne par multi-set
  (un token hyp utilisé une seule fois, comme
  rare_token_recall Sprint 71), retourne `{n_gt_tokens,
  n_searchable, recall, missed_tokens, max_distance}` avec
  `recall=None` quand n_gt=0 (différencie GT vide de aucun
  match), court-circuit longueur (Levenshtein ≥ |Δlen|) et
  arrêt précoce sur match exact.  `searchability_recall_metric`
  enregistré dans le registre typé Sprint 34 pour la jonction
  `(TEXT, TEXT)` (convention float : 0.0 si GT vide).  Tableau
  Elastic ``fuzziness: AUTO`` (≤ 2) en défaut, paramétrable.
  Limites documentées : tokenisation par split whitespace ;
  Levenshtein non pondéré ; pas de sémantique (BERTScore
  reporté).  +28 tests dans `test_sprint84_searchability.py`
  (Levenshtein 9 cas dont identité/insertion/suppression/
  substitution/disjoint/empty/kitten classique, computation
  13 cas dont identité, complètement différent, GT vide
  (recall None), hypothèse vide (recall 0), max_distance=0
  exact, max_distance=2 swap, max_distance large, casse
  insensible, casse sensible opt-in, multiplicité,
  missed_tokens préserve casse GT, ValueError sur
  max_distance négatif, deux **cas réalistes opposés**
  (« Charles → Charlemagne » non retrouvé vs « maistre →
  maitre » retrouvé), intégration registre 4 cas dont
  `compute_at_junction`).  **Verrou levé** : un bench BnF
  d'archive numérique peut désormais classer ses moteurs sur
  la dimension *« mes corpus seront-ils retrouvables après
  OCRisation ? »* — proxy direct de la valeur d'usage.

- **Sprint 83 — A.II.4 : métriques de fiabilité (couche de
  calcul).**  Premier sprint de l'Étape 4 du plan d'évolution
  2026 après la clôture de A.I.  Une publication scientifique
  qui rapporte un CER LLM sans stabilité est méthodologiquement
  faible ; un benchmark qui ignore le plafond humain (« deux
  paléographes ne sont pas même d'accord ») crée des
  classements faussement optimistes.  Nouveau module
  `picarones/core/reliability.py` couvrant deux familles :

  - **Inter-annotator agreement (IAA) au niveau caractère.**
    `cohen_kappa(annotations_a, annotations_b)` : κ standard
    avec gestion des cas dégénérés (tailles incompatibles →
    `None`, séquences vides → `None`, un seul label →
    convention 1.0/0.0 documentée car κ mathématiquement
    indéfini quand pe = 1).  `krippendorff_alpha(units)` : α
    de Krippendorff en mode nominal, généralisé à N
    annotateurs avec missing values autorisées (cellules
    `None`), formule `1 - D_o / D_e` avec `D_e` calculé sur
    les paires sans remise.  `compute_iaa(transcription_a,
    transcription_b)` : aligne deux GT caractère par
    caractère via `_aligned_char_pairs` (segments `equal` et
    `replace` de `SequenceMatcher`, les `insert`/`delete`
    n'ayant pas d'alignement bilatéral exploitable) puis
    calcule κ et α sur les paires alignées + agreement_rate
    + n_aligned_chars.

  - **Stabilité multi-runs.**  `compute_multirun_stability(runs,
    reference=None)` mesure la variance d'une pipeline
    LLM/VLM non-déterministe relancée N fois sur le même
    document : pairwise_disagreement (Jaccard token-level)
    moyen et max, identical_run_rate, n_distinct_outputs.  Si
    `reference` fournie, on calcule `cer_per_run`,
    `cer_mean`, `cer_stdev`, `cer_cv` (coefficient de
    variation, `None` quand mean=0 pour éviter la division
    par zéro).  Retourne `None` si moins de 2 runs.

  Périmètre Sprint 83 : **couche de calcul uniquement**.
  L'extension du loader pour accepter `doc_001.gt.A.txt` et
  `doc_001.gt.B.txt` comme GT multiples, l'option
  `--repeats N` du runner et le détecteur narratif
  `engine_unstable` arriveront dans des sprints suivants.
  +26 tests dans `test_sprint83_reliability.py` (cohen_kappa
  6 cas dont accord parfait/désaccord pire que hasard/un seul
  label, krippendorff_alpha 5 cas, compute_iaa 5 cas dont
  empty/one-empty, compute_multirun_stability 6 cas dont
  reference parfaite/CV indéfini, _aligned_char_pairs 4 cas).
  **Verrou levé** : le rapport pourra demain afficher le
  plafond humain à côté du CER (« CER de Pero 4,2 % approche
  le κ inter-paléographes 0,89 ») et signaler les pipelines
  LLM dont la variance dépasse un seuil.

- **Sprint 82 — A.I.9 : section « Leviers d'amélioration »
  (couche calcul + cards HTML).**  Le moteur narratif
  (Sprint 19) émet des `Fact` qui décrivent **ce qui s'est
  passé** dans le benchmark.  Ce sprint répond à une question
  complémentaire : *« sur quelle dimension le bénéfice attendu
  d'une amélioration serait-il le plus visible ? »*.  Approche
  strictement **non-prescriptive** : aucune recommandation
  *« faites X »*, uniquement des **observations factuelles**
  agrégées depuis les modules d'analyse (Sprints 75-81).
  Nouveau module `picarones/core/levers.py` : dataclass
  ``Lever(type, importance, payload, engines_involved)``,
  ``LeverImportance`` (HIGH/MEDIUM/LOW), registre via
  décorateur ``@register_lever``, helper ``detect_levers`` qui
  trie par importance décroissante.  **5 détecteurs livrés** :
  ``dominant_recoverable_class`` (≥30 % d'erreurs récupérables
  selon la catégorisation Sprint 77), ``pareto_concentration``
  (top-20 % docs ≥50 % du CER cumulé), ``complementarity_observation``
  (factuel sur ``inter_engine_analysis.complementarity_gap``,
  Sprint 35), ``lexical_modernization_observation`` (top-3
  tokens GT systématiquement modernisés, Sprint 80),
  ``robustness_projection_observation`` (déficit projeté ≥2
  points de CER, Sprint 81).  Nouveau module
  `picarones/report/levers_render.py` : ``build_levers_section_html``
  rend des **cards** server-side avec étiquette i18n + phrase
  factuelle + détail compact + niveau d'importance coloré.
  Adaptive masking : ``""`` si aucun levier exploitable.
  Anti-injection systématique via ``html.escape``.  Garde-fou
  anti-hallucination identique au moteur narratif : chaque
  chiffre rendu est dans le ``payload`` du levier.  +18 clés
  i18n FR/EN (``levers_*``).  +40 tests dans
  `test_sprint82_levers.py` (modèle 3, dominant 6, pareto 5,
  complementarity 4, lexical 4, robustness 4, pipeline 3,
  rendu 6, anti-hallucination FR+EN 3, complétude i18n 2).
  **Verrou levé** : le rapport ne se contente plus de décrire
  *ce qui est* — il propose une lecture compacte des
  **dimensions où un effort éditorial pourrait porter**, sans
  jamais imposer un verdict.

- **Sprint 81 — A.I.8 : robustesse synthétique projetée sur le
  corpus réel (couche de calcul).**  Le module
  ``picarones/core/robustness.py`` (Sprint 8) génère des courbes
  CER vs niveau de dégradation **synthétique** ;
  ``image_quality.py`` mesure le bruit/flou réels du corpus.  Ce
  sprint **projette** les caractéristiques réelles sur les
  courbes synthétiques pour estimer le **déficit attendu de CER**
  sur le corpus dans son état actuel.
  - Nouveau module `picarones/core/robustness_projection.py` :
    - ``_interpolate_cer(levels, cer_values, target_level)``
      interpolation linéaire avec **clip** aux bornes (pas
      d'extrapolation hasardeuse).  Filtre les ``cer_values``
      à ``None``.
    - ``_extract_quality_value(quality_dict, degradation_type,
      custom_mapping)`` extrait la valeur pertinente depuis
      ``ImageQualityResult.as_dict()`` (mapping default :
      noise→noise_level, blur→blur_score, etc.).
    - ``project_robustness_on_corpus(curves, image_qualities,
      quality_to_level, critical_threshold)`` retourne
      ``{engine: {degradation_type: {n_docs, n_docs_with_data,
      expected_cer_mean, expected_cer_median, baseline_cer,
      deficit_vs_baseline, n_docs_above_critical,
      critical_threshold_level, critical_threshold_cer}}}``.
    - ``aggregate_projection_per_engine(projection)`` somme les
      déficits sur tous les types de dégradation et identifie le
      **type le plus pénalisant** (worst_degradation_type).
      Hypothèse d'indépendance des dégradations documentée.
  - +22 tests dans `test_sprint81_robustness_projection.py` :
    interpolation (7 cas — exact, linéaire, clip lower/upper,
    vide, all None, partiel None) ; extraction qualité (4 cas —
    default, unknown, missing, custom) ; projection (7 cas —
    single curve, doc above critical, doc sans data, multi
    moteurs/types, no curves, no docs, threshold override) ;
    agrégation (4 cas — total, worst, None skipped, vide).
  - **Verrou levé** : un benchmark BnF avec
    ``image_quality_aggregated`` peut désormais lire *« 30 %
    de vos documents ont un bruit où Tesseract perd 8 points de
    CER — déficit attendu global 2,4 points »*.  La courbe de
    robustesse n'est plus déconnectée du corpus réel.

- **Sprint 80 — A.I.7 : sur-normalisation lexicale en vue
  analytique dédiée (couche calcul + table HTML).**  Le détecteur
  ``llm_hallucination_flag`` (Sprint 19) signale qu'un moteur
  sur-normalise via un score agrégé.  Mais ce score ne dit rien
  sur **quoi** corriger dans le prompt.  Ce sprint produit une
  **table de fréquences détaillée** par token GT.
  - Nouveau module `picarones/core/lexical_modernization.py` :
    - ``compute_lexical_modernization(reference, hypothesis,
      stop_list, case_sensitive)`` aligne mot-à-mot via
      ``difflib.SequenceMatcher`` et accumule par token GT :
      ``{n_total, n_modernized, rate_modernized, variants}``.
    - ``aggregate_lexical_modernization(per_doc_results)`` somme
      les compteurs corpus-wide.
    - ``top_modernized_tokens(data, n=20, min_total=1)`` retourne
      les N tokens GT les plus modernisés (tri décroissant par
      taux, tie-break par n_total).  Filtre les anecdotiques
      via ``min_total``.
    - Stop-list paramétrable (tokens GT à ignorer même s'ils
      sont modifiés) — par défaut vide, le module ne devine pas
      ce qui est « moderne ».
    - Cas particuliers : token GT supprimé → variant ``∅``.
  - Nouveau module `picarones/report/lexical_modernization_render.py` :
    - ``build_lexical_modernization_html(data, labels, top_n,
      min_total)`` produit un tableau HTML 4 colonnes (forme
      historique GT, variantes OCR, n GT, % modernisé).
    - Cellule ``% modernisé`` colorée en gradient blanc → orange.
    - Compactage des variants : top 3 affichés + ``+N`` pour le
      reste.
    - Adaptive : ``""`` si ``data is None`` ou aucun token
      modernisé.
  - +6 clés i18n FR/EN (``lexmod_*``).
  - +20 tests dans `test_sprint80_lexical_modernization.py` :
    couche calcul (9 cas — systématique, préservé, partiel,
    multi-variants, stop-list, casse, suppression, vide, None) ;
    agrégation (2 cas) ; top (2 cas — tri, min_total) ; rendu
    (5 cas — None, no_modernization, table, %, anti-injection) ;
    complétude i18n FR + EN.
  - **Verrou levé** : le chercheur peut désormais lire « maistre
    → maître modernisé dans 100 % des cas » et ajuster son prompt
    en conséquence pour préserver l'orthographe historique.
    L'information est exploitable au lieu d'un score agrégé
    abstrait.

- **Sprint 79 — A.I.6 : projection de coût en volume cible
  (couche de calcul).**  La vue Pareto (Sprint 20) trace CER vs
  coût mais le coût est par unité (1 000 pages).  Pour décider
  business-side, il faut projeter ce coût sur le **volume cible**
  que l'utilisateur prévoit de traiter — payer 50 € de plus sur
  50 pages est trivial, sur 5 millions ça change tout.
  - Nouveau module `picarones/core/cost_projection.py` :
    - Dataclass ``ProjectedCost(engine_key, target_pages,
      cost_total_eur, co2_total_g, cost_per_1k_pages_eur,
      co2_per_1k_pages_g, type)``.
    - ``project_cost_total(engine_cost, target_pages)`` : coût
      total linéaire en pages.  ``None`` si données insuffisantes
      ou ``target_pages < 0``.
    - ``project_co2_total(engine_cost, target_pages)`` :
      empreinte CO₂ en grammes pour le volume cible (étiqueté
      « expérimental » dans ``pricing.py`` Sprint 20).
    - ``project_engine(engine_cost, target_pages)`` : retourne
      le ``ProjectedCost`` complet.
    - ``project_all_engines(engine_costs, target_pages)``
      projette N moteurs en une passe.  ``ValueError`` si
      ``target_pages < 0``.
    - ``cost_gap_table(projections, baseline_engine)`` retourne
      ``{engine: {total, delta_abs, delta_rel}}`` vs baseline ;
      ``KeyError`` si baseline inconnue ; ``delta_rel = None`` si
      baseline = 0 (pas de division silencieuse).
  - +17 tests dans `test_sprint79_cost_projection.py` :
    couche calcul (5 cas — linear, zero, négatif, no_data,
    fractionnel), CO₂ (2 cas), engine (2 cas), all_engines (3
    cas), gap_table (4 cas — vs baseline, baseline inconnue,
    baseline=0, données manquantes), **cas réaliste BnF**
    (80 000 pages BMS avec 4 moteurs : Tesseract 3,20 €, Pero
    0 €, Mistral 280 €, GPT-4o 600 €).
  - **Verrou levé** : la couche calcul est prête pour câbler le
    panneau « Avancé » (Sprint 21) avec le champ « Volume cible »
    qui recalcule la vue Pareto et la table coût en valeur
    totale projetée.  L'UX et le câblage HTML suivront — la
    base est testée et auto-documentée.

- **Sprint 78 — A.I.5 : équivalences diplomatiques en curseur
  fin (couche de calcul).**  Aujourd'hui les profils de
  ``picarones/core/normalization.py`` (``medieval_french``,
  ``early_modern_french``, etc.) appliquent un **bloc entier**
  de transformations.  Mais un éditeur peut vouloir nuancer :
  *« je tolère ``ſ → s`` mais pas ``u → v`` »*.  Ce sprint
  éclate chaque profil en règles d'équivalence **nommées et
  indépendantes** que l'utilisateur peut activer ou désactiver
  une par une.
  - Nouveau module `picarones/core/equivalence_profile.py` :
    - Dataclass ``EquivalenceRule(name, source, target,
      description, profile_tag)``.
    - Catalogue ``BUILTIN_EQUIVALENCES`` construit
      automatiquement depuis les ``DIPLOMATIC_*`` existants avec
      noms canoniques stables (``longs_s``, ``u_eq_v``,
      ``i_eq_j``, ``ae_ligature``, ``thorn_th``, ``vv_eq_w``,
      etc.) : 15 règles couvrant les 4 profils intégrés.
    - ``list_equivalences_by_profile(profile_name=None)`` pour
      grouper par profil dans l'UX.
    - ``apply_selected_equivalences(text, selected_names)``
      applique uniquement les règles dont le nom est dans
      ``selected_names``.  Règles inconnues ignorées
      silencieusement avec warning.  Texte vide / None → ``""``.
    - ``compute_cer_with_equivalences(reference, hypothesis,
      selected_names)`` retourne le CER après normalisation
      sélective sur les **deux** côtés (GT et hyp).
  - Aucune modification de ``normalization.py`` — purement
    additif.
  - +17 tests dans `test_sprint78_equivalence_profile.py` :
    catalogue (4 cas — règles canoniques, structure, noms
    uniques, longs_s correct), liste par profil (3 cas), apply
    (6 cas — sélectif, exclu, multi, vide, texte vide, règle
    inconnue), compute_cer (4 cas — drop avec eq, application
    bilatérale, diff résiduelle, vide).
  - **Verrou levé** : la couche calcul est en place pour qu'un
    développeur frontend puisse câbler le panneau « Avancé » du
    rapport (Sprint 21) avec des cases à cocher granulaires et
    recalcul JS client.  L'UX panneau avancé (état URL
    persisté, debounce 1s) suivra dans un sprint dédié — la
    base est livrée, testée, et auto-documentée.

- **Sprint 77 — A.I.4 chantier 3 : taxonomie comparative
  côte-à-côte (clôture A.I.4).**  Troisième et dernier chantier
  d'A.I.4.  Le détecteur ``error_profile_outlier`` (Sprint 19)
  signale qu'un moteur a un profil taxonomique éloigné de ses
  concurrents, mais sans visualisation.  Ce sprint répond à
  *« deux moteurs ont le même CER global, mais lequel fait des
  erreurs plus récupérables ? »*.
  - Nouveau module `picarones/core/taxonomy_comparison.py` :
    - ``compare_taxonomies(engine_a, counts_a, engine_b, counts_b)``
      normalise les comptes en proportions (somme = 1), calcule
      les ``deltas`` signés (b - a) par classe, et agrège par
      niveau de **récupérabilité éditoriale** :

      - ``recoverable``   : case_error, ligature_error,
        abbreviation_error (corrigeables par post-processing
        trivial)
      - ``difficult``     : diacritic_error, visual_confusion,
        hapax (effort modéré requis)
      - ``irrecoverable`` : lacuna, oov_character,
        segmentation_error (impossibles sans relire l'image)
    - Constante ``RECOVERABILITY`` exportée pour utilisation
      externe.
    - Retourne ``None`` si les deux moteurs ont 0 erreur chacun.
  - Nouveau module `picarones/report/taxonomy_comparison_render.py` :
    - ``build_taxonomy_comparison_html(data, labels)`` produit
      titre + note d'usage + diagramme miroir SVG + tableau
      résumé par catégorie.
    - ``_build_mirror_chart_svg`` server-side : une ligne par
      classe, deux barres horizontales (A à gauche, B à droite),
      étiquette de classe au centre, valeurs en %.  Couleur de
      la barre selon ``recoverability`` (vert / orange / rouge).
      Échelle normalisée à la proportion max pour visibilité
      uniforme.
    - ``_build_recoverability_summary_html`` : tableau 3 lignes
      (Récupérable / Difficile / Irrécupérable) × 2 colonnes
      (engine A / engine B) avec pastille colorée et %.
    - Adaptive : ``""`` si ``data is None`` ou pas de classes.
    - Anti-injection systématique sur noms de moteurs et labels
      i18n.  Accessible : ``role="img"`` + ``aria-label``.
  - +6 clés i18n FR/EN (``taxocomp_*``) avec template Python
    ``{engine_a}/{engine_b}``.
  - +18 tests dans `test_sprint77_taxonomy_comparison.py` :
    couche calcul (7 cas — proportions, deltas signés,
    récupérabilité, vide, classe unique chez un moteur, totaux,
    sanité ``RECOVERABILITY`` couvre toutes ``ERROR_CLASSES``),
    rendu (7 cas — None, SVG, noms moteurs, labels classes,
    résumé récupérabilité, % affichés, codes couleur), anti-
    injection (nom moteur + label i18n), complétude i18n FR + EN.
  - **Choix éditorial assumé** : la classification
    ``recoverable``/``difficult``/``irrecoverable`` est un
    **guide pragmatique pour le chercheur**, pas un verdict
    imposé.  La note explicative dit textuellement « à CER égal,
    un moteur dont les erreurs sont majoritairement vertes est
    préférable pour une édition critique » — c'est au chercheur
    de juger selon ses besoins.
  - **A.I.4 livré bout-en-bout** : co-occurrence (Sprint 75) +
    intra-document (Sprint 76) + comparatif (Sprint 77).

- **Sprint 76 — A.I.4 chantier 2 : évolution intra-document
  des classes taxonomiques (couche calcul + heatmap SVG).**
  Deuxième des trois chantiers d'A.I.4.  ``line_metrics.py``
  (Sprint 10) avait déjà une heatmap **CER × position** dans le
  document ; ce sprint l'étend à toutes les classes
  taxonomiques : où dans le document apparaît tel type d'erreur ?
  Lecture concrète : ``ligature_error`` concentré dans la
  première tranche → erreur de **marge** ; uniformément réparti
  → erreur de **scribe**.
  - Nouveau module `picarones/core/taxonomy_intra_doc.py` :
    - ``compute_taxonomy_position_heatmap(reference, hypothesis,
      n_bins=10)`` calcule, pour chaque classe taxonomique, le
      nombre d'erreurs par tranche de position.  Réutilise la
      logique mot-à-mot de ``classify_errors`` (Sprint 5) en
      gardant la position du mot GT (``i1`` dans la diff
      word-level) et en binnifiant par
      ``floor(i1 / n_gt_words * n_bins)``.
    - ``_classify_word_pair`` : variante pure de la
      classification (sans modifier de compteurs externes).
    - Helper ``_bin_for_position`` : clip entre 0 et n_bins-1.
    - ``ValueError`` si ``n_bins ≤ 0``.  Retourne ``None`` si
      la GT est vide.
  - Nouveau module `picarones/report/taxonomy_intra_doc_render.py` :
    - ``build_taxonomy_intra_doc_html(data, labels)`` produit
      heatmap SVG + titre + note d'usage.
    - ``_build_heatmap_svg`` server-side : grille
      classes_avec_erreurs × n_bins, gradient blanc → orange
      profond (#c2410c), valeur affichée si > 0, étiquettes de
      colonnes (positions 1..N) et de lignes (noms de classes),
      légende axe X.  Densité **relative au max de la classe**
      (mise en évidence des positions concentrées).
    - Adaptive : ``""`` si ``data is None``, ``total_errors=0``
      ou aucune classe avec erreurs.  Filtrage : seules les
      classes ayant ≥ 1 erreur apparaissent en ligne.
    - Accessible : ``role="img"`` + ``aria-label``.
  - +3 clés i18n FR/EN (``intradoc_title``, ``intradoc_note``,
    ``intradoc_n_words`` avec template Python).
  - +16 tests dans `test_sprint76_taxonomy_intra_doc.py` :
    couche calcul (8 cas — identité, GT vide, erreur en début,
    erreur en fin, distribution uniforme, ``n_bins`` invalide,
    breakdown par classe, plus de bins que de mots), rendu (5
    cas — None, no_errors, SVG, labels, n_words affichés),
    anti-injection, complétude i18n FR + EN.
  - **Verrou levé** : un chercheur peut désormais voir, pour un
    document donné, **où** chaque type d'erreur apparaît — utile
    pour distinguer erreurs de marge, erreurs de scribe, et
    erreurs concentrées sur des sections spécifiques (titres,
    manchettes…).

- **Sprint 75 — A.I.4 chantier 1 : co-occurrence taxonomique
  (couche calcul + heatmap SVG bout-en-bout).**  Premier des trois
  chantiers d'A.I.4.  La taxonomie d'erreurs (10 classes,
  ``picarones/core/taxonomy.py``) est calculée par document
  depuis longtemps mais le rapport ne montre qu'un seul
  histogramme global.  Ce sprint répond à *« quelles classes
  d'erreur tendent à apparaître ensemble dans les mêmes
  documents ? »* — utile pour stratifier *a posteriori* (« mes
  documents difficiles ont tous ``ligature_error`` +
  ``abbreviation_error`` ensemble : signal d'un type de scribe »).
  - Nouveau module `picarones/core/taxonomy_cooccurrence.py` :
    - ``compute_taxonomy_cooccurrence(per_doc_classes,
      min_doc_count=1, top_n_pairs=10)`` calcule l'indice de
      **Jaccard** entre paires de classes au niveau **document**
      (présence binaire — un doc « contient » la classe X ou
      pas).  Symétrique, diagonale = 1.0 pour les classes
      présentes.
    - Filtrage des classes anecdotiques via ``min_doc_count``
      (défaut 1).
    - ``top_pairs`` : top-N paires triées par Jaccard décroissant
      (utile pour la table HTML compacte).
    - Retourne ``None`` si ``per_doc_classes`` vide ou si aucune
      classe ne dépasse ``min_doc_count``.
  - Nouveau module `picarones/report/taxonomy_cooccurrence_render.py` :
    - ``build_taxonomy_cooccurrence_html(data, labels)`` produit
      titre + note + heatmap SVG + table top_pairs.
    - ``_build_heatmap_svg`` server-side : grille N×N avec
      cellules colorées par gradient blanc → bleu profond
      (#1e3a8a) selon Jaccard, valeur affichée si > 0,05,
      étiquettes rotées -45° en haut, normales à gauche.  SVG
      accessible (``role="img"`` + ``aria-label``).
    - ``_build_top_pairs_table`` : table HTML avec cellule
      Jaccard colorée pour lecture rapide.
    - Adaptive : ``""`` si ``data is None`` ou matrice vide.
  - +5 clés i18n FR/EN (``taxocooc_*``).
  - +22 tests dans `test_sprint75_taxonomy_cooccurrence.py` :
    couche calcul (11 cas — toujours/jamais ensemble, diagonale,
    symétrie, chevauchement partiel, vide, ``min_doc_count``,
    ``top_pairs`` triées et limitées, ``doc_count``, doc=None),
    rendu (7 cas — None, classes vides, SVG, table, valeurs
    affichées, étiquettes, n_docs), anti-injection (classe
    ``<script>`` + label i18n), complétude i18n FR + EN.
  - **Verrou levé** : un chercheur peut désormais voir d'un coup
    d'œil quelles classes d'erreur sont corrélées dans son
    corpus, et utiliser cette info pour stratifier *a posteriori*
    ses documents difficiles.

- **Sprint 74 — A.I.3 chantier 1 : encart « Ce corpus est-il
  habituel ? » (clôture A.I.3).**  Suite directe Sprint 73
  (couche calcul + détecteur narratif).  Ce sprint livre le
  rendu HTML de l'encart qui place la difficulté du corpus
  courant dans la distribution des corpus précédents stockés
  en SQLite (Sprint 8) — phrase factuelle + mini-boxplot SVG.
  - Nouveau module `picarones/report/baseline_render.py` :
    - ``build_corpus_difficulty_baseline_html(percentile_data,
      historical_values, labels)`` produit l'encart complet
      (titre + phrase factuelle + boxplot SVG si valeurs
      fournies).  Phrase template auto-sélectionnée selon les
      flags ``harder_than_usual`` / ``easier_than_usual`` /
      « usual » du percentile_data.
    - ``_build_difficulty_boxplot_svg(historical_values,
      current, width, height)`` construit un boxplot horizontal
      SVG **server-side** (pas de JavaScript) avec :
      - moustache min → max (ligne grise)
      - boîte Q1 → Q3 (rectangle gris clair)
      - médiane (trait noir épais)
      - point courant (cercle coloré)
    - **Couleur du point courant adaptive** :
      - bleu (#3b87d8) si current < Q1 (corpus plus facile que
        d'habitude)
      - rouge (#d8553b) si current > Q3 (plus difficile)
      - vert (#5fa860) sinon (habituel)
    - Étiquettes numériques min / max / current visibles (fonts
      explicites).
    - SVG accessible : ``role="img"`` + ``aria-label``.
    - Adaptive : retourne ``""`` si ``percentile_data is None``
      (rapport adaptatif).  Si ``historical_values`` vide /
      ``None``, seule la phrase factuelle est rendue (le boxplot
      est omis silencieusement).
  - Helper interne ``_quantiles(values)`` calcule
    (min, Q1, median, Q3, max) avec méthode inclusive — gère le
    cas N=0 et N=1.
  - +4 clés i18n FR/EN (``baseline_corpus_title``,
    ``baseline_corpus_harder``, ``baseline_corpus_easier``,
    ``baseline_corpus_usual``).  Templates Python avec
    placeholders ``{current:.2f}``, ``{percentile:.0f}``,
    ``{n_runs}``.
  - +20 tests dans `test_sprint74_baseline_html.py` :
    - ``_quantiles`` (3 cas — simple, vide, single)
    - SVG (8 cas — bien formé, vide, couleurs harder/easier/usual,
      box+moustaches+cercle, dégénéré tous identiques, current
      hors range historique)
    - HTML (6 cas — None, harder/easier/usual, SVG omis sans
      values, SVG présent avec values)
    - anti-injection sur label i18n
    - complétude i18n FR + EN
  - **Verrou levé** : un benchmark BnF avec un historique SQLite
    chargé peut désormais générer en tête de rapport un encart
    qui dit *« ce corpus est plus difficile que la moyenne — au
    88ᵉ percentile des 47 corpus précédents »* avec un boxplot
    qui le visualise.  L'A.I.3 est livré bout-en-bout (Sprint 73
    couche calcul + détecteur, Sprint 74 vue HTML).

- **Sprint 73 — A.I.3 chantier 2 : détecteur narratif
  ``engine_off_baseline`` (couche calcul + narrative).**  L'historique
  SQLite (Sprint 8) existait depuis longtemps mais aucun détecteur
  narratif ne le lisait.  Ce sprint répond à *« comment ce moteur
  se comporte-t-il sur ce corpus, par rapport à ses runs précédents
  de mon institution ? »*.  L'encart HTML « Ce corpus est-il
  habituel ? » (chantier 1 d'A.I.3, boxplot SVG) suit Sprint 74.
  - Nouveau module `picarones/core/baseline_comparison.py` :
    - ``compute_engine_baseline(history, engine_name, corpus_name,
      current_cer, *, current_run_id, min_runs=5,
      relative_delta_threshold=0.20)`` retourne un dict avec
      ``cer_current``, ``cer_historical_mean``,
      ``cer_historical_median``, ``n_runs``, ``absolute_delta``,
      ``relative_delta``, ``off_baseline``.  Filtre par moteur ×
      corpus (apple-to-apple), exclut le run courant si fourni,
      ignore les CER négatifs / None, retourne ``None`` si moins
      de ``min_runs`` runs historiques.
    - ``compute_corpus_difficulty_percentile(history,
      current_difficulty, *, min_runs=5)`` place la difficulté du
      corpus courant dans la distribution historique (lit
      ``HistoryEntry.metadata["difficulty"]``).  Retourne
      ``percentile``, ``median_historical``, flags
      ``harder_than_usual`` (P75+) et ``easier_than_usual`` (P25-).
  - Nouveau ``FactType.ENGINE_OFF_BASELINE`` dans
    ``narrative/facts.py``.
  - Nouveau détecteur ``detect_engine_off_baseline`` dans
    ``narrative/detectors.py`` (priority 150) :
    - Lit ``benchmark_data["baseline_comparisons"]`` (liste de
      dicts produits par ``compute_engine_baseline``).
    - Émet 1 Fact par moteur off_baseline.
    - Importance ``HIGH`` si ``|relative_delta| ≥ 50 %``,
      ``MEDIUM`` sinon.
    - Garde-fous : silencieux si ``baseline_comparisons`` absent
      ou vide, si ``relative_delta`` est ``None`` (baseline = 0
      non calculable), si ``off_baseline=False``.
  - Nouveaux templates FR/EN dans
    ``narrative/templates/{fr,en}.yaml``.  Phrase factuelle type :
    *« tess a obtenu 5,2 % CER ici, vs 4,1 % en moyenne sur les
    12 runs précédents… »*.
  - +21 tests dans `test_sprint73_baseline_comparison.py` :
    - couche calcul (off_baseline_higher, within_baseline,
      min_runs filter, custom_min_runs, current_run_excluded,
      filter par engine+corpus, CER None ignorés, baseline=0 →
      relative None, current_cer invalide)
    - difficulty_percentile (calcul, harder/easier, min_runs)
    - détecteur (silent sans data, silent off=False, silent
      relative=None, fact émis, importance HIGH si ≥50%, multiple
      moteurs)
    - **traçabilité anti-hallucination** FR + EN : chaque nombre
      dans le texte rendu est traçable au payload.
  - **Verrou levé** : un benchmark BnF qui pousse ses résultats
    dans l'historique SQLite et qui passe ``baseline_comparisons``
    au moteur narratif voit automatiquement, dans la synthèse en
    tête de rapport, *« ce moteur a un CER inhabituel sur ce
    corpus par rapport à vos 12 runs précédents »*.

- **Sprint 72 — A.I.1 chantier 1 : vue « Worst lines globale »
  (clôture A.I.1).**  Suite directe Sprint 71 : la roadmap A.I.1
  comporte deux chantiers — la métrique rare-token recall (livrée)
  et la vue HTML qui expose les lignes individuelles les plus mal
  transcrites du corpus.  Ce sprint livre la vue.
  - Nouveau module `picarones/core/worst_lines.py` :
    - Dataclass ``WorstLineEntry(rank, cer, engine_name, doc_id,
      line_index, gt_line, hyp_line, script_type)``.
    - ``extract_worst_lines(benchmark, top_n=20, engine_filter,
      script_type_filter)`` collecte transversalement à tous les
      moteurs et documents, filtre par moteur et par strate
      (Sprint 45 ``doc_strata``), trie par CER décroissant, retourne
      les ``top_n`` premières avec rang 1-based.
    - Récupération des textes GT/hyp par re-split du
      ``DocumentResult.ground_truth`` / ``hypothesis`` à l'index de
      ligne (cf. limite : suppose un ``BenchmarkResult``
      non-compacté).
    - Lignes avec ``cer == 0.0`` ignorées (pas dans le worst).
  - Nouveau module `picarones/report/worst_lines_render.py` :
    - ``build_worst_lines_table_html(entries, labels)`` : tableau
      HTML server-side avec colonnes Rang / CER (cellule colorée
      gradient jaune→rouge) / Moteur / Document / Ligne # /
      [Strate] / Diff GT→OCR.  Colonne strate **adaptive**
      (omise si aucune entry n'a de ``script_type``).
    - Diff caractère par caractère via
      ``diff_utils.compute_char_diff`` (réutilisation Sprint 5),
      rendu inline avec rouge clair barré pour suppressions et vert
      clair pour insertions.
    - Anti-injection systématique sur engine_name, doc_id, GT/hyp
      lines, labels i18n.
    - Retourne ``""`` si la liste est vide (rapport adaptatif).
  - +25 tests dans `test_sprint72_worst_lines.py` :
    extraction (top_n, tri par CER décroissant, rang 1-based,
    top_n=0, lignes CER=0 ignorées) ; filtres (par moteur, par
    strate, valeurs inconnues) ; cas limites (pas de line_metrics,
    benchmark vide, sans doc_strata, hyp plus courte que GT) ;
    rendu (tableau, colonnes attendues, strate adaptive, cellule
    CER colorée, diff rendu, % affiché) ; anti-injection
    (engine_name, doc_id, GT line, label i18n).
  - **Verrou levé** : un chercheur qui voit *« 5 % de mes lignes
    ont un CER > 0,42 »* dans le rapport peut désormais voir
    **quelles** lignes — diff inline, document parent, ligne #,
    moteur — pour comprendre ce qui casse précisément.

- **Sprint 71 — A.I.1 chantier 2 : rare-token recall (couche
  de calcul, démarrage de la résolution des critiques
  structurelles A.I).**  Premier sprint du chantier A.I qui
  s'attaque à la critique « la granularité ne s'arrête plus à la
  page ».  Pour répondre à *« mon OCR a 5 % de CER, mais
  préserve-t-il les noms propres rares qui m'intéressent pour
  l'indexation prosopographique ? »*, le module mesure le **rappel
  sur les tokens rares** d'un corpus (hapax + dis legomena,
  défaut ``max_freq=2``).
  - Nouveau module `picarones/core/rare_tokens.py` :
    - ``tokenize(text)`` Unicode-aware : préserve les
      contractions (``L'an``, ``d’une``), composés
      (``peut-être``, ``c'est-à-dire``), apostrophe typographique
      ``’`` (U+2019).
    - ``frequency_distribution(documents, case_sensitive=False)``
      : ``{token: count}`` corpus-wide.
    - ``extract_rare_tokens(documents, max_freq=2)`` retourne le
      ``frozenset`` des tokens dont la fréquence corpus-wide est
      ``≤ max_freq``.  ``max_freq < 1`` → ``ValueError``.
    - ``compute_rare_token_recall(reference, hypothesis,
      rare_tokens)`` retourne ``{n_rare_tokens_in_reference,
      n_rare_tokens_recalled, recall, missed_tokens}``.
      Alignement **bag-of-tokens avec multiplicité** : un token
      rare présent 2× en GT et 1× en hyp donne recall = 0,5 (pas
      1,0).  ``missed_tokens`` liste les manqués avec
      multiplicité.
    - ``rare_token_recall`` raccourci.
  - **Pas d'enregistrement dans le registre typé Sprint 34** : la
    métrique exige un **3ᵉ argument** (le set des tokens rares,
    calculé corpus-wide en amont).  L'utilisateur appelle
    explicitement la fonction avec son set — pas de magie globale.
  - +28 tests dans `test_sprint71_rare_tokens.py` :
    tokenisation (8 cas — contractions ASCII et typographiques,
    composés, diacritiques, ponctuation, nombres, vide),
    frequency_distribution (4 cas — single/multi/casse), extraction
    (4 cas — hapax, hapax+dis legomena, max_freq invalide, vide),
    recall (10 cas — full/partiel/zéro, multiplicité, GT sans
    rare, hyp vide, None, casse), raccourci, et **2 cas réalistes
    « registre d'état civil »** dont un test de propriété qui
    démontre la conjecture du plan : un OCR qui rate les noms
    propres a un rare-token recall plus dégradé qu'un OCR qui
    rate un mot fréquent (« le »), même si leur CER caractère est
    similaire.
  - **Verrou levé** : un bench BnF qui veut savoir « ce moteur
    préserve-t-il bien les noms de famille de mes registres ? »
    a maintenant la métrique adaptée.  La vue HTML « Worst lines
    + tokens rares manqués » suit Sprint 72 (chantier 1 d'A.I.1).

- **Sprint 70 — CLI pour piloter les pipelines composées sans
  Python (axe B, suite Sprints 63-69).**  Permet de spécifier une
  pipeline ou une comparaison de N pipelines dans un fichier
  **YAML déclaratif** et de les exécuter via la CLI Picarones, sans
  écrire une ligne de Python.  Utile pour la reproductibilité
  (specs versionnées en git) et pour les non-développeurs.
  - Nouveau module `picarones/core/pipeline_spec_loader.py` :
    - ``load_pipeline_spec_from_yaml(path)`` /
      ``load_pipeline_spec_from_dict(data)`` : parse un YAML et
      construit une ``PipelineSpec``.  Format :
      ``name`` + liste de ``steps``, chaque step ayant ``name``,
      ``module`` (dotted path Python vers la classe ``BaseModule``
      tierce), ``args`` (kwargs constructeur, optionnel),
      ``inputs_from`` (DAG branchant Sprint 66, optionnel).
    - ``load_comparison_specs_from_yaml(path)`` : parse un YAML
      contenant ``pipelines: [...]`` et retourne ``(specs,
      extras)`` où ``extras`` est le dict YAML brut (pour
      récupérer ``rankings``, ``baseline``…).
    - Import dynamique via ``importlib.import_module`` ; la classe
      référencée doit hériter de ``BaseModule`` (validation
      stricte).
    - Exception dédiée ``PipelineSpecLoadError`` avec messages
      explicites pour 8 cas d'erreur (chemin invalide, module
      introuvable, classe absente, classe non-BaseModule,
      constructeur incompatible, ``args`` non dict, type
      d'artefact inconnu, champ requis absent).
    - **Aucun module métier n'est créé** : le YAML référence
      uniquement les classes que l'utilisateur a installées dans
      son environnement Python.  Picarones se contente de les
      importer et de les brancher.
  - Nouveau sous-groupe CLI `picarones pipeline` dans
    `picarones/cli.py` :
    - ``picarones pipeline run <spec.yaml> --corpus <dir>``
      [``--output-json``] [``--output-html``] [``--lang``] :
      exécute une pipeline composée sur un corpus.  Affiche le
      résumé par étape (succès, taux), exporte JSON brut et/ou
      HTML autonome (Sprint 67).
    - ``picarones pipeline compare <specs.yaml> --corpus <dir>``
      [``--output-html``] [``--baseline``] [``--lang``] : compare
      N pipelines sur le même corpus.  Affiche le ranking par
      CER, exporte le rapport comparatif HTML autonome (Sprint
      68) avec ``ranking_specs`` extraits du YAML
      (``rankings`` au top-level) ou par défaut CER seul.
  - +27 tests dans `test_sprint70_pipeline_cli.py` :
    ``_resolve_class`` (5 cas — valide, sans dot, module
    introuvable, classe absente, cible non-classe),
    ``load_pipeline_spec_from_dict`` (9 cas — valide minimal,
    avec args, name/steps/module manquants, args non dict, classe
    non BaseModule, constructeur invalide, inputs_from valide,
    inputs_from type inconnu), ``load_pipeline_spec_from_yaml``
    (3 cas — fichier introuvable, YAML invalide, round-trip
    valide), ``load_comparison_specs`` (2 cas), CLI ``pipeline
    run`` (2 cas — basic + avec output JSON+HTML), CLI ``pipeline
    compare`` (2 cas — basic + avec HTML et baseline), CLI help
    (3 cas — pipeline groupe listé, run et compare avec leurs
    options).
  - **Tous les modules utilisés sont des mocks** définis dans le
    fichier de test (``_CLIMockOCR``, ``_NotABaseModule``).
    Picarones n'expose volontairement aucun module métier.
  - **Verrou levé** : un workflow BnF type — décrire la pipeline
    dans ``my_pipeline.yaml``, versionner le fichier en git, lancer
    ``picarones pipeline run my_pipeline.yaml --corpus ./scans
    --output-html rapport.html`` — fonctionne sans qu'un
    ingénieur Python soit dans la boucle.

- **Sprint 69 — Documentation utilisateur « Écrire un module pour
  le banc d'essai de pipelines » (axe B, suite Sprints 63-68).**
  Premier guide pédagogique dédié à l'axe B : un chercheur ou
  ingénieur qui veut **brancher son propre module** dans Picarones
  (correcteur LLM, reconstructeur ALTO, classifieur d'entités,
  re-segmenteur…) trouve maintenant un guide complet bout-en-bout.
  - Nouveau document `docs/tutorials/writing-a-pipeline-module.md` :
    - **TL;DR** avec un exemple `MyCorrector` minimal en 25 lignes.
    - Section **« Le contrat ``BaseModule`` »** : tableau des
      champs obligatoires (``input_types``, ``output_types``,
      ``execution_mode``, ``name``, ``process``) et liste des
      ``ArtifactType`` disponibles.
    - Section **« Exemples pédagogiques »** : trois mocks
      explicitement étiquetés « pédagogique » et marqués « à NE
      PAS copier en production » — correcteur LLM TEXT→TEXT,
      reconstructeur TEXT→ALTO, classifieur TEXT→ENTITIES.
    - Section **« Orchestrer une pipeline »** : mono-document
      (Sprint 63), corpus complet avec factory personnalisée
      (Sprint 64), comparaison de N pipelines (Sprint 65), DAG
      branchant via ``inputs_from`` (Sprint 66) — chaque
      sous-section avec snippet exécutable.
    - Section **« Générer un rapport HTML autonome »** : pipeline
      unique (Sprint 67) et comparaison (Sprint 68) avec snippets
      ``Path("rapport.html").write_text(...)``.
    - Section **« Bonnes pratiques »** : discipline des types,
      erreurs gracieuses, mesure du temps wall-clock, **pas de
      seuils éditoriaux dans votre module** (le chercheur juge,
      pas le module).
    - Section **« Anti-patterns »** : trois questions FAQ avec
      réponses explicites — « Pourquoi pas de correcteur LLM
      intégré ? », « Et si je veux juste tester un OCR seul ? »,
      « Mon module a besoin d'état mutable entre documents ? ».
    - **Tableau de référence rapide** des sprints axe B (32-34
      pour les fondations, 63-68 pour l'orchestration et le
      rapport).
  - +34 tests dans `test_sprint69_user_doc.py` :
    - **présence des 7 sections principales** (TL;DR, contrat,
      exemples, orchestration, rapport HTML, bonnes pratiques,
      anti-patterns) — anti-régression doc
    - **15 concepts API mentionnés** (``BaseModule``,
      ``ArtifactType``, ``input_types``, ``inputs_from``,
      ``RankingSpec``, ``compare_pipelines``,
      ``build_pipeline_comparison_report_html``, etc.)
    - **philosophie « banc d'essai pas atelier »** présente
      explicitement dans le doc, mention « aucun module métier »,
      exemples étiquetés « pédagogique »
    - **références aux 6 sprints axe B** (63-68) + au moins un
      sprint de la phase 0 (32-34)
    - **≥ 5 blocs de code Python** + imports valides depuis les
      vrais modules ``picarones.core.*`` et ``picarones.report.*``
  - **Verrou levé** : la barrière d'entrée pour un utilisateur
    tiers qui veut évaluer son module passe d'« il faut lire le
    code source des Sprints 63-68 pour comprendre comment ça
    marche » à « il y a un guide qui couvre le tout en une page,
    avec des snippets prêts à copier ».

- **Sprint 68 — Vue HTML de comparaison de N pipelines composées
  (axe B, suite Sprints 63-67).**  Suite directe Sprint 67 — la
  vue mono-pipeline est étendue avec un rendu **comparatif** entre
  N pipelines exécutées sur le même corpus (Sprint 65).  Pattern
  inchangé : server-side, pas de JS, anti-injection systématique.
  - Extension de ``picarones/report/pipeline_render.py`` :
    - ``RankingSpec(artifact_type, metric_name, higher_is_better=False,
      label=None)`` — dataclass légère qui décrit un classement à
      afficher.  ``display_label`` auto-généré (``"<at>.<metric>"``)
      ou explicite.
    - ``build_pipeline_ranking_table_html(comparison, ranking_spec)``
      — tableau rang × pipeline × valeur, classé selon
      ``ranking_by_final_metric`` (Sprint 65).  Cellule de rang
      colorée par gradient vert (1er) → rouge (dernier).  Pipelines
      sans valeur listés en queue avec tirets.  Vide si la
      comparaison ne contient aucune pipeline.
    - ``build_pipeline_gain_table_html(comparison, ranking_spec,
      baseline_pipeline)`` — tableau pipeline × {valeur, gain
      absolu, gain relatif} vs baseline.  Cellule de gain colorée
      en vert (favorable) ou rouge (défavorable) selon
      ``higher_is_better``.  Baseline marquée explicitement
      ``(référence)``.  Retourne ``""`` si la baseline est inconnue.
    - ``build_pipeline_comparison_summary_html(comparison)`` —
      encart résumé : corpus, n_docs, n_pipelines, durée totale,
      mini-résumé par pipeline (nom + ``n_succeeded/n_docs``).
    - ``build_pipeline_comparison_report_html(comparison,
      ranking_specs, baseline_pipeline, lang)`` — **document HTML
      autonome** (``<!doctype html>``) qui assemble le summary +
      les rankings + les gain tables.  Aucune auto-détection
      magique : si ``ranking_specs`` est vide, on n'affiche que
      le summary ; si ``baseline_pipeline`` est ``None``, pas de
      gain tables.  L'utilisateur déclare explicitement ce qu'il
      veut voir.
  - +14 clés i18n FR/EN (``pipeline_comparison_*``,
    ``pipeline_ranking_*``, ``pipeline_gain_*``,
    ``pipeline_baseline_marker``).  Les libellés
    ``pipeline_ranking_title`` et ``pipeline_gain_title`` sont des
    templates avec placeholders ``{label}`` et ``{baseline}``.
  - +26 tests dans
    `test_sprint68_pipeline_comparison_html.py` :
    ``RankingSpec`` (display_label auto/explicite, défaut
    ``higher_is_better``) ; ranking table (ordre ascendant/
    descendant, pipelines sans valeur en queue, cellule rang
    colorée, vide si comparison vide, label explicite dans titre) ;
    gain table (baseline marquée, valeurs absolues et relatives,
    cellules vert favorable / rouge défavorable selon
    ``higher_is_better``, baseline inconnue → vide) ; summary
    (corpus, comptes, mini-résumé par pipeline) ; document
    autonome (doctype, structure, lang FR/EN, rankings affichés
    si specs, pas de gain table sans baseline) ; anti-injection
    sur pipeline name / corpus / labels i18n ; complétude i18n
    sur les 14 nouvelles clés FR ET EN.
  - **Toujours pas de classification automatique** : on classe et
    on affiche les gains, mais on ne dit jamais « pipeline A est
    la meilleure ».  Le chercheur lit le ranking et décide selon
    ses critères.
  - **Verrou levé** : un chercheur peut désormais générer en une
    ligne le rapport HTML autonome d'une comparaison entre N
    pipelines :

    .. code-block:: python

       html = build_pipeline_comparison_report_html(
           comparison,
           ranking_specs=[
               RankingSpec(ArtifactType.TEXT, "cer", label="CER"),
               RankingSpec(ArtifactType.TEXT, "wer", label="WER"),
           ],
           baseline_pipeline="ocr_only",
       )
       Path("comparaison.html").write_text(html)

- **Sprint 67 — Vue HTML d'un benchmark de pipeline composée
  (axe B, suite Sprints 63-66).**  Pattern identique aux
  Sprints 41 (NER), 43 (calibration) et 62 (philologie) : rendu
  **server-side**, pas de JavaScript, déterministe, anti-injection
  systématique via ``html.escape``.
  - Nouveau module `picarones/report/pipeline_render.py` :
    - ``build_pipeline_summary_html(bench)`` — encart résumé
      global (pipeline, corpus, n_docs, n_pipelines_succeeded /
      failed avec cellule colorée par taux de succès, durée
      totale formatée).
    - ``build_pipeline_steps_table_html(bench)`` — tableau par
      étape avec colonnes : nom, ``n_succeeded`` / ``n_failed``,
      taux de succès (gradient rouge → vert), durée mean /
      median, métriques aux jonctions formatées
      (``<type>.<metric>: mean (n=N)``), error_breakdown
      catégorisé (``raised_exception`` / ``missing_input`` /
      ``missing_output`` / ``pipeline_aborted`` / ``other``).
      Adaptive : retourne ``""`` si aucune étape n'a été agrégée.
    - ``build_pipeline_report_html(bench, lang)`` — **document
      HTML autonome** (``<!doctype html>`` + ``<html>`` +
      ``<head>`` avec styles CSS inline + ``<body>``) que
      l'utilisateur peut écrire directement sur disque, **sans
      dépendre du générateur OCR historique** (rapport pipeline
      distinct du rapport ``BenchmarkResult``).  Helper
      ``_format_duration`` adaptatif (ms / s / mm:ss).
  - **Vue distincte du rapport OCR** : le rapport HTML existant
    (``ReportGenerator``) attend un ``BenchmarkResult`` (axe A) ;
    pour les pipelines composées on utilise
    ``PipelineBenchmarkResult`` (axe B).  Plutôt que de fusionner
    les deux, on livre un rapport autonome à part — clarté
    architecturale et pas de couplage.
  - +18 clés i18n FR/EN (``pipeline_report_*``,
    ``pipeline_summary_*``, ``pipeline_steps_*``,
    ``pipeline_*_label``).
  - +21 tests dans `test_sprint67_pipeline_html.py` :
    - summary : nom de pipeline / corpus / succeeded-failed
      affichés, durée formatée
    - steps table : noms, colonnes (8 attendues), métriques aux
      jonctions affichées (``text.cer 0.182 (n=10)``),
      error_breakdown affiché, vide sans agrégats, cellule de
      taux colorée
    - document autonome : doctype, structure html/head/body,
      styles inline, title contenant le pipeline name, attribut
      ``lang`` (FR + EN), summary et steps inclus
    - anti-injection : pipeline name / corpus name / step name /
      labels i18n contenant ``<script>`` tous correctement
      échappés
    - complétude i18n : 17 clés ``pipeline_*`` présentes en FR
      ET EN
  - **Pas de classification automatique** : le document affiche
    les chiffres bruts par étape ; aucun verdict « pipeline bonne
    ou mauvaise » imposé.
  - **Reporté Sprint 68** : rendu d'un
    ``PipelineComparisonResult`` (ranking + gain table entre N
    pipelines).
  - **Verrou levé** : un chercheur peut désormais générer un
    rapport HTML autonome après ``run_pipeline_benchmark`` —
    ``Path("rapport.html").write_text(build_pipeline_report_html(
    bench))`` — sans rien d'autre.

- **Sprint 66 — DAG branchant via ``inputs_from`` (axe B, suite
  Sprints 63-65).**  Les Sprints 63-65 traitaient des pipelines
  séquentielles : la sortie d'une étape alimente automatiquement
  la suivante via le bag d'artefacts (la dernière version d'un
  type écrase la précédente).  Ce sprint permet de **désigner
  explicitement la source d'un artefact** quand plusieurs étapes
  produisent le même type, débloquant des scénarios fork/merge
  dans une même pipeline (ex. comparer deux corrections LLM en
  parallèle d'un même OCR sans devoir basculer sur deux pipelines
  distinctes via Sprint 65).
  - ``PipelineStep.inputs_from: dict[ArtifactType, str]`` (vide
    par défaut) — pour chaque type d'entrée, l'étape peut désigner
    le nom de l'étape source dont consommer l'artefact.  La chaîne
    spéciale ``"__initial__"`` désigne les entrées initiales
    (utile pour les pipelines démarrant par un type fourni en
    entrée).
  - **Bag versionné** dans ``PipelineRunner.run`` : on stocke
    désormais ``versioned[(type, source_step_name)] = artifact``
    et on maintient un index ``latest[type] = step_name``.  En
    l'absence d'``inputs_from``, le runner prend la version la
    plus récente — comportement Sprint 63 strictement préservé.
  - **Validation étendue** dans ``PipelineSpec.validate`` :
    détecte les références ``inputs_from`` vers une étape inconnue,
    une étape qui ne produit pas le type demandé, ou un type que
    le module ne consomme pas.  Tous les problèmes sont remontés
    avec un message explicite indiquant l'étape concernée et la
    référence litigieuse.
  - **Référence vers étape qui a échoué** : si ``inputs_from``
    pointe vers un step qui a levé une exception, l'étape en aval
    rapporte une erreur ``entrée manquante : <type>@<step>`` —
    le marqueur ``@step`` permet au lecteur de comprendre
    immédiatement que la dépendance pointait vers un step en
    échec, pas un type absent.
  - **Rétrocompat stricte** : sans ``inputs_from``, le
    comportement Sprint 63 est intégralement préservé.  Les 42
    tests Sprints 63-65 passent sans modification.
  - +11 tests dans `test_sprint66_dag_branching.py` :
    - défaut ``inputs_from`` vide
    - validation : référence valide, ``"__initial__"``, étape
      inconnue, type non consommé
    - DAG fork explicite : 2 corrections en parallèle d'un même
      OCR avec métriques indépendantes
    - **fork vs chain divergent** : test propriété qui prouve que
      placer ``inputs_from={TEXT: "ocr"}`` change le résultat
      final (CER 0 en fork vs CER > 0 en chain) sur un même corpus
    - référence vers étape qui a échoué → erreur ``@step`` propre
    - rétrocompat sans ``inputs_from``
  - **Tous les modules utilisés sont des mocks** (``MockOCR``,
    ``TextFixer``, ``TextDoubler``, ``AlwaysFails``).  Picarones
    n'expose volontairement aucun module métier.
  - **Verrou levé** : un chercheur peut désormais composer une
    pipeline qui fork un même OCR vers plusieurs branches de
    correction et évaluer chacune indépendamment, dans une seule
    spec — sans devoir basculer sur ``compare_pipelines`` quand
    le besoin est de tracer le branchement dans un seul contexte
    d'exécution.

- **Sprint 65 — Comparaison de N pipelines composées sur le même
  corpus (axe B, suite Sprints 63-64).**  Réponse à la question
  typique BnF : « OCR seul vs OCR+correcteur A vs OCR+correcteur
  B : laquelle est la meilleure sur mon corpus, et de combien ? ».
  Philosophie inchangée — banc d'essai, pas atelier de production.
  - Nouveau module `picarones/core/pipeline_comparison.py` :
    - ``compare_pipelines(specs, corpus, factories=None)`` exécute
      séquentiellement N ``PipelineSpec`` sur le **même** corpus
      (même GT, comparaison apple-to-apple).  Garde-fou : noms
      uniques exigés (sinon ``ValueError`` explicite).
    - ``factories`` optionnel : dict ``{pipeline_name:
      InitialInputsFactory}`` pour personnaliser les entrées
      initiales par pipeline (utile pour comparer une pipeline qui
      démarre par ``IMAGE`` et une qui démarre par ``TEXT``).
    - ``PipelineComparisonResult`` : conteneur avec
      ``per_pipeline: dict[name → PipelineBenchmarkResult]``,
      ``pipeline_names()`` qui préserve l'ordre d'insertion,
      et deux utilitaires comparatifs.
    - ``ranking_by_final_metric(artifact_type, metric_name,
      higher_is_better=False)`` : trie les pipelines par la valeur
      **finale** de la métrique demandée à la jonction
      ``artifact_type`` (récupère le ``mean`` de la dernière étape
      qui a produit ce type, cohérent avec
      ``PipelineResult.junction_metrics_for`` Sprint 63).  Les
      pipelines sans métrique vont en queue.
    - ``gain_table(artifact_type, metric_name, baseline_pipeline)``
      : pour chaque pipeline, ``{value, absolute, relative}``
      vs baseline.  ``relative`` à ``None`` si baseline = 0
      (évite division par zéro silencieuse).  ``KeyError`` si
      la baseline est inconnue.
  - Approche purement infrastructure : aucun module métier, on se
    contente de réutiliser ``run_pipeline_benchmark`` (Sprint 64)
    pour chaque spec et d'ajouter la couche comparative au-dessus.
  - +13 tests dans `test_sprint65_pipeline_comparison.py`
    (single/multi pipelines, ordre préservé, noms en double,
    corpus vide, ranking ascendant/descendant, pipelines sans
    métrique en queue, ``gain_table`` avec baseline inconnue,
    baseline = 0 → relative None, baseline=self → absolute=0,
    cas réaliste OCR fautif vs OCR+correcteur, factories par
    pipeline, dataclass).
  - **Tous les modules utilisés sont des mocks** (``MockOCR``,
    ``TextFixer``, ``AlwaysFails``).  Picarones n'expose
    volontairement aucun module métier.
  - **Verrou levé** : un chercheur peut désormais comparer N
    pipelines tierces sur le **même** corpus en une commande,
    obtenir le ranking selon la métrique d'intérêt et la table
    de gain vs baseline.  Vue HTML dédiée et tests statistiques
    inter-pipelines arrivent dans les sprints suivants.

- **Sprint 64 — Orchestration corpus-wide d'une pipeline composée
  (axe B, suite directe Sprint 63).**  Le ``PipelineRunner`` du
  Sprint 63 exécute une pipeline sur un seul document ; ce sprint
  fournit l'orchestration sur un **corpus complet** et
  l'agrégation des résultats par étape.  Toujours dans la
  philosophie « banc d'essai, pas atelier de production » : aucun
  module métier n'est ajouté côté Picarones, l'utilisateur amène
  ses propres ``BaseModule`` (Sprint 33).
  - Nouveau module `picarones/core/pipeline_benchmark.py` :
    - ``InitialInputsFactory = Callable[[Document], dict[
      ArtifactType, Any]]`` : type pour la fonction qui produit
      les artefacts initiaux par document.
    - ``default_initial_inputs(doc)`` : factory par défaut qui
      retourne ``{IMAGE: doc.image_path}`` (cas standard d'une
      pipeline qui démarre par un OCR).  L'utilisateur peut
      fournir une factory personnalisée pour brancher d'autres
      sources (par exemple un ``ALTO`` pré-existant).
    - ``StepAggregate(step_name, n_docs, n_succeeded, n_failed,
      duration_seconds_total/mean/median, failing_doc_ids,
      junction_metrics, error_breakdown)`` : agrégat d'une étape
      sur tout le corpus.  Les métriques aux jonctions sont
      agrégées par type d'artefact, avec ``mean`` / ``median`` /
      ``n`` par métrique numérique (les non-numériques sont
      ignorées dans l'agrégat global mais restent visibles par
      doc).  ``error_breakdown`` catégorise les erreurs en
      ``missing_input`` / ``raised_exception`` /
      ``missing_output`` / ``pipeline_aborted`` / ``other`` via
      heuristique stable sur les messages produits par
      ``pipeline_runner._run_step``.
    - ``PipelineBenchmarkResult(pipeline_name, corpus_name,
      n_docs, per_doc_results, per_step_aggregates,
      total_duration_seconds)`` : conteneur global avec
      ``n_pipelines_succeeded`` / ``n_pipelines_failed`` calculés
      à la volée et ``aggregate_for_step(name)`` pour récupérer
      l'agrégat par nom.
    - ``run_pipeline_benchmark(spec, corpus,
      initial_inputs_factory)`` : itère séquentiellement sur les
      documents, appelle ``PipelineRunner.run`` sur chacun,
      capture gracieusement les erreurs de la factory, agrège
      par étape via ``_aggregate_step``.  Une spec invalide
      propage l'erreur à tous les documents (chacun a un
      ``PipelineResult`` avec ``error`` non vide et aucune étape
      exécutée).
  - **Périmètre Sprint 64** : séquentiel inter-documents.
    Comparaison de N pipelines sur le même corpus (Sprint 65),
    DAG branchant (Sprint 66), vue HTML pipelines (Sprint 67),
    parallélisation reportée à arbitrer.
  - +13 tests dans `test_sprint64_pipeline_benchmark.py` :
    factory par défaut, corpus vide, 1 doc OK, métriques agrégées
    sur 3 docs (CER mean/median/n), mix succès/échecs (1 doc
    crash → comptes corrects + failing_doc_ids + error_breakdown
    catégorisé en ``raised_exception``), 2 étapes avec rebond
    propre (étape 1 plante → étape 2 reçoit ``missing_input``
    avec le bon breakdown), spec invalide → tous les docs en
    pipeline_aborted, factory personnalisée, factory qui lève sur
    un doc → autres continuent, dataclasses (success_rate,
    aggregate_for_step retourne None pour nom inconnu).
  - **Tous les modules utilisés sont des mocks définis dans le
    test** (``MockOCR``, ``MockCrasherSometimes``,
    ``MockTextRewriter``).  Picarones n'expose volontairement
    aucun module métier.
  - **Verrou levé** : un utilisateur peut maintenant lancer une
    pipeline composée tierce sur **tout son corpus** en une
    commande, obtenir l'agrégat par étape (durée mean/median,
    métriques mean/median, taux d'erreur par catégorie) et les
    résultats détaillés par document.  La comparaison de
    plusieurs pipelines sur le même corpus arrive Sprint 65, la
    vue HTML dédiée Sprint 67.

- **Sprint 63 — Banc d'essai de pipelines composées : runner +
  évaluation aux jonctions (démarrage axe B du plan 2026).**
  Picarones est et reste un **banc d'essai**, pas un atelier de
  production : ce sprint livre l'infrastructure qui permet
  d'**évaluer des pipelines composées de modules tiers** que
  l'utilisateur amène (ses propres ``BaseModule`` du Sprint 33),
  **sans qu'aucun module métier ne soit fourni par Picarones**
  (pas de reconstructeur ALTO, pas de correcteur LLM, pas de
  re-segmenteur).
  - Nouveau module `picarones/core/pipeline_runner.py` :
    - ``PipelineStep(name, module)`` : une étape lit ses
      ``input_types`` / ``output_types`` directement depuis le
      ``BaseModule`` fourni par l'utilisateur.
    - ``PipelineSpec(name, steps)`` : DAG séquentiel de
      ``PipelineStep`` avec validation statique des types
      (``validate(initial_inputs)`` retourne la liste des
      problèmes ; ``is_valid`` raccourci booléen).
    - ``StepResult(step_name, duration_seconds, output_types,
      junction_metrics, error)`` : résultat d'une étape avec
      durée chronométrée, types effectivement produits, métriques
      aux jonctions et erreur éventuelle.
    - ``PipelineResult(pipeline_name, doc_id, steps,
      total_duration_seconds, error)`` : résultat complet pour un
      document, avec ``succeeded``, ``failing_steps``, et
      ``junction_metrics_for(artifact_type)`` qui retourne les
      métriques de la **dernière étape réussie** ayant produit le
      type demandé.
    - ``PipelineRunner.run(spec, document, initial_inputs)`` :
      exécute la pipeline sur **un seul document**.  À chaque
      étape : valide les entrées disponibles, exécute le module
      avec chronométrage wall-clock, capture gracieusement les
      exceptions (``RuntimeError``, etc.), valide que les sorties
      déclarées sont effectivement produites, met à jour le bag
      d'artefacts disponibles, et **évalue automatiquement chaque
      type produit contre la GT du même niveau** (Sprint 32) via
      ``compute_at_junction`` (Sprint 34) — sélectionnant les
      métriques pertinentes selon les types.
  - **Eager-load** des modules de métriques au top du
    ``pipeline_runner.py`` (``builtin_metrics``, les six modules
    philologiques, NER, reading_order, readability) pour garantir
    que le registre typé soit peuplé avant l'évaluation aux
    jonctions — sans ça, le runner trouverait un registre vide.
  - **Périmètre Sprint 63** : runner séquentiel mono-document.
    DAG branchant, parallélisation, agrégation corpus-wide et
    vue HTML dédiée aux pipelines sont reportés à des sprints
    dédiés.
  - +16 tests dans `test_sprint63_pipeline_runner.py` :
    validation de spec (vide, chaînée, manque d'entrée),
    exécution 1 étape (parfait + imparfait), exécution 2 étapes
    avec évaluation à chaque jonction et CER qui baisse après
    correction par le rewriter, erreurs gracieuses (module qui
    lève → RuntimeError capturé sans arrêter la chaîne ; module
    silencieux qui ne produit pas la sortie déclarée → erreur
    explicite ; spec invalide → erreur en amont, aucune étape
    exécutée), pas de GT → pas de métriques sans erreur, mesure
    du temps par étape, dataclasses (``StepResult`` /
    ``PipelineResult.succeeded`` / ``failing_steps`` /
    ``junction_metrics_for`` qui ignore les étapes en erreur).
  - **Tous les modules utilisés dans les tests sont des mocks
    définis dans le fichier de test** (``MockOCR``,
    ``MockTextRewriter``, ``MockCrasher``, ``MockSilentDropper``)
    — Picarones n'expose volontairement aucun module métier.
  - **Verrou levé** : l'utilisateur peut désormais brancher ses
    propres modules tiers (un correcteur LLM, un reconstructeur
    ALTO, un re-segmenteur, un classifieur d'entités), composer
    une pipeline et obtenir automatiquement les métriques à
    chaque étape contre la GT correspondante.  L'orchestration
    corpus-wide et la vue HTML dédiée arrivent dans les sprints
    suivants de l'axe B.

- **Sprint 62 — Vue HTML « Profil philologique » (clôture du
  câblage philologique bout-en-bout).**  Suite directe Sprint 61
  (câblage backend) — produit le bloc HTML qui remonte les six
  modules philologiques (Sprints 55-60) dans le rapport.  Pattern
  identique aux Sprints 41 (NER) et 43 (calibration) : rendu
  server-side, pas de JavaScript, déterministe.
  - Nouveau module `picarones/report/philological_render.py` :
    - 6 fonctions de rendu de section (une par module) :
      ``build_unicode_blocks_section``,
      ``build_abbreviations_section``, ``build_mufi_section``,
      ``build_early_modern_section``,
      ``build_modern_archives_section``,
      ``build_roman_numerals_section``.
    - Agrégateur ``build_philological_profile_html`` qui assemble
      les sections non vides en un bloc unique avec titre et note
      d'usage explicite (« L'outil ne classifie pas la convention
      adoptée par chaque moteur — c'est au chercheur de lire les
      chiffres et de conclure selon ses critères éditoriaux »).
    - **Adaptive masking complet** : chaque section n'apparaît que
      si au moins un moteur a du signal pour son module ; si aucun
      module n'a de signal, l'agrégateur retourne ``""`` et le
      bloc HTML complet est omis.
    - Cellules colorées par gradient rouge → jaune → vert
      proportionnel au score (identique au pattern Sprint 41
      ``ner_render``) ; pour le statut ``lost`` des numéraux
      romains, la coloration est inversée (un haut taux de perte
      est mauvais).
    - Affichage des effectifs ``n=…`` à côté de chaque score pour
      donner au chercheur le contexte (un score de 100 % sur n=1
      n'a pas la même valeur qu'un 80 % sur n=500).
  - Câblage dans ``ReportGenerator.generate`` : appel de
    ``build_philological_profile_html`` après les blocs
    NER/calibration/inter-moteurs/stratification, passage au
    template via la variable ``philological_profile_html``.
  - Câblage dans ``view_analyses.html`` : un nouveau
    ``chart-card`` pleine largeur conditionné au contenu
    (``{% if philological_profile_html %}``).
  - Anti-injection HTML systématique : tous les noms de moteurs,
    catégories, statuts, libellés i18n passent par
    ``html.escape`` avant insertion (testé : ``<script>`` dans le
    nom du moteur correctement échappé).
  - **Aucune classification automatique** : le mot
    « diplomatique » / « modernisant » n'apparaît que dans la
    note explicative en bas de section, jamais comme étiquette
    accolée à un moteur.
  - +25 clés i18n FR/EN (``philo_profile_*``,
    ``philo_unicode_*``, ``philo_abbreviations_*``,
    ``philo_mufi_*``, ``philo_early_modern_*``,
    ``philo_modern_archives_*``, ``philo_roman_numerals_*``,
    ``philo_roman_status_*``).
  - +18 tests dans `test_sprint62_philological_html.py` (sections
    individuelles ×6, adaptive masking complet, anti-injection sur
    nom de moteur et libellé i18n, valeurs en %, code couleur,
    pas de classification imposée, complétude i18n FR/EN sur les
    25 clés).
  - **Verrou levé** : les six modules philologiques sont désormais
    livrés bout-en-bout (calcul Sprints 55-60 + backend Sprint 61
    + HTML Sprint 62).  Un benchmark sur n'importe quel fonds
    patrimonial européen produit automatiquement, sans
    configuration, un profil philologique lisible dans le rapport
    HTML — donné par catégorie/bloc/statut, sans verdict.

- **Sprint 61 — Câblage backend des métriques philologiques au
  runner (Sprints 55-60).**  Suite directe Sprints 55-60 — les six
  modules philologiques (unicode_blocks, abbreviations, mufi,
  early_modern, modern_archives, roman_numerals) sont désormais
  calculés automatiquement par le runner pour chaque document et
  agrégés par moteur, **sans aucune option à activer**.
  - Nouveau module `picarones/core/philological_runner.py` :
    - ``compute_philological_metrics(reference, hypothesis)``
      calcule les six modules et retourne un dict avec une clé par
      module ayant du **signal exploitable** dans la GT
      (``n_markers_reference > 0``, ``n_mufi_chars_reference > 0``,
      au moins un caractère hors Basic Latin pour unicode_blocks,
      etc.).  Retourne ``None`` si aucun module n'a de signal.
    - ``aggregate_philological_metrics(per_doc_list)`` agrège les
      compteurs bruts par module (somme), recalcule les scores
      globaux à partir des sommes (accuracy, coverage, strict,
      expansion, value, preservation), et préserve les structures
      ``per_block`` / ``per_abbreviation`` / ``per_char`` /
      ``per_category`` / ``per_status`` agrégées.
    - **Adaptive masking** : un module n'apparaît dans le résultat
      que si au moins un document a eu du signal pour lui — les
      rapports restent lisibles sur les corpus sans marqueur
      philologique pertinent (typique des fonds XXIᵉ propres).
  - Nouveaux champs sur ``DocumentResult.philological_metrics`` et
    ``EngineReport.aggregated_philological`` (``Optional[dict]``,
    ``None`` par défaut, sérialisés conditionnellement par
    ``as_dict``, libérés par ``compact``).
  - Câblage dans ``runner._compute_document_result`` : le calcul
    est inconditionnel (coût O(N) sur le texte, négligeable face à
    l'OCR) et l'erreur d'un module individuel ne propage pas — on
    omet le module et on logue un warning explicite (jamais
    ``except: pass`` selon les règles CLAUDE.md).
  - Câblage dans ``run_benchmark`` : agrégation par moteur
    appelée juste après les autres agrégations Sprint 5/10/40/42.
  - **Rétrocompat stricte** : aucun paramètre ajouté, aucun
    comportement existant modifié ; un benchmark sans signal
    philologique voit ses ``philological_metrics`` à ``None`` (pas
    de champ dans le JSON de sortie).
  - +24 tests dans `test_sprint61_philological_runner.py` (champs
    par défaut, sérialisation conditionnelle, libération par
    compact, calcul adaptive sur 6 cas de figure — médiéval,
    imprimé ancien, moderne, numéral romain, diacritiques,
    ASCII pur —, agrégation : sommes correctes, recalcul des scores
    globaux, per_category modern_archives, intégration runner
    end-to-end avec mock ``EngineResult``).
  - **Verrou levé** : les six modules philologiques sont désormais
    visibles dans le pipeline standard de bench ; il manque
    uniquement la vue HTML dédiée (Sprint 62 à venir).

- **Sprint 60 — Numéraux romains transversaux : couche de calcul
  (clôture extension philologique par période).**  Suite directe
  Sprints 56-59.  Les numéraux romains traversent les trois
  périodes patrimoniales : médiéval (minuscules + ``j`` final
  ``mcclxxxij`` = 1282), imprimé ancien (``Tome IV``), moderne
  (``Louis XIV``, ``MCMXIV``).  Pour chaque numéral GT, l'OCR
  peut le **préserver strictement**, **changer la casse**,
  **supprimer le ``j`` médiéval**, **convertir en chiffres
  arabes**, ou le **perdre**.  Ce module classifie les 5 statuts
  pour que le chercheur juge lui-même la convention.
  - Nouveau module `picarones/core/roman_numerals.py` :
    - ``roman_to_int(s)`` : parsing tolérant casse + ``j`` médiéval
      final, validation stricte des paires soustractives canoniques
      (IV, IX, XL, XC, CD, CM uniquement) — rejette « ICI » (faux
      positif), « VV », « LL », « DD », « IIIII ».
    - Forme additive médiévale ``IIII``/``XXXX`` acceptée.
    - ``int_to_roman(n)`` : conversion vers la forme canonique
      majuscule.
    - ``detect_roman_numerals(text, min_length=1)`` : regex
      ``\b[IVXLCDMivxlcdmj]+\b`` + validation par parsing.
      Paramètre ``min_length`` pour filtrer les single-letter
      ambigus (« I » pronom anglais, « M » initiale).
    - ``compute_roman_numeral_metrics(ref, hyp)`` classifie chaque
      numéral GT selon 5 statuts ordonnés par priorité :
      ``strict_preserved``, ``case_changed``, ``j_dropped``,
      ``converted_to_arabic``, ``lost``.  Retourne
      ``per_status``, ``per_numeral``, ``lost_numerals``,
      ``global_strict_score`` (forme exacte uniquement) et
      ``global_value_score`` (toute forme préservant la valeur).
    - Le breakdown ``per_status`` discrimine la convention adoptée :
      - majoritaire ``strict_preserved`` → diplomatique ;
      - majoritaire ``case_changed`` → modernisation typo ;
      - majoritaire ``j_dropped`` → modernisation orthographique
        médiévale ;
      - majoritaire ``converted_to_arabic`` → modernisation
        profonde du système numéral ;
      - majoritaire ``lost`` → erreur OCR.
  - ``roman_numeral_strict_score`` et
    ``roman_numeral_value_score`` enregistrés dans le registre
    typé Sprint 34 pour ``(TEXT, TEXT)``.
  - **Choix éditorial assumé** : aucune classification
    automatique imposée — le chercheur lit ``per_status`` et
    conclut.
  - +93 tests dans `test_sprint60_roman_numerals.py`
    (parsing/conversion bidirectionnelle paramétrée standard +
    minuscules + ``j`` médiéval, formes invalides rejetées,
    aller-retour ``int_to_roman``, détection avec ``min_length``,
    frontière de mot anti-faux-positif (``VIVE`` ne match pas),
    rejet faux positif ``ICI``, **5 statuts discriminés
    individuellement**, priorité strict > arabic, **3 cas
    réalistes par période** — charte médiévale ``mcclxxxij``,
    imprimé ancien ``Tome IV``, moderne ``Louis XIV`` —,
    comptage exhaustif somme des per_status = total, dégénérés,
    raccourcis, intégration registre).
  - **Verrou levé** : l'extension philologique est livrée pour
    les trois périodes principales (médiéval Sprints 56-57,
    imprimé ancien Sprint 58, moderne Sprint 59) **plus** la
    dimension transversale numérale ; un benchmark sur n'importe
    quel fonds patrimonial européen peut classer les moteurs
    sur leur traitement des numéraux romains, indépendamment de
    la période.

- **Sprint 59 — Marqueurs et abréviations des archives modernes
  XIXᵉ-XXᵉ : couche de calcul (extension axe philologique aux
  périodes contemporaines).**  Suite des Sprints 56-58.  Sur les
  fonds modernes BnF (état civil, recensements, presse, monographies,
  archives militaires, annuaires), la typographie historique a
  disparu mais subsiste un riche système d'abréviations propres à
  l'archive contemporaine.  Le module les couvre en **9 catégories**
  pour qu'un chercheur puisse juger lui-même la convention adoptée
  par chaque moteur, **sans qu'aucune classification automatique
  ne soit imposée**.
  - Nouveau module `picarones/core/modern_archives.py` :
    - 9 catégories de marqueurs :
      1. ``civility_titles`` : Mme, Mlle, Mgr, Dr, Pr, Me, M.,
         R.P., S.M., S.A.R., S.E., S.S.
      2. ``ordinals`` : 1ᵉʳ, 1ʳᵉ, 2ᵉ, 2ᵈ, 2ᵈᵉ, 3ᵉ, Iᵉʳ, Vᵉ,
         XIᵉ-XXᵉ (avec exposants Unicode ᵉ ʳ ᵈ).
      3. ``currency`` : ₶ (livre tournois), ₣ ƒ (franc), £, l. s. d.
         (livre/sol/denier d'Ancien Régime).
      4. ``administrative`` : arr., dép., cant., com., reg., prov.
      5. ``civil_status`` : ° (né), † (mort), ✶, ⚭, ép., vve.
      6. ``typographic_punctuation`` : « », –, —, …, ’, ‘.
      7. ``latin_abbr_modern`` : e.g., i.e., etc., cf., ibid.,
         op. cit., ad lib., N.B.
      8. ``bibliographic`` : vol., t., p., pp., n°, fasc., éd.,
         ms., f., r°, v°.
      9. ``address`` : bd, av., r., pl., imp., fbg.
    - ``get_category(marker)`` classe un marqueur ou retourne
      ``None`` ; ``get_expansions(marker)`` retourne les formes
      développées connues.
    - ``detect_modern_markers(text)`` retourne la liste ordonnée
      ``[(index, marker, category)]`` avec **stratégie greedy
      « plus long gagne »** (S.A.R. avant S.A.) et **frontières
      de mot adaptées** (espace/ponctuation pour les abréviations
      à point comme « M. ", « arr. », « r° » ; ``\b`` standard
      pour les alphabétiques comme « Mme » ou « bd »).
    - ``compute_modern_archives_metrics(ref, hyp)`` retourne, par
      catégorie, **deux scores** dans la lignée du Sprint 56 :
      ``strict_score`` (forme abrégée préservée telle quelle) et
      ``expansion_score`` (forme abrégée OU forme développée
      présente, sensible à la casse seulement pour les
      abréviations alphanumériques).  Le ratio des deux par
      catégorie permet au chercheur de **juger lui-même la
      convention** : strict ≈ expansion ≈ 1 → diplomatique ;
      strict = 0, expansion = 1 → modernisant ; les deux faibles
      → erreur OCR.
    - ``missed_markers`` distingue les **pertes pures** (ni
      abrégé ni développé) des **modernisations** (abrégé absent
      mais développé présent) via le booléen
      ``expansion_preserved``.
  - ``modern_archives_strict_score`` et
    ``modern_archives_expansion_score`` enregistrés dans le
    registre typé Sprint 34 pour ``(TEXT, TEXT)``.
  - **Choix éditorial assumé** : aucun module ne classe un moteur
    « diplomatique » ou « modernisant ".  L'outil est conçu pour
    un usage de **recherche** (BnF, philologie, sciences
    historiques) — il fournit les chiffres bruts et laisse le
    chercheur conclure.
  - +75 tests dans `test_sprint59_modern_archives.py`
    (catégorisation paramétrée 33 marqueurs sur 9 catégories,
    ``get_expansions``, détection par catégorie ×9, **greedy
    plus-long-gagne** sur S.A.R., **frontière de mot** sur « bd »
    vs « abdomen ", scénarios standards diplomatique/modernisant/
    erreur, breakdown per_category, dégénérés, missed_markers
    distinguant pertes pures et modernisations, **5 cas réalistes**
    par catégorie clé — citation bibliographique, registre d'état
    civil, adresse de recensement, protocole royal, monnaie
    d'Ancien Régime, ponctuation typographique —, comptage
    exhaustif, sanité tables, raccourcis, intégration registre).
  - **Verrou levé** : un benchmark sur fonds modernes XIXᵉ-XXᵉ
    peut désormais classer les moteurs sur leur traitement des
    abréviations contemporaines — symétrique au Sprint 56
    (médiéval) et au Sprint 58 (imprimé ancien).  L'extension
    philologique couvre maintenant trois périodes principales
    des fonds patrimoniaux européens.

- **Sprint 58 — Marqueurs typographiques de l'imprimé ancien :
  couche de calcul (extension axe philologique aux périodes
  XVIᵉ-XVIIIᵉ).**  Les Sprints 56 (abréviations Capelli) et 57
  (couverture MUFI) sont orientés **médiéval scribal**.  Picarones
  doit aussi servir les éditeurs d'**imprimés anciens**, pour qui
  les marqueurs caractéristiques sont **typographiques** (et non
  scribaux) : ligatures composées, s long ſ, i sans point ı,
  esperluette &, tildes nasaux indiquant une abréviation.
  - Nouveau module `picarones/core/early_modern_typography.py` :
    - 5 catégories de marqueurs typographiques : ``ligatures``
      (ﬀ ﬁ ﬂ ﬃ ﬄ ﬅ ﬆ), ``long_s`` (ſ), ``dotless_i`` (ı),
      ``ampersand`` (&), ``nasal_tildes`` (ã Ã ñ Ñ õ Õ ũ Ũ ẽ Ẽ ĩ Ĩ
      pré-composés + séquences ``voyelle + U+0303``).
    - ``get_category(char)`` classe un caractère dans une catégorie
      ou retourne ``None``.
    - ``detect_markers(text)`` retourne la liste ordonnée des
      marqueurs avec leur index et leur catégorie ; reconnaît à la
      fois les caractères pré-composés et les séquences combinantes
      (``a`` + U+0303 → nasal_tildes).
    - ``compute_early_modern_metrics(ref, hyp)`` retourne le taux
      de préservation par catégorie + ``global_preservation`` +
      ``missed_markers`` (liste des marqueurs ratés avec index et
      catégorie).
    - Approche identique aux Sprints 56-57 (alignement caractère
      par caractère via ``difflib.SequenceMatcher``).
  - ``early_modern_preservation`` enregistré dans le registre typé
    Sprint 34 pour ``(TEXT, TEXT)``.
  - **Le breakdown par catégorie discrimine la convention** : un
    moteur diplomatique préserve toutes les catégories ; un moteur
    modernisant préserve typiquement l'esperluette mais pas les
    ligatures, le s long, le i sans point ni les tildes nasaux ;
    un moteur mixte panache.
  - +38 tests dans `test_sprint58_early_modern.py` (catégorisation
    paramétrée sur 18 caractères, détection des 5 catégories,
    séquence ``voyelle + U+0303``, ordre préservé, **trois
    scénarios standards** discriminés — diplomatique 1.0,
    modernisant 0.2, mixte 0.4 —, breakdown per_category, cas
    dégénérés, comptage exhaustif preserved+missed=total, sets
    disjoints, raccourci, intégration registre).
  - **Verrou levé** : un benchmark sur des imprimés anciens XVIᵉ-XVIIIᵉ
    peut désormais classer les moteurs sur leur convention typographique
    éditoriale (diplomatique vs modernisante) — symétrique à ce que
    le Sprint 56 fait pour les manuscrits médiévaux.

- **Sprint 57 — A.II.3.3 Couverture MUFI : couche de calcul
  (clôture A.II.3 philologique côté calcul).** Suite des Sprints
  55-56 dans l'axe philologique.  La **Medieval Unicode Font
  Initiative** (MUFI v4.0) standardise les caractères médiévaux
  attendus en transcription fidèle ; le module mesure le taux de
  caractères MUFI de la GT correctement restitués dans l'OCR.
  - Nouveau module `picarones/core/mufi.py` :
    - ``_MUFI_RANGES`` : 4 plages Unicode caractéristiques (PUA
      ``E000-F8FF``, Latin Extended-D ``A720-A7FF``, Combining
      Diacritical Marks Supplement ``1DC0-1DFF``, Alphabetic
      Presentation Forms ``FB00-FB4F``).
    - ``_MUFI_EXPLICIT_CHARS`` : liste raisonnée de lettres
      médiévales hors plages (þ, Þ, ð, Ð, ƿ, Ƿ, ſ, æ, Æ, œ, Œ,
      ø, Ø, ƀ, ŧ, đ, ħ, ȝ, Ȝ, ꜿ).
    - ``is_mufi_char(char, custom_chars=None)`` extensible via
      paramètre.
    - ``compute_mufi_coverage(reference, hypothesis, custom_chars)``
      aligne caractère par caractère via
      ``difflib.SequenceMatcher`` (même méthode que le bloc Unicode
      du Sprint 55), classe les positions GT MUFI, et compte les
      positions correctement restituées.
    - Retourne ``coverage`` global + ``per_char`` (total /
      preserved / coverage par caractère MUFI rencontré) + liste
      ``missed_chars`` (caractères MUFI ratés).
  - ``mufi_coverage`` enregistré dans le registre typé Sprint 34
    pour ``(TEXT, TEXT)``.
  - +41 tests dans `test_sprint57_mufi.py` (détection sur 28
    caractères clés, plage PUA, custom_chars extensible ; coverage
    diplomatique → 1, modernisante → 0, partielle avec breakdown
    per_char ; cas dégénérés ; comptage exhaustif ;
    intégration registre).

- **Sprint 56 — A.II.3.2 Score d'expansion d'abréviations
  médiévales : couche de calcul.** Suite du Sprint 55 dans l'axe
  A.II.3 (philologique).  Sur les manuscrits médiévaux, les
  scribes utilisent intensivement des signes d'abréviation
  (``ꝑ``=per, ``ꝓ``=pro, ``⁊``=et, etc.).  Un OCR/HTR a trois
  comportements possibles face à eux : préserver, développer, ou
  perdre.  Le module discrimine les trois.
  - Nouveau module `picarones/core/abbreviations.py` :
    - Table ``ABBREVIATION_EXPANSIONS`` des signes Capelli + MUFI
      les plus courants (10 entrées : ꝑ, ꝓ, ꝗ, ꝙ, ꝯ, ⁊, ꝝ, ꝫ,
      ꝭ, et séquences ``lettre + U+0303`` comme ``p̃``, ``q̃``).
    - ``detect_abbreviations(text)`` retourne la liste ordonnée
      des abréviations, avec doublons préservés et tolérance
      NFC/NFD.
    - ``compute_abbreviation_metrics(ref, hyp)`` produit deux
      scores complémentaires :
      - **strict_score** : taux d'abréviations Unicode dont la
        forme abrégée est préservée telle quelle (édition
        diplomatique).
      - **expansion_score** : taux d'abréviations dont SOIT la
        forme abrégée SOIT la forme développée attendue est
        présente (édition modernisée acceptée).
    - Le **ratio strict/expansion** dit la convention adoptée :
      si égal et proche de 1 → diplomatique ; si strict << expansion
      → modernisant ; les deux faibles → erreur OCR.
    - Frontière de mot exigée pour les expansions courtes (« et »,
      « us ») afin de limiter le bruit (« permettre » ne match
      pas « per »).
  - ``abbreviation_strict_score`` et ``abbreviation_expansion_score``
    enregistrés dans le registre typé Sprint 34 pour ``(TEXT, TEXT)``.
  - +23 tests dans `test_sprint56_abbreviations.py` couvrant la
    détection (Unicode + tilde combinant + duplications + NFD),
    les **trois scénarios standards** (diplomatique → strict=1,
    expansion=1 ; modernisant → strict=0, expansion=1 ;
    mauvais OCR → 0/0 ; mixte → strict=0.5, expansion=1), le
    breakdown per_abbreviation, les cas dégénérés, la frontière
    de mot pour les expansions courtes, l'intégration registre,
    et la sanité de la table d'expansions.

- **Sprint 55 — A.II.3.1 Précision par bloc Unicode : couche de
  calcul (démarrage A.II.3, métriques philologiques).** Pour un
  éditeur d'imprimés anciens ou un médiéviste, la question
  pertinente n'est pas seulement « quel CER global ? » mais
  « quels caractères historiques ce moteur restitue-t-il
  fidèlement ? ».  Le module produit la phrase actionnable du plan :
  *« GPT-4o restitue 100 % du Latin de Base mais 0 % des formes de
  présentation latine (ﬁ, ſ…) »*.
  - Nouveau module `picarones/core/unicode_blocks.py` :
    - Table de 22 blocs Unicode standard centrée sur les corpus
      patrimoniaux européens (Latin de Base, Latin Étendu A/B/C/D/E,
      Latin Extended Additional, Diacritiques combinants,
      Présentation latine, MUFI PUA, Greek, Cyrillic, etc.).
      Tout caractère hors table → ``"Other"`` (couverture
      exhaustive : ``sum(total) == len(GT)``).
    - ``get_block(char)`` retourne le nom du bloc Unicode.
    - ``compute_unicode_block_accuracy(reference, hypothesis)`` :
      aligne caractère par caractère via ``difflib.SequenceMatcher``,
      classe chaque caractère GT dans son bloc, et compte les
      positions correctement restituées (opcodes ``equal``).
      Retourne ``per_block`` (correct/total/accuracy par bloc) +
      ``global_accuracy``.
    - ``unicode_block_global_accuracy`` : raccourci enregistré dans
      le registre typé Sprint 34 pour ``(TEXT, TEXT)``.
  - +24 tests dans `test_sprint55_unicode_blocks.py` :
    - ``get_block`` sur 10 caractères clés (ASCII, é, ç, ƒ, ſ, ﬁ,
      diacritique combinant) + Other (vide, émoji)
    - calcul d'accuracy : identité, vide bilatéral / unilatéral,
      None, substitutions ciblées par bloc
    - **cas réaliste du plan** : OCR modernisant remplace ſ→s et
      ﬁ→fi → 100 % Latin de Base mais 0 % Présentation latine et
      0 % Latin Extended-A
    - insertions/suppressions, coverage exhaustive, raccourci,
      intégration registre

- **Sprint 54 — A.II.2.2 Layout F1 par type de région : couche de
  calcul (clôture A.II.2 côté calcul).** Dernière brique de l'axe
  A.II.2 (métriques structurelles).  Pour les manuscrits glosés
  (texte principal vs glose) ou les journaux multi-colonnes, c'est
  la métrique qui répond à *« le moteur sépare-t-il bien le texte
  principal de la glose ? »*.
  - Nouveau module `picarones/core/layout.py` :
    - dataclass `Region(id, type, bbox)` avec validation (bbox
      strictement positive)
    - `_iou_bbox` calcule l'IoU de deux rectangles (origine en haut
      à gauche, convention ALTO/PAGE)
    - `_align_regions` apparie GT ↔ hypothèse en greedy par IoU
      décroissant, **same type required** (case-insensitive),
      pattern identique au NER (Sprint 38)
    - `compute_layout_metrics(refs, hyps, iou_threshold=0.5)`
      retourne global F1 + per_type + listes
      ``missed_regions`` (FN) et ``hallucinated_regions`` (FP)
    - `layout_f1` : raccourci pour le F1 global
  - Conventions : seuil IoU par défaut à 0,5 (convention ICDAR),
    coercion automatique dict → ``Region``, comparaison de type
    insensible à la casse.
  - Pas d'enregistrement registre typé pour ce sprint — la métrique
    suppose un parsing préalable (extraction des régions avec types
    et bbox depuis l'ALTO/PAGE) qui ne s'inscrit pas directement
    dans le pattern `(ArtifactType, ArtifactType)`.  L'enregistrement
    suivra quand le parser ALTO standard sera livré.
  - +20 tests dans `test_sprint54_layout.py` (validation Region,
    IoU mathématique, cas standards : parfait, mauvais type,
    hallucination, FN, IoU sous/sur seuil, multi-type breakdown,
    alignement greedy avec best-IoU wins, dégénérés, type
    case-insensitive, shortcut).

- **Sprint 53 — A.II.2.1 Reading order F1 (ICDAR 2015) : couche de
  calcul.** Suite du Sprint 52 dans l'axe A.II.2 (métriques
  structurelles).  Sur un manuscrit glosé ou un journal multi-colonnes,
  un moteur peut avoir un excellent CER caractère et un ordre de
  lecture catastrophique — le CER seul ne capture pas cette
  dimension.
  - Nouveau module `picarones/core/reading_order.py` :
    - ``compute_reading_order_metrics(ref_order, hyp_order)`` :
      pour chaque paire ``(a, b)`` où ``a`` précède ``b`` dans la GT,
      vérifie si ``a`` précède aussi ``b`` dans l'hypothèse.  Retourne
      precision/recall/F1 + détails (TP/FP/FN, paires totales, régions
      communes vs disjointes).
    - ``reading_order_f1`` : raccourci qui retourne juste le F1.
  - Conventions : doublons traités à la première occurrence,
    séquences ``None``/vides → F1 = 0 (pas de récompense gratuite),
    séquence à 1 région → 0 paire émise → F1 = 0 (convention de bord).
  - Format compatible avec ``ReadingOrderGT.region_order`` du
    Sprint 32 — l'utilisateur fournit directement la liste d'IDs.
  - ``reading_order_f1`` enregistré dans le registre typé Sprint 34
    pour la jonction ``(READING_ORDER, READING_ORDER)``.
  - +16 tests dans `test_sprint53_reading_order.py` (cas canoniques :
    identique → F1=1, inversé → F1=0, permutation locale, insertion,
    suppression ; cas dégénérés : vide, single region, doublons,
    None ; comptages détaillés ; intégration registre typé).

- **Sprint 52 — A.II.2.3 Différence Flesch : couche de calcul
  (démarrage de l'Étape 3 / axe A — métriques structurelles).**
  Stratégie identique aux Sprints 35/38/39 (couche pure d'abord,
  câblage runner+HTML après).
  - Nouveau module `picarones/core/readability.py` :
    - ``count_syllables_word`` : heuristique groupes de voyelles
      consécutives (avec diacritiques FR/EN), fallback à 1 syllabe
      pour les mots sans voyelle (acronymes type « BNF »).
    - ``count_words`` (regex Unicode) et ``count_sentences``
      (découpe sur ``.!?…``, minimum 1 si le texte contient au
      moins un mot).
    - ``flesch_score(text, lang)`` avec coefficients FR
      (Kandel-Moles 1958, ``207 - 1.015·m/p - 73.6·s/m``) et EN
      (Flesch 1948, ``206.835 - 1.015·m/p - 84.6·s/m``).  Score
      borné dans ``[0, 100]``.
    - ``flesch_delta(reference, hypothesis, lang)`` retourne la
      différence ``Flesch(OCR) - Flesch(GT)``.  **Positif = signal
      d'over-normalisation LLM** (le LLM a lissé la langue
      historique).
  - **Aucun alignement caractère/mot requis** : la métrique reste
    calculable même quand l'OCR est très dégradé — c'est l'avantage
    clé pour repérer les VLM/LLM qui hallucinent du texte moderne
    plausible mais déconnecté de la GT.
  - `flesch_delta_fr` et `flesch_delta_en` enregistrés dans le
    registre typé Sprint 34 pour la jonction ``(TEXT, TEXT)``.
  - +25 tests dans `test_sprint52_readability.py` (compteurs de
    base avec cas limites, score Flesch borné, FR/EN cohérents,
    delta nul sur textes identiques, **cas réaliste de
    modernisation LLM** → delta > 10 pts, OCR dégradé borné,
    intégration registre typé).

- **Sprint 51 — Adapter Azure Document Intelligence : exposition de
  `Word.confidence` (clôture de l'adaptation engines).** Suite directe
  des Sprints 47-50. Azure DI expose ``analyzeResult.pages[].words[]``
  avec ``content`` et ``confidence`` ∈ [0, 1]. L'adapter parcourt cette
  hiérarchie et émet une entrée par mot au format Sprint 42.
  - Refactor : ``_run_ocr_with_result(image_path) → (text,
    analyze_result_dict)`` centralise les deux chemins (SDK
    ``azure-ai-documentintelligence`` et REST direct via ``urllib``
    avec polling Azure asynchrone).
  - ``_sdk_result_to_dict`` convertit l'objet SDK en dict normalisé
    identique à la réponse REST pour traitement uniforme.
  - ``_extract_token_confidences_from_result`` parcourt
    ``pages[].words[]``, extrait ``content`` et ``confidence``,
    filtre les confidences None / négatives et les contenus vides.
  - Le texte ``EngineResult.text`` est extrait depuis
    ``pages[].lines[]`` (préservation rétrocompat octet par octet).
  - Flag config ``expose_confidences: false``.
  - L'API est appelée une seule fois — aucun overhead.
  - +16 tests dans ``test_sprint51_azure_confidences.py`` (extraction
    multi-pages, filtrage 4 cas, cas dégénérés 4 cas, conversion SDK
    → dict, surcharge ``run()`` avec mock, échec API, intégration
    runner).

- **Sprint 50 — Adapter Google Vision : exposition de
  `Word.confidence`.** Suite directe des Sprints 47-49.
  ``DOCUMENT_TEXT_DETECTION`` expose ``Word.confidence`` au niveau mot
  sur ``page > block > paragraph > word`` ; l'adapter parcourt cette
  hiérarchie et émet une entrée par mot au format Sprint 42.
  - Refactor : ``_run_ocr_with_full_annotation(image_path) → (text,
    full_dict)`` centralise les deux chemins (SDK
    ``google-cloud-vision`` et REST direct via ``urllib``).
    ``_run_ocr`` reste rétrocompat (retourne juste le texte).
  - ``_sdk_full_text_to_dict`` convertit la réponse proto SDK en
    dict normalisé identique à la réponse REST, pour traitement
    uniforme.
  - ``_extract_token_confidences_from_full_text`` parcourt
    ``pages → blocks → paragraphs → words``, reconstruit chaque mot
    par concaténation des ``word.symbols[i].text``, et émet
    ``{"token": str, "confidence": float}`` (confidence ∈ [0, 1] —
    le runner Sprint 42 accepte directement ce format).
  - Filtrage cohérent avec les autres adapters : confidence None /
    négative → ignorée, mots vides → ignorés.
  - ``TEXT_DETECTION`` (mode "court") ne fournit pas de confidence
    par mot → ``token_confidences = None``.
  - Flag config ``expose_confidences: false``.
  - L'API est appelée une seule fois — aucun overhead.
  - +17 tests dans `test_sprint50_google_vision_confidences.py`
    (reconstruction depuis symbols, multi-pages/blocks, filtrage,
    flag, cas dégénérés, conversion SDK → dict, surcharge ``run()``
    avec mock du chemin réseau, REST avec urllib mocké, intégration
    runner).

- **Sprint 49 — Adapter Mistral OCR : exposition des
  `token_confidences` quand l'API les fournit.** Suite des Sprints
  47 (Tesseract) et 48 (Pero OCR). Mistral OCR a deux chemins :
  l'endpoint dédié `/v1/ocr` (modèle `mistral-ocr-latest`) qui peut
  exposer des champs `confidence` à différents niveaux, et l'API
  chat/vision (`pixtral-*`) qui ne fournit pas de confidences.
  - Refactor : nouvelle méthode `_run_ocr_with_response(image_path)`
    retourne `(text, raw_response)`. `_run_ocr_native_api` retourne
    désormais aussi le JSON brut. Le chemin chat/vision retourne
    `(text, None)` car aucune confidence n'est disponible.
  - `_extract_token_confidences_from_response` parse la réponse
    `/v1/ocr` en cascade :
    1. `pages[i].words[j]` avec `{"text", "confidence"}` →
       extraction directe
    2. `pages[i].lines[j]` avec `{"text", "confidence"}` →
       propagation de la confidence à chaque mot (pattern Pero
       Sprint 48)
    3. `pages[i].blocks[j]` → idem
  - Filtrage cohérent avec Tesseract/Pero : texte vide, confidence
    None, confidence négative → ignorés.
  - Si l'API ne retourne aucun champ `confidence` exploitable
    (cas courant si Mistral retourne uniquement du markdown), ou si
    on est sur le chemin chat/vision, `token_confidences = None`.
  - Nouveau paramètre config `expose_confidences: false` cohérent
    avec les autres adapters.
  - L'API est appelée **une seule fois** ; le coût est strictement
    identique à l'implémentation historique.
  - +17 tests dans `test_sprint49_mistral_confidences.py` couvrant
    l'extraction (words explicites, propagation lines/blocks,
    combinaison, filtrage texte vide / conf None / négative), les
    cas dégénérés (None, dict vide, pas de pages, markdown sans
    confidences, types invalides), le flag `expose_confidences=False`,
    la surcharge `run()` (mock du chemin réseau, chat/vision sans
    confidences, échec API), et l'intégration runner.

- **Sprint 48 — Adapter Pero OCR : exposition des `token_confidences`
  natifs.** Suite directe du Sprint 47 (Tesseract). Pero OCR fournit
  une confidence par ligne (``transcription_confidence``, probabilité
  CTC moyenne) ; l'adapter la propage à chaque mot de la ligne.
  - ``PeroOCREngine.run()`` surchargé : un seul appel
    ``parser.process_page`` produit le ``page_layout`` ; texte ET
    confidences en sont extraits sans coût supplémentaire (vs
    Tesseract qui doit faire deux appels distincts).
  - Refactor : ``_run_pero_pipeline(image_path) -> (text,
    page_layout)`` centralise l'appel au pipeline ; ``_run_ocr``
    devient un wrapper trivial pour rétrocompat.
  - ``_extract_token_confidences_from_layout`` parcourt
    regions/lines, applique ``transcription_confidence`` à chaque
    mot de la ligne, ignore les transcriptions vides / confidences
    None / confidences négatives, retourne ``None`` si aucune ligne
    n'avait de confidence exploitable.
  - Nouveau paramètre config ``expose_confidences: false`` (cohérent
    avec Tesseract Sprint 47).
  - Pipeline appelé une seule fois → **aucun overhead** par rapport
    à l'implémentation historique (vs un appel supplémentaire pour
    Tesseract).
  - +14 tests dans ``test_sprint48_pero_confidences.py`` couvrant :
    extraction depuis layout (tokens uniques, multi-lignes,
    transcription vide, confidence None / négative), flag
    ``expose_confidences=False``, cas dégénérés (None / regions
    vides / aucune confidence), surcharge ``run()`` (texte préservé
    octet par octet, échec du pipeline), intégration runner avec
    ``calibration_metrics`` correctement calculée, fallback gracieux
    quand pero-ocr est absent.

- **Sprint 47 — Adapter Tesseract : exposition des `token_confidences`
  natifs.** Premier des engines adaptés au câblage calibration
  (Sprint 42). L'utilisateur qui benchmarke avec Tesseract obtient
  désormais automatiquement ECE/MCE et reliability diagram dans le
  rapport, sans configuration supplémentaire.
  - `TesseractEngine.run()` est surchargé : appelle `image_to_string`
    pour le texte (rétrocompat octet par octet) **et** `image_to_data`
    pour les confidences mot par mot, retourne un `EngineResult` avec
    `token_confidences = [{"token": str, "confidence": float}, …]`
    (confidence ∈ [0, 100], le runner Sprint 42 normalise en [0, 1]).
  - Helper `_extract_token_confidences()` séparé du chemin OCR
    principal : si `image_to_data` lève, l'OCR continue normalement
    et `token_confidences = None` (warning explicite, pas
    `except: pass`).
  - Filtrage à la source : non-mots Tesseract (conf = -1), tokens
    vides, longueurs incompatibles → ignorés.
  - Nouveau paramètre config `expose_confidences: false` pour
    désactiver le second appel Tesseract (économie d'un appel par
    image en cas de besoin).
  - Coût additionnel : un appel `image_to_data` par image. Le texte
    de `image_to_string` n'est jamais reconstruit depuis
    `image_to_data` — préservation stricte du comportement
    historique.
  - +9 tests dans `test_sprint47_tesseract_confidences.py` couvrant
    l'exposition des confidences (avec mock pytesseract), la
    préservation octet par octet du texte, le flag
    `expose_confidences=False`, le fallback gracieux quand
    `image_to_data` lève (warning + `None`), le filtrage des
    non-mots/longueurs incompatibles, l'intégration bout-en-bout
    avec le runner (`calibration_metrics` calculé), et le cas
    pytesseract absent.

- **Sprint 46 — A.III stratification par `script_type` : vue HTML +
  détecteur narratif (clôture A.III)**. Suite directe du Sprint 45
  (couche backend). La vue stratifiée est désormais rendue dans le
  rapport et un détecteur signale automatiquement les corpus
  hétérogènes.
  - Nouveau module `picarones/report/stratification_render.py` :
    `build_stratified_ranking_html` rend un `<details>` natif
    (collapsible sans JS) par strate avec tableau moteur × (médiane,
    moyenne, docs). Cellule médiane colorée par gradient vert (faible
    CER) → rouge (élevé). Premier `<details>` ouvert par défaut pour
    donner le contexte. Bandeau d'avertissement en tête si
    `corpus_homogeneity` fourni (écart inter-strate du leader).
  - `_build_report_data` expose `available_strata`,
    `stratified_ranking`, `corpus_homogeneity` au top-level. Le bloc
    HTML est passé au template `view_ranking.html` qui l'insère après
    le tableau principal **uniquement si stratification disponible**
    (rapport adaptatif).
  - Nouveau `FactType.STRATIFICATION_RECOMMENDED` (priority 45,
    importance MEDIUM ou HIGH selon le gap) avec détecteur
    `detect_stratification_recommended` qui lit `corpus_homogeneity`
    et émet un Fact quand le gap inter-strate du leader dépasse
    5 points de CER (HIGH au-delà de 10 points). Templates FR/EN
    sans nombres en dur.
  - L'arbitre marque la paire `{GLOBAL_LEADER_CER,
    STRATIFICATION_RECOMMENDED}` comme **complémentaire** : la
    recommandation peut cohabiter avec la phrase du leader pour
    nuancer.
  - +8 clés i18n FR/EN pour la vue stratifiée
    (`stratification_caption`, `stratification_description`,
    `stratification_*_label`, `stratification_gap_summary`).
  - Anti-injection HTML via `html.escape` sur les noms de moteurs et
    les noms de strates.
  - +38 tests dans `test_sprint46_stratification_html.py` couvrant
    le rendu (un `<details>` par strate, métriques visibles, premier
    ouvert), le bandeau d'hétérogénéité, le masquage adaptatif (4
    cas), l'anti-injection (engine et stratum avec balises HTML),
    les seuils du détecteur (4 cas), la traçabilité
    anti-hallucination FR + EN, l'absence de chiffres en dur dans
    les templates, l'intégration `ReportGenerator` FR + EN, et la
    complétude i18n.

- **Sprint 45 — A.III stratification par `script_type` : couche
  d'agrégation backend.** Première brique de la « plus haute valeur
  ajoutée transversale » du plan d'évolution. Le rapport peut
  désormais classer les moteurs **par strate** (manuscrit gothique,
  cursive administrative, imprimé ancien, humanistique…) — la moyenne
  globale ment quand le corpus est hétérogène.
  - `BenchmarkResult.doc_strata: Optional[dict[str, str]]` :
    map ``{doc_id: script_type}`` capturée par le runner avant
    ``compact()`` (qui efface ``image_quality``).
  - `BenchmarkResult.available_strata()` : liste triée des strates
    distinctes, ignore les valeurs vides.
  - `BenchmarkResult.stratified_ranking()` : retourne
    ``{stratum: [ranking_entry]}`` avec mean/median CER **recalculés
    par strate**, tri par médiane (cohérent avec Sprint 44). Inclut
    les moteurs sans aucun doc dans une strate sous forme d'entrée
    dégénérée (mean/median = None).
  - `BenchmarkResult.corpus_homogeneity()` : pour le moteur leader
    global, retourne l'écart inter-strate de la médiane CER et
    identifie la paire de strates min/max — base du futur
    avertissement automatique « ce corpus est hétérogène,
    consultez la vue stratifiée ».
  - `as_dict()` expose `doc_strata`, `available_strata`,
    `stratified_ranking`, `corpus_homogeneity` quand renseignés
    (rétrocompat stricte sinon — clés absentes).
  - Le runner peuple `doc_strata` avant compact en lisant
    ``DocumentResult.image_quality["script_type"]``.
  - +16 tests dans `test_sprint45_stratification.py` couvrant les
    fields, available_strata, stratified_ranking (1 entrée/moteur/
    strate, métriques per-strate, tri par médiane, moteurs absents),
    corpus_homogeneity (None < 2 strates, calcul d'écart),
    sérialisation as_dict, et un **test propriété réaliste** : le
    leader global peut perdre sur une strate (Tesseract domine
    globalement mais perd sur le manuscrit où Pero gagne).

- **Sprint 44 — A.I.2 : tri par médiane CER par défaut + détecteur
  d'asymétrie.** Réponse à la critique structurelle 2 du plan
  d'évolution : sur les corpus patrimoniaux, la moyenne est facilement
  tirée par quelques documents catastrophiques et masque les
  performances réelles ; la médiane est plus représentative.
  - `EngineReport.median_cer` : nouvelle propriété qui lit
    `aggregated_metrics["cer"]["median"]`.
  - `BenchmarkResult.ranking()` :
    - inclut désormais `median_cer` dans chaque entrée (additif)
    - **trie par médiane CER croissante par défaut** (et non plus
      par moyenne)
    - retombe sur `mean_cer` quand `median_cer` est absent
      (rétrocompat pour le cas pathologique)
  - Nouveau `FactType.MEDIAN_MEAN_GAP_WARNING` et détecteur
    `detect_median_mean_gap_warning` (priority 140) : émet un Fact
    quand `|mean - median| / median > 30 %` pour le moteur leader.
    Importance MEDIUM par défaut, HIGH si gap relatif ≥ 100 %.
    Garde-fou : ne déclenche pas si la médiane est nulle.
  - Templates FR/EN — aucun nombre en dur, tout vient du payload
    (vérifié par test).
  - L'arbitre marque la paire `{GLOBAL_LEADER_CER,
    MEDIAN_MEAN_GAP_WARNING}` comme **complémentaire** : les deux
    phrases peuvent coexister dans la synthèse pour nuancer le
    leader.
  - +15 tests dans `test_sprint44_median_default.py` (propriété
    median_cer, tri par médiane sur cas asymétrique réaliste,
    fallback sur la moyenne, déclenchement du détecteur sur 4 cas
    dégénérés, importance MEDIUM/HIGH selon gap, traçabilité
    anti-hallucination FR + EN, intégration via build_synthesis).

- **Sprint 43 — A.II.1.b Calibration : vue HTML reliability diagram +
  tableau ECE/MCE (clôture A.II.1.b côté rapport).** Suite directe du
  Sprint 42 (câblage runner). Les chiffres de calibration sont
  désormais visibles dans le rapport HTML.
  - Nouveau module `picarones/report/calibration_render.py` :
    - `build_calibration_summary_html(engines_summary, labels)` :
      tableau résumé par moteur (ECE, MCE, Précision moyenne,
      Confiance moyenne, n_predictions, doc_count). Cellule ECE
      colorée par gradient vert (bien calibré) → rouge (mal
      calibré).
    - `build_reliability_diagram_svg(aggregated_calibration, labels,
      engine_name)` : SVG d'un reliability diagram avec barres
      d'accuracy par bin, ligne reliant les points
      `(avg_confidence, accuracy)`, diagonale de référence en
      pointillé, axes annotés (graduations 0/0.5/1).
    - `build_reliability_diagrams_grid_html(engines_summary,
      labels)` : grille auto-fit, un SVG par moteur ayant un
      `aggregated_calibration`.
    - Rendu strictement server-side, déterministe, **pas de
      JavaScript**, cohérent avec le SVG du CDD (Sprint 18) et les
      sections inter-moteurs (Sprint 37) et NER (Sprint 41).
  - `_build_report_data` expose `aggregated_calibration` par moteur
    dans `engines_summary`. `ReportGenerator.generate` calcule les
    deux blocs et les passe au template `view_analyses.html` qui les
    affiche dans une `chart-card` à largeur pleine **uniquement si
    au moins un moteur a un `aggregated_calibration`** (rapport
    adaptatif).
  - +13 clés i18n FR/EN (`h_calibration`, `calibration_note`,
    `calibration_summary_caption`, `calibration_engine_label`,
    `calibration_ece_label`, `calibration_mce_label`,
    `calibration_n_label`, `calibration_acc_label`,
    `calibration_conf_label`, `calibration_docs_label`,
    `reliability_diagram_title`, `reliability_x_axis`,
    `reliability_y_axis`).
  - +43 tests dans `test_sprint43_calibration_html.py` couvrant le
    rendu (résumé, SVG avec barres/points/diagonale, grille
    multi-moteurs), le masquage adaptatif (4 cas dégénérés),
    l'anti-injection (engine name `<script>` ou `<img>`),
    l'intégration rapport FR + EN, et la complétude i18n sur les
    13 clés × 2 langues.

- **Sprint 42 — A.II.1.b Calibration : exposition `token_confidences` +
  câblage runner.** Suite directe du Sprint 39 (couche de calcul). Le
  runner peut maintenant calculer ECE/MCE/reliability dès qu'un moteur
  expose des confidences au niveau token.
  - `EngineResult.token_confidences: Optional[list[dict[str, Any]]]`
    ajouté. Format attendu : `[{"token": str, "confidence": float}, …]`,
    confidence ∈ [0, 1] ou ∈ [0, 100] (normalisé par le runner).
    `None` par défaut → comportement strictement rétrocompat pour tous
    les adapters historiques (Tesseract, Pero, Mistral OCR, Google
    Vision, Azure DI). L'adaptation de chaque adapter à exposer ses
    confidences natives est reportée à des sprints dédiés (un par
    adapter).
  - `DocumentResult.calibration_metrics: Optional[dict]` ajouté
    (sérialisé dans `as_dict` quand renseigné, libéré par `compact()`).
  - `EngineReport.aggregated_calibration: Optional[dict]` ajouté.
  - Helper `_calibration_from_engine_result(ground_truth, token_confidences)` :
    aligne par bag-of-words avec multiplicité (proxy oracle, comme
    `oracle_token_recall` du Sprint 35), normalise les confidences en
    pourcentage à `[0, 1]`, ignore les confidences négatives
    (Tesseract met -1 pour les non-mots), retourne `None` sur entrée
    vide. Appelé dans `_compute_document_result` quand
    `EngineResult.token_confidences` est non-vide.
  - Helper `_aggregate_calibration(doc_results)` : combine les bins de
    tous les docs en somme pondérée par count, recalcule ECE/MCE micro
    sur l'ensemble. Renvoie `None` si aucun doc n'a de
    `calibration_metrics`.
  - +17 tests dans `test_sprint42_calibration_runner.py` couvrant le
    nouveau champ EngineResult, la sérialisation et compact des
    nouveaux champs DR/ER, l'helper d'alignement (calibration parfaite,
    normalisation %, skip négatifs, bag-of-words avec multiplicité,
    skip entrées invalides), l'agrégateur (combinaison de bins
    multi-docs, recalcul ECE/MCE micro), et la rétrocompat
    (pas de calcul sans token_confidences).

- **Sprint 41 — A.II.1.a NER : vue HTML dédiée (clôture A.II.1.a).**
  Suite directe des Sprints 38-40. Le moteur narratif et le runner ont
  déjà tout ce qu'il faut ; ce sprint rend les chiffres visibles et
  vérifiables dans le rapport.
  - Nouveau module `picarones/report/ner_render.py` :
    - `build_ner_summary_html(engines_summary, labels)` : tableau
      résumé F1 global / Précision / Rappel / Docs évalués /
      Hallucinations / Missed par moteur, cellule F1 colorée par
      gradient rouge → jaune → vert.
    - `build_ner_per_category_html(engines_summary, labels)` : heatmap
      moteur × catégorie d'entité (PER, LOC, DATE, ORG, MISC…).
      Cellules colorées par F1, cellule vide marquée d'un `—` pour
      les catégories non observées chez ce moteur. Tooltip `support=N`
      sur chaque cellule.
    - Rendu strictement serveur-side, déterministe, **pas de
      JavaScript**.
    - Anti-injection : noms de moteurs et labels de catégories passés
      à `html.escape`.
  - `_build_report_data` expose `aggregated_ner` par moteur dans
    `engines_summary`. `ReportGenerator.generate` calcule les deux
    blocs HTML et les passe au template `view_analyses.html` qui les
    affiche dans une `chart-card` à largeur pleine **uniquement si au
    moins un moteur a un `aggregated_ner`** (principe du rapport
    adaptatif).
  - +12 clés i18n FR/EN (`h_ner`, `ner_note`, `ner_summary_caption`,
    `ner_per_category_caption`, `ner_engine_label`, `ner_f1_label`,
    `ner_precision_label`, `ner_recall_label`, `ner_doc_count_label`,
    `ner_hallucinated_label`, `ner_missed_label`, `ner_no_data_label`).
  - +38 tests dans `test_sprint41_ner_html.py` couvrant le rendu
    (résumé, heatmap, multi-moteurs, union des catégories, cellule
    vide), le masquage adaptatif (3 cas dégénérés), l'anti-injection
    (engine et label avec balises HTML), l'intégration rapport (FR +
    EN), et la complétude i18n sur les 12 clés × 2 langues.

- **Sprint 40 — A.II.1.a NER : backend extracteur + câblage runner.**
  Suite directe du Sprint 38 (couche de calcul pure). Le runner peut
  maintenant calculer les métriques NER de bout-en-bout quand le corpus
  porte une GT entités (`EntitiesGT` du Sprint 32).
  - Nouveau module `picarones/core/ner_backends.py` :
    - `EntityExtractor` (Protocol) : tout callable
      `(text) -> list[dict]` est un extracteur valide.
    - `SpacyEntityExtractor(model_name, label_mapping=None)` : lazy-import
      de spaCy, charge le modèle au premier appel, met en cache. Si
      spaCy absent OU modèle non téléchargé, fallback gracieux silencieux
      (retourne `[]`) avec **warning explicite** au premier appel
      (cf. règle CLAUDE.md). Mapping par défaut spaCy → conventions HIPE
      (PERSON → PER, GPE → LOC, TIME → DATE, etc.).
    - `SPACY_PROFILES` : 6 profils nommés (fr, fr_lg, en, en_lg,
      multilingual, hipe).
    - `get_extractor(profile)` : factory qui accepte clé de profil ou
      nom de modèle direct.
    - `is_spacy_available()` : test sans charger de modèle.
  - `DocumentResult.ner_metrics: Optional[dict]` ajouté ; sérialisé
    dans `as_dict()` quand renseigné, libéré par `compact()`.
  - `EngineReport.aggregated_ner: Optional[dict]` ajouté avec micro-F1
    global recalculé à partir des sommes TP/FP/FN, détail par catégorie,
    totaux d'hallucinations et missed.
  - `runner.run_benchmark` accepte un nouveau paramètre optionnel
    `entity_extractor`. Si fourni, le runner appelle deux helpers en
    post-process (main process, **pas** dans les sous-processus pour
    éviter de pickler spaCy) :
    - `_attach_ner_metrics(corpus, doc_results, extractor)` : pour
      chaque doc avec `GTLevel.ENTITIES`, extrait les entités sur
      l'hypothèse et calcule `compute_ner_metrics`.
    - `_aggregate_ner(doc_results)` : agrège au niveau du moteur
      (micro-F1 + per_category + totaux).
    Les exceptions par doc sont dégradées en warning, le benchmark
    continue.
  - **Rétrocompat stricte** : sans `entity_extractor`, aucun calcul
    NER n'a lieu, aucun champ n'est ajouté, le rapport reste identique.
  - Nouveau extra `[ner]` dans `pyproject.toml` (`spacy>=3.7.0`) — non
    installé par défaut.
  - +16 tests dans `test_sprint40_ner_runner.py` couvrant le fallback
    sans spaCy + warning, l'idempotence du load, les profils + factory,
    la sérialisation des nouveaux champs (omis quand None, présents
    quand renseignés, libérés par compact), le câblage runner avec un
    extracteur mock injecté, l'agrégation micro-F1, la rétrocompat
    sans extracteur, et la robustesse à un extracteur qui lève.

- **Sprint 39 — A.II.1.b Calibration des moteurs : couche de calcul.**
  Deuxième brique des trois métriques prioritaires de l'Étape 2 (axe A —
  fiabilité). Stratégie identique aux Sprints 35-38 : couche de calcul
  pure, exposition des `token_confidences` sur les `EngineResult` et
  câblage runner+narratif+HTML aux sprints suivants.
  - Nouveau module `picarones/core/calibration.py` :
    - dataclass `CalibrationBin(bin_low, bin_high, avg_confidence,
      accuracy, count)` avec propriété `gap` (renvoie `None` si bin vide)
    - `reliability_diagram(confidences, is_correct, n_bins=10)` : binning
      équidistant de la confiance, calcul de la précision moyenne et de
      la confiance moyenne par bin
    - `expected_calibration_error` (ECE) : moyenne pondérée par bin de
      `|conf - accuracy|`, ∈ [0, 1], 0 = calibration parfaite
    - `maximum_calibration_error` (MCE) : pire écart sur tous les bins
      non vides
    - `compute_calibration_metrics` : vue agrégée
  - **Calcul d'index de bin par multiplication** (`int(c * n_bins)`)
    plutôt que division, pour éviter les pièges IEEE 754 (`0.6 / 0.1 =
    5.999…` en flottant). Cas testé.
  - Aucune dépendance externe ; les listes `confidences` et `is_correct`
    sont fournies en entrée. L'extraction depuis les engines existants
    (Tesseract `tsv`, Pero `PageLayout`, Mistral `confidence`, Google
    Vision `Word.confidence`) est explicitement reportée à un sprint
    dédié.
  - +32 tests dans `test_sprint39_calibration.py` couvrant la
    calibration parfaite (ECE = 0), les cas extrêmes (sur-confiance et
    sous-confiance → ECE = 0,5), le biais constant (ECE = `|conf - acc|`),
    le binning correct (bornes équidistantes, c=1.0 dans le dernier bin,
    affectation correcte y compris pour 0.6), les bins vides
    (avg/accuracy/gap = `None`), les listes vides, les garde-fous
    (longueurs incompatibles, conf hors [0, 1], n_bins ≤ 0), `n_bins`
    paramétrable + monotonie « ECE ne décroît pas avec un binning plus
    fin ».

- **Sprint 38 — A.II.1.a NER : couche de calcul.** Première brique
  des trois métriques prioritaires de l'Étape 2 du plan d'évolution
  (axe A — utilité aval). Stratégie de découpage analogue à la
  divergence taxonomique (Sprints 35-37) : couche de calcul d'abord,
  câblage runner+narratif+HTML aux sprints suivants.
  - Nouveau module `picarones/core/ner.py` :
    - dataclass `Entity(label, start, end, text)` avec validation de
      span ;
    - `compute_ner_metrics(reference_entities, hypothesis_entities,
      iou_threshold=0.5)` qui aligne par chevauchement IoU
      (greedy, IoU décroissant, chaque entité matchée une seule fois)
      et retourne precision/recall/F1 globaux + par catégorie, plus
      les listes des entités hallucinées (FP) et manquées (FN) ;
    - format de dict compatible `EntitiesGT` du Sprint 32 (clé
      `{label, start, end, text}`).
  - Métrique `ner_f1` enregistrée dans le registre typé Sprint 34
    pour la jonction `(ENTITIES, ENTITIES)` — peut être appelée par
    `compute_at_junction` aussi simplement que `cer` sur `(TEXT, TEXT)`.
  - Aucune dépendance externe (pas de spaCy ni Stanza dans ce sprint) :
    les listes d'entités sont fournies en entrée. Le backend extracteur
    suit dans un sprint dédié.
  - +19 tests dans `test_sprint38_ner_metrics.py` (cas standard parfait/FN/FP,
    label case-insensitive, IoU sous/sur seuil, custom threshold,
    multi-catégorie, alignement greedy avec une seule entité matchée,
    best-IoU wins, cas dégénérés vide/asymétrique, validation Entity,
    intégration registre).

- **Sprint 37 — Section inter-moteurs dans le rapport HTML.** Suite directe
  des Sprints 35-36 : la couche de calcul est maintenant visible côté
  rapport.
  - Nouveau module `picarones/report/inter_engine_render.py` :
    `build_divergence_matrix_html` (table HTML colorée — heatmap CSS
    inline, gradient blanc → rouge proportionnel au max hors-diagonale,
    diagonale étiquetée et grisée, paire la plus divergente annoncée
    en sous-titre) et `build_oracle_gap_html` (encart factuel avec
    best engine, recall préservé, oracle, gap absolu/relatif, doc count).
    Rendu strictement serveur-side, **pas de JavaScript** — déterministe
    comme le SVG du CDD (Sprint 18).
  - `ReportGenerator.generate` calcule les deux blocs et les passe au
    template `view_analyses.html` qui les insère dans une nouvelle
    `chart-card` à largeur pleine, **uniquement si présents**
    (principe du rapport adaptatif : moins de 2 moteurs ou taxonomie
    absente → section omise sans laisser de trace).
  - +14 clés i18n FR/EN (`h_inter_engine`, `inter_engine_note`,
    `divergence_*`, `oracle_*`).
  - Anti-injection HTML : tous les noms de moteurs et étiquettes sont
    passés à `html.escape` avant insertion.
  - +42 tests dans `test_sprint37_inter_engine_html.py` (rendu de la
    matrice avec valeurs et paire max, masquage adaptatif sur 4 cas
    dégénérés, anti-injection sur engine name `<script>`, intégration
    rapport FR + EN, complétude i18n sur les 14 clés × 2 langues).

- **Sprint 36 — Câblage inter-moteurs au runner et au moteur narratif.**
  Suite directe du Sprint 35 (couche de calcul) :
  - `picarones/core/inter_engine.py` gagne `compute_inter_engine_analysis`
    qui agrège la complémentarité doc par doc (oracle global + recall par
    moteur + per_doc top 50 trié par gap), et la divergence taxonomique
    (matrice + paire la plus divergente).
  - `BenchmarkResult.inter_engine_analysis: Optional[dict]` exposé dans
    `as_dict()` quand renseigné, absent sinon (rétrocompat stricte).
  - `runner.run_benchmark` collecte les hypothèses brutes par moteur
    avant `compact()`, calcule l'analyse inter-moteurs si ≥ 2 moteurs,
    et stocke le résultat sur le `BenchmarkResult`.
  - Nouveau `FactType.ENSEMBLE_OPPORTUNITY` + détecteur
    `detect_ensemble_opportunity` (priority 130, importance MEDIUM/HIGH
    selon `relative_gap`). Seuil de déclenchement à 25 % du gap relatif
    pour ne pas bruiter la synthèse.
  - Templates FR et EN ajoutés à `narrative/templates/{fr,en}.yaml` —
    aucun nombre en dur (vérifié par test), tout vient du payload.
  - `_build_report_data` du générateur HTML expose
    `report_data["inter_engine_analysis"]` afin que le détecteur le
    voie en mode "rapport".
  - +22 tests dans `tests/test_sprint36_ensemble_narrative.py`
    (agrégation, exposition `BenchmarkResult.as_dict`, déclenchement et
    seuils du détecteur, fallback paire sans taxonomie, intégration
    `build_synthesis` FR + EN, traçabilité anti-hallucination,
    template sans chiffres en dur).

- **Sprint 35 — Étape 2 du plan d'évolution : métriques inter-moteurs
  (couche de calcul).** Nouveau module `picarones/core/inter_engine.py`
  qui expose deux familles de mesures qui ne dépendent que des données
  déjà produites par le runner :
  - **Divergence taxonomique** : `kl_divergence`,
    `jensen_shannon_divergence` (symétrique, bornée dans `[0, 1]`),
    `taxonomy_divergence_matrix` qui construit la matrice triangulaire
    inter-moteurs sur les distributions de classes d'erreur (issues de
    `taxonomy.py`). Lissage epsilon des zéros pour éviter `log(0)`.
  - **Complémentarité** : `oracle_token_recall` (borne supérieure
    bag-of-words du recall atteignable par voting), `complementarity_gap`
    qui retourne aussi `best_single_recall` / `absolute_gap` /
    `relative_gap` / `best_engine`, `pairwise_disagreement_rate` pour
    quantifier le potentiel d'ensemble entre deux moteurs spécifiques.
  - Fonctions pures, sans I/O ni intégration runner — la couche de calcul
    est livrable indépendamment ; le câblage au moteur narratif
    (`ENSEMBLE_OPPORTUNITY`) et au rapport HTML (matrice de divergence,
    badge oracle gap) suit au Sprint 36.
  - +27 tests dans `tests/test_sprint35_inter_engine.py` couvrant les
    invariants mathématiques (KL ≥ 0, KL(p,p) = 0, JS symétrique et
    bornée, oracle ≥ best_single), les cas concrets (moteurs
    spécialisés ressortent comme candidats à un ensemble, complémentarité
    parfaite atteint oracle = 1), les garde-fous (référence vide,
    hypothèses vides, métrique inconnue).

- **Sprint 34 — Phase 0.3 : registre typé de métriques (clôture Phase 0).**
  Nouveaux modules `picarones/core/metric_registry.py` et
  `picarones/core/builtin_metrics.py` :
  - `MetricSpec` (dataclass figée) déclare `name`, `func`,
    `input_types: tuple[ArtifactType, ArtifactType]`, `description`,
    `higher_is_better`, `tags`
  - Décorateur `@register_metric(name=..., input_types=..., ...)`
    enregistre une métrique dans un registre global ; double
    enregistrement avec le même nom interdit, signature non-paire rejetée
  - `select_metrics(input_types)` retourne les métriques applicables à
    une jonction
  - `compute_at_junction(reference, hypothesis, input_types)` calcule
    toutes les métriques sélectionnées et tolère les erreurs unitaires
    (`logger.warning`, jamais `except: pass`)
  - `builtin_metrics.py` enregistre `cer`, `wer`, `mer`, `wil` sur
    `(TEXT, TEXT)` plus le stub `text_preservation_after_reconstruction`
    sur `(TEXT, ALTO)` comme preuve de concept de jonction hétérogène
  - **Approche additive stricte** : ni `metrics.py` ni `compute_metrics`
    ne sont modifiés ; le rapport HTML existant reste strictement
    identique octet par octet
  - +21 tests dans `tests/test_sprint34_metric_registry.py` couvrant
    l'enregistrement, la sélection par signature exacte, la résilience
    aux erreurs (`skip_on_error`), la **parité numérique** avec
    `compute_metrics` legacy sur 4 paires de textes (CER/WER/MER/WIL
    identiques à 1e-9 près), les garde-fous (double enregistrement,
    arité), et le stub TEXT→ALTO

- **Sprint 33 — Phase 0.2 : interface module générique.** Création de
  `picarones/core/modules.py` :
  - Enum `ArtifactType` (IMAGE, TEXT, ALTO, PAGE, ENTITIES, READING_ORDER) —
    valeurs string alignées sur `GTLevel` pour conversion triviale
  - Classe abstraite `BaseModule` avec `input_types`/`output_types`
    déclaratifs, `execution_mode: "io"|"cpu"`, méthode `process` typée
    `dict[ArtifactType, Any] → dict[ArtifactType, Any]`, helpers
    `validate_inputs`/`validate_outputs`, `metadata()` libre
  - `BaseOCREngine` hérite désormais de `BaseModule` avec
    `input_types=(IMAGE,)`, `output_types=(TEXT,)`. Sa nouvelle méthode
    `process` wrappe l'API historique `run()`. Aucun adaptateur OCR
    existant (Tesseract, Pero, Mistral OCR, Google Vision, Azure DI) n'est
    touché — le test_engines.py passe sans modification.
  - +23 tests dans `tests/test_sprint33_module_interface.py` couvrant le
    contrat (instanciation, validation I/O, repr), un `TextToAltoMock`
    démonstratif (TEXT→ALTO, critère explicite du plan), la délégation
    `BaseOCREngine.process → run`, et la cohérence ArtifactType/GTLevel.

- **Sprint 32 — Phase 0.1 : modèle de données GT multi-niveaux.**
  Refonte de `picarones/core/corpus.py` :
  - Enum `GTLevel` (TEXT, ALTO, PAGE, ENTITIES, READING_ORDER)
  - Payloads typés `TextGT`, `AltoGT`, `PageGT`, `EntitiesGT`,
    `ReadingOrderGT` avec `source_path` traçable
  - Champ `Document.ground_truths: dict[GTLevel, GTPayload]` synchronisé
    automatiquement avec le champ historique `ground_truth: str`
    (rétrocompatibilité stricte — toute API publique inchangée)
  - Détection automatique au chargement des fichiers `.gt.alto.xml`,
    `.gt.page.xml`, `.gt.entities.json`, `.gt.reading_order.json` à
    côté de l'image
  - `Corpus.gt_level_coverage()` et `Corpus.available_gt_levels` pour
    interroger la couverture par niveau
  - Erreurs de parse dégradées en `logger.warning` (CLAUDE.md :
    pas de `except: pass`) — le document conserve les niveaux qui ont
    pu être chargés
  - +17 tests dans `tests/test_sprint32_multi_level_gt.py` couvrant
    rétrocompat, détection, couverture partielle, synchronisation
    bidirectionnelle, JSON cassé

### Tests

- 1478 → 2086 tests (+17 Sprint 32, +23 Sprint 33, +21 Sprint 34,
  +27 Sprint 35, +22 Sprint 36, +42 Sprint 37, +19 Sprint 38,
  +32 Sprint 39, +16 Sprint 40, +38 Sprint 41, +17 Sprint 42,
  +43 Sprint 43, +15 Sprint 44, +16 Sprint 45, +38 Sprint 46,
  +9 Sprint 47, +14 Sprint 48, +17 Sprint 49, +17 Sprint 50,
  +16 Sprint 51, +25 Sprint 52, +16 Sprint 53, +20 Sprint 54,
  +24 Sprint 55, +23 Sprint 56, +41 Sprint 57). Aucune régression. **Phase 0 close ; Étape 2
  intégralement livrée ; Étape 3 / axe A.II.2 (métriques
  structurelles) couches de calcul intégralement livrées
  (Sprints 52-54) ; Étape 3 / axe A.II.3 (métriques philologiques)
  démarrée :** Précision par bloc Unicode (A.II.3.1, Sprint 55).

---

## [1.1.x] — Sprints 23-30 — 2026-04

### Ajouté

- **Sprint 23** — intégrité anti-hallucination du moteur narratif :
  whitelist `{"95", "100"}` vidée, `confidence_level=95` propagé dans
  `CONFIDENCE_WARNING`, `cost_unit_pages=1000` propagé dans
  `PARETO_ALTERNATIVE`/`COST_OUTLIER`, paramètre `select_facts(..., type_order=...)`,
  test stabilité bootstrap (±0,5 pp inter-seeds), test E2E synthèse EN.
  Doc « Politique éditoriale » dans `docs/explanation/narrative-engine.md`.
- **Sprint 24** — durcissement sécurité institutionnelle : mode public
  (`PICARONES_PUBLIC_MODE=1`), `PICARONES_BROWSE_ROOTS`, validation Pillow
  sur upload (CVE-2023-50447), rate limit + sémaphore concurrence,
  middleware CSP + en-têtes durcis, `SECURITY.md` à la racine.
- **Sprint 25** — refactor frontend en Jinja2 : `_HTML_TEMPLATE` (3000 L)
  → 8 partials `picarones/web/templates/` + `static/web-app.js`. CSP
  durcie en partie (script externalisé).
- **Sprint 26** — persistance jobs SQLite : `picarones/core/jobs.py`,
  `JobStore` thread-safe (WAL), `BenchmarkJob` persiste chaque event,
  endpoint SSE supporte `Last-Event-ID`, jobs orphelins marqués
  `interrupted` au boot, fallback DB sur `/api/benchmark/{id}/status`.
- **Sprint 27** — snapshots de reproductibilité dans le rapport HTML :
  `picarones/report/snapshot.py` embarque YAML brut de `pricing.yaml`,
  glossaire trié, profil de normalisation, version Picarones+Python+
  commit+deps figées.
- **Sprint 28** — UX : save/load config (`/api/config/save|load`),
  comparaison de runs (`picarones compare`, exit code 2 si régression),
  synthesis preview (`/api/benchmark/{id}/synthesis_preview`),
  `/api/history/regressions` qui surface l'infra Sprint 8.
- **Sprint 29** — registre déclaratif des détecteurs narratifs :
  `@register_detector(fact_type, priority, importance)` ;
  `DEFAULT_TYPE_ORDER` dérivé du registre. Ajouter un détecteur passe
  de 4 fichiers à 2.
- **Sprint 30** — polish/accessibilité/DX : `.pre-commit-config.yaml`
  avec ruff + check YAML/JSON/secrets, badges CER WCAG (icône + bordure
  pattern + `aria-label`), `i18n.py` thread-safe avec `lru_cache`,
  `_safe_version` log la stacktrace en DEBUG, backport CHANGELOG
  Sprints 10-22, mise à jour SPECS pour narrative/Pareto/glossaire.

### Tests

- 1242 → 1426 tests (+184 sur les Sprints 23-30).

---

## [1.0.x] — Sprints 10-22 — 2025-04 → 2026-03

### Ajouté

- **Sprint 10** — distribution erreurs par ligne (Gini, percentiles)
  dans `picarones/core/line_metrics.py`, détection hallucinations VLM
  dans `picarones/core/hallucination.py` (anchor score, length ratio).
- **Sprint 11** — internationalisation FR/EN, profils de normalisation
  anglais (`early_modern_english`, `medieval_english`, `secretary_hand`).
- **Sprint 12** — upload ZIP depuis le navigateur, filtrage fichiers
  macOS `._*`, profils d'exclusion de caractères (`sans_ponctuation`,
  `sans_apostrophes`), sélecteur dynamique de modèles via
  `/api/models/{provider}`.
- **Sprint 13** — nettoyage `pyproject.toml`, parallélisation runner
  (ThreadPool/ProcessPool selon `execution_mode`), timeout par doc,
  résultats partiels NDJSON, validation statistique Wilcoxon.
- **Sprint 14** — filtrage robuste des moteurs côté CLI/web, validation
  corpus avant lancement.
- **Sprint 15** — fix bug pipeline OCR+LLM sortie vide : `mistral_adapter`
  normalise les `ContentChunk`, log `finish_reason` + tokens.
- **Sprint 16** — câblage de `line_metrics` et `hallucination` dans
  `runner` et l'agrégation `EngineReport` ; fondations du moteur
  narratif (`core/narrative/` avec `Fact`/`DetectorRegistry`) ;
  Pillow `getdata()` → `tobytes()` ; deux `except: pass` → warnings.
- **Sprint 17** — refactor du rapport monolithique : `generator.py`
  3690 → 617 lignes via Jinja2, 10 fichiers externes dans
  `picarones/report/templates/`, i18n migrée vers
  `report/i18n/{fr,en}.json`. +16 tests de non-régression.
- **Sprint 18** — test de Friedman multi-moteurs + Nemenyi post-hoc +
  Critical Difference Diagram (Demšar 2006) ; `core/statistics.py`
  étendu, fallback pur Python, scipy optionnel via extra `[stats]`.
  Détecteur narratif `STATISTICAL_TIE` activé. +41 tests.
- **Sprint 19** — moteur narratif complet : 9 détecteurs implémentés
  (global_leader_cer, significant_gap, stratum_winner/collapse,
  error_profile_outlier, llm_hallucination_flag, robustness_fragile,
  speed_winner, confidence_warning), arbitre, renderer YAML,
  `_narrative_summary.html`. Garde-fou anti-hallucination testé.
  +32 tests.
- **Sprint 20** — modélisation coût + vue Pareto : `core/pricing.py`,
  `data/pricing.yaml`, `compute_pareto_front` multi-objectifs,
  Chart.js Pareto avec axes coût/vitesse/carbone. Détecteurs
  `pareto_alternative` + `cost_outlier` activés. +28 tests.
- **Sprint 21** — glossaire contextuel (25 entrées bilingues) +
  panneau « Mode avancé » : choix de colonnes, filtres par strate,
  vue opt-in « score composite personnel » avec curseurs à 0 par défaut
  et formule visible. État persisté en URL. +21 tests.
- **Sprint 22** — études de cas (`docs/case-studies/`),
  `docs/tutorials/reading-a-report.md`, trois guides développeur dans
  `docs/developer/`. Garde-fou « pas de fausses études prétendant
  être réelles ». +18 tests.

### Modifié

- `pyproject.toml` : extras `[stats]`, `[hf]`, mises à jour de
  `dev`/`web` pour `python-multipart`.
- `picarones/core/runner.py` : refactor pour gérer le `execution_mode`
  des moteurs (IO-bound vs CPU-bound).

### Corrigé

- `python-multipart` durablement présent dans `[dev]` et `[web]`
  (FastAPI vérifie l'import au décorateur `@app.post`).
- Tests Windows SQLite `test_history_empty_db` (gc.collect avant
  unlink).
- `test_search_language_filter` (HuggingFace) — assertion corrigée.

---

## [1.0.0] — Sprint 9 — 2025-03

### Ajouté
- `README.md` complet bilingue (français + anglais) avec badges CI, description des fonctionnalités, tableau des moteurs, variables d'environnement
- `INSTALL.md` — guide d'installation détaillé pour Linux (Ubuntu/Debian), macOS et Windows, incluant Tesseract, Pero OCR, Ollama, configuration des clés API, Docker
- `CHANGELOG.md` — historique des sprints 1 à 9
- `CONTRIBUTING.md` — guide pour contribuer : ajouter un moteur OCR, un adaptateur LLM, soumettre une PR
- `Makefile` — commandes `make install`, `make test`, `make demo`, `make serve`, `make build`, `make build-exe`, `make docker-build`, `make lint`, `make clean`
- `Dockerfile` — image Docker multi-étape basée sur Python 3.11-slim, Tesseract pré-installé, `CMD ["picarones", "serve", "--host", "0.0.0.0"]`
- `docker-compose.yml` — service Picarones + service Ollama optionnel (profil `ollama`)
- `.github/workflows/ci.yml` — pipeline GitHub Actions : tests sur Python 3.11/3.12, Linux/macOS/Windows, rapport de couverture
- `picarones.spec` — configuration PyInstaller pour générer des exécutables standalone (Linux, macOS, Windows)
- `picarones/__main__.py` — permet l'exécution via `python -m picarones`
- Version bumped à `1.0.0` dans `pyproject.toml` et `__init__.py`
- Extras PyPI `[llm]`, `[ocr-cloud]`, `[all]` dans `pyproject.toml`
- Tests Sprint 9 : `tests/test_sprint9_packaging.py` (30 tests)

### Modifié
- `pyproject.toml` : version 1.0.0, nouveaux extras, classifiers mis à jour, URLs projet ajoutées

---

## [0.8.0] — Sprint 8 — 2025-03

### Ajouté
- **eScriptorium** (`picarones/importers/escriptorium.py`)
  - `EScriptoriumClient` : connexion par token API, listing projets/documents/pages, gestion de la pagination
  - `import_document()` : import d'un document avec ses transcriptions comme corpus Picarones
  - `export_benchmark_as_layer()` : export des résultats benchmark comme couche OCR nommée dans eScriptorium
  - `connect_escriptorium()` : connexion avec validation automatique
- **Gallica API** (`picarones/importers/gallica.py`)
  - `GallicaClient` : recherche SRU par cote/titre/auteur/date/langue/type
  - Récupération OCR Gallica texte brut (`f{n}.texteBrut`)
  - Import IIIF Gallica avec enrichissement OCR comme vérité terrain de référence
  - Métadonnées OAI-PMH (`/services/OAIRecord`)
  - `search_gallica()`, `import_gallica_document()` — fonctions de commodité
- **Suivi longitudinal** (`picarones/core/history.py`)
  - `BenchmarkHistory` : base SQLite horodatée par run, moteur, corpus, CER/WER
  - `record()` depuis `BenchmarkResult`, `record_single()` pour imports manuels
  - `query()` avec filtres engine/corpus/since/limit
  - `get_cer_curve()` : données prêtes pour Chart.js
  - `detect_regression()` / `detect_all_regressions()` : seuil configurable en points de CER
  - `export_json()` — export complet de l'historique
  - `generate_demo_history()` : 8 runs fictifs avec régression simulée au run 5
- **Analyse de robustesse** (`picarones/core/robustness.py`)
  - 5 types de dégradation : bruit gaussien, flou, rotation, réduction de résolution, binarisation
  - `degrade_image_bytes()` : Pillow (préféré) ou fallback pur Python
  - `RobustnessAnalyzer.analyze()` : CER par niveau, seuil critique automatique
  - `DegradationCurve`, `RobustnessReport`, `_build_summary()`
  - `generate_demo_robustness_report()` : rapport fictif réaliste sans moteur réel
- **CLI Sprint 8**
  - `picarones history` : historique avec filtres, détection de régression, export JSON, mode `--demo`
  - `picarones robustness` : analyse de robustesse, barres ASCII, export JSON, mode `--demo`
  - `picarones demo --with-history --with-robustness` : démonstration intégrée
- `picarones/importers/__init__.py` mis à jour pour exporter les nouveaux importeurs

### Tests
- `tests/test_sprint8_escriptorium_gallica.py` : 74 tests (eScriptorium, Gallica, CLI)
- `tests/test_sprint8_longitudinal_robustness.py` : 86 tests (history, robustesse, CLI)
- **Total** : 743 tests (anciennement 583)

---

## [0.7.0] — Sprint 7 — 2025-02

### Ajouté
- **Rapport HTML v2**
  - Intervalles de confiance Bootstrap à 95% (`bootstrap_ci()`)
  - Tests de Wilcoxon et matrices de tests par paires (`wilcoxon_test()`, `pairwise_stats()`)
  - Courbes de fiabilité (CER cumulatif par percentile de qualité)
  - Diagrammes de Venn des erreurs communes/exclusives entre concurrents (2 et 3 ensembles)
  - Clustering des patterns d'erreurs (k-means simplifié sur n-grammes d'erreur)
  - Matrice de corrélation entre métriques (Pearson)
  - Score de difficulté intrinsèque par document (`compute_difficulty()`, `compute_all_difficulties()`)
  - Scatter plots interactifs qualité image vs CER, colorés par type de script
  - Heatmaps de confusion unicode améliorées
- `picarones/core/statistics.py` : module dédié aux tests statistiques
- `picarones/core/difficulty.py` : score de difficulté intrinsèque

### Tests
- `tests/test_sprint7_advanced_report.py` : 100 tests (bootstrap, Wilcoxon, Venn, clustering, difficulté)
- **Total** : 583 tests (anciennement 483)

---

## [0.6.0] — Sprint 6 — 2025-02

### Ajouté
- **Interface web FastAPI** (`picarones/web/app.py`)
  - Endpoints REST pour lancer des benchmarks, consulter les résultats, lister les moteurs
  - Streaming des logs en temps réel (Server-Sent Events)
  - `picarones serve` — lancement du serveur uvicorn
- **Import HuggingFace Datasets** (`picarones/importers/huggingface.py`)
  - Recherche, filtrage et import partiel de datasets OCR/HTR
  - Datasets patrimoniaux pré-référencés : IAM, RIMES, READ-BAD, Esposalles…
  - Cache local avec gestion des versions
- **Import HTR-United** (`picarones/importers/htr_united.py`)
  - Listing et import depuis le catalogue HTR-United
  - Lecture des métadonnées : langue, script, institution, époque
- **Adaptateurs Ollama** (`picarones/llm/ollama_adapter.py`)
  - Support de Llama 3, Gemma, Phi et tout modèle Ollama local
  - Mode texte seul (LLMs non multimodaux)
- **Profils de normalisation pré-configurés**
  - Français médiéval, Français moderne, Latin médiéval, Imprimés anciens
  - Profil personnalisé exportable/importable

### Tests
- `tests/test_sprint6_web_interface.py` : 90 tests
- **Total** : 483 tests (anciennement 393)

---

## [0.5.0] — Sprint 5 — 2025-02

### Ajouté
- **Matrice de confusion unicode** (`picarones/core/confusion.py`)
  - `build_confusion_matrix()`, `aggregate_confusion_matrices()`
  - Affichage compact trié par fréquence d'erreur
- **Scores ligatures et diacritiques** (`picarones/core/char_scores.py`)
  - `compute_ligature_score()` : fi, fl, ff, ffi, ffl, st, ct, œ, æ, ꝑ, ꝓ…
  - `compute_diacritic_score()` : accents, cédilles, trémas, diacritiques combinants
- **Taxonomie des erreurs en 10 classes** (`picarones/core/taxonomy.py`)
  - Confusion visuelle, erreur diacritique, casse, ligature, abréviation, hapax, segmentation, hors-vocabulaire, lacune, sur-normalisation LLM
- **Analyse structurelle** (`picarones/core/structure.py`)
  - Score d'ordre de lecture, taux de segmentation des lignes, conservation des sauts de paragraphe
- **Métriques de qualité image** (`picarones/core/image_quality.py`)
  - Netteté (Laplacien), niveau de bruit, contraste (Michelson), détection rotation résiduelle
  - Corrélations image ↔ CER
- Intégration de toutes ces métriques dans le rapport HTML (vue Analyse, vue Caractères)
- Scatter plots qualité image vs CER

### Tests
- `tests/test_sprint5_advanced_metrics.py` : 100 tests
- **Total** : 393 tests (anciennement 293)

---

## [0.4.0] — Sprint 4 — 2025-01

### Ajouté
- **Adaptateurs APIs cloud OCR**
  - Mistral OCR (`picarones/engines/mistral_ocr.py`) — Mistral OCR 3, multimodal
  - Google Vision (`picarones/engines/google_vision.py`) — Document AI
  - Azure Document Intelligence (`picarones/engines/azure_doc_intel.py`)
- **Import IIIF v2/v3** (`picarones/importers/iiif.py`)
  - Sélecteur de pages (`"1-10"`, `"1,3,5"`, `"all"`)
  - Téléchargement images et extraction des annotations de transcription si disponibles
  - Compatibilité : Gallica, Bodleian, British Library, BSB, e-codices, Europeana
  - `picarones import iiif <url>` — commande CLI
- **Normalisation unicode** (`picarones/core/normalization.py`)
  - NFC, caseless, diplomatique (tables ſ=s, u=v, i=j, æ=ae, œ=oe…)
  - Profils configurables via YAML
  - CER diplomatique dans les métriques

### Tests
- `tests/test_sprint4_normalization_iiif.py` : 100 tests
- **Total** : 293 tests (anciennement 193)

---

## [0.3.0] — Sprint 3 — 2025-01

### Ajouté
- **Pipelines OCR+LLM** (`picarones/pipelines/base.py`)
  - Mode 1 — Post-correction texte brut (LLM reçoit la sortie OCR)
  - Mode 2 — Post-correction avec image (LLM reçoit image + OCR)
  - Mode 3 — Zero-shot LLM (LLM reçoit uniquement l'image)
  - Chaînes composables multi-étapes
- **Adaptateurs LLM**
  - OpenAI (`picarones/llm/openai_adapter.py`) — GPT-4o, GPT-4o mini
  - Anthropic (`picarones/llm/anthropic_adapter.py`) — Claude Sonnet, Haiku
  - Mistral (`picarones/llm/mistral_adapter.py`) — Mistral Large, Pixtral
- **Détection de sur-normalisation LLM** (`picarones/pipelines/over_normalization.py`)
  - Mesure du taux de modification sur des passages déjà corrects
  - Classe 10 dans la taxonomie des erreurs
- **Bibliothèque de prompts**
  - Prompts pour manuscrits médiévaux, imprimés anciens, latin
  - Versionning des prompts dans les métadonnées du rapport
- Vue spécifique OCR+LLM dans le rapport : diff triple GT / OCR brut / après correction

### Tests
- `tests/test_sprint3_llm_pipelines.py` : 100 tests
- **Total** : 193 tests (anciennement 93)

---

## [0.2.0] — Sprint 2 — 2025-01

### Ajouté
- **Rapport HTML interactif** (`picarones/report/generator.py`)
  - Fichier HTML auto-contenu, lisible hors-ligne
  - Tableau de classement des concurrents (CER, WER, scores), tri par colonne
  - Graphique radar (spider chart) : CER / WER / Précision diacritiques / Ligatures
  - Vue Galerie : toutes les images avec badges CER colorés (vert→rouge), filtres
  - Vue Document : image zoomable + diff coloré façon GitHub, scroll synchronisé N-way
  - Vue Analyse : histogrammes de distribution CER, scatter plots
  - Recommandation automatique de moteur
  - Exports CSV, JSON, ALTO XML depuis le rapport
- **Diff coloré** (`picarones/report/diff_utils.py`)
  - Diff au niveau caractère et mot
  - Insertions (vert), suppressions (rouge), substitutions (orange)
  - Bascule diplomatique / normalisé
- `picarones demo` — rapport de démonstration avec données fictives réalistes
- `picarones report --results results.json` — génère le HTML depuis un JSON existant
- `picarones/fixtures.py` — générateur de benchmarks fictifs (12 textes médiévaux, 4 concurrents)

### Tests
- `tests/test_report.py`, `tests/test_diff_utils.py` : 93 tests
- **Total** : 93 tests (anciennement 20)

---

## [0.1.0] — Sprint 1 — 2025-01

### Ajouté
- **Structure complète du projet** Python avec `pyproject.toml`, `setup`, packaging
- **Adaptateur Tesseract 5** (`picarones/engines/tesseract.py`) via `pytesseract`
  - Configuration lang, PSM, DPI
  - Récupération de la version
- **Adaptateur Pero OCR** (`picarones/engines/pero_ocr.py`)
  - Chargement de modèle, traitement d'image
- **Interface abstraite** `BaseOCREngine` avec `process_image()`, `get_version()`, propriétés
- **Calcul CER et WER** (`picarones/core/metrics.py`) via `jiwer`
  - CER brut, NFC, caseless
  - WER, WER normalisé, MER, WIL
  - Longueurs de référence et hypothèse
- **Chargement de corpus** (`picarones/core/corpus.py`)
  - Dossier local : paires image / `.gt.txt`
  - Détection automatique des extensions image (jpg, png, tif, bmp…)
  - Classe `Corpus`, `Document`
- **Export JSON** (`picarones/core/results.py`)
  - `BenchmarkResult`, `EngineReport`, `DocumentResult`
  - `ranking()` : classement par CER moyen
  - `to_json()` avec horodatage et métadonnées
- **Orchestrateur benchmark** (`picarones/core/runner.py`)
  - Traitement séquentiel des documents par moteur
  - Barre de progression `tqdm`
  - Cache des sorties par hash SHA-256
- **CLI Click** (`picarones/cli.py`)
  - `picarones run` — benchmark complet
  - `picarones metrics` — CER/WER entre deux fichiers
  - `picarones engines` — liste des moteurs avec statut
  - `picarones info` — version et dépendances
  - `--fail-if-cer-above` pour intégration CI/CD

### Tests
- `tests/test_metrics.py`, `test_corpus.py`, `test_engines.py`, `test_results.py` : 20 tests
