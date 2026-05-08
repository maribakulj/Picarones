# CLAUDE.md — Picarones

Plateforme de benchmark OCR/HTR pour documents patrimoniaux.
Repo : github.com/maribakulj/Picarones
HuggingFace Space : huggingface.co/spaces/Ma-Ri-Ba-Ku/Picarones (Docker, port 7860)

---

## Architecture — coexistence legacy + rewrite

Voir le manifeste complet dans [`docs/explanation/architecture.md`](docs/explanation/architecture.md).

Le projet a deux arborescences qui cohabitent **par design** depuis le
**rewrite ciblé S27-S46** (cf. [`docs/migration/rewrite-status-s46.md`](docs/migration/rewrite-status-s46.md)) :

**Arbo cible (post-rewrite)** — chemin canonique pour tout nouveau code :

```
domain → formats → evaluation → pipeline → adapters → app → reports_v2 → interfaces
```

Règle d'import stricte : les dépendances vont uniquement de l'extérieur
vers l'intérieur.  Vérifié par `tests/architecture/test_layer_dependencies.py`.

**Arbo legacy (pré-rewrite)** — `picarones/{cli,web,llm,pipelines,
measurements,extras}/` reste exécutable le temps que les callers
externes (HuggingFace Space, scripts BnF) migrent.  Ne pas y
ajouter de nouveau code.  Calendrier de retrait documenté dans le
CHANGELOG.  Sous-paquets retirés en mai 2026 :
``engines/`` + ``modules/`` (Lot E), ``report/`` (Lot F),
``core/`` (Lots A-G — entièrement vidé du legacy au profit
de ``domain/``, ``evaluation/``, ``formats/``).

---

## Setup

```bash
pip install -e ".[dev,web]"          # toujours inclure [web] pour les tests
pytest tests/ -q --tb=short          # lancer les tests
picarones demo --output rapport.html # rapport démo sans moteur installé
picarones serve --port 8080          # interface web locale
```

---

## Structure

Architecture canonique en **8 couches concentriques** :

```
picarones/
├── domain/                     Couche 1 — types purs (Pydantic + stdlib only)
│   ├── artifacts.py            Artifact, ArtifactType (10 types)
│   ├── corpus.py               CorpusSpec
│   ├── documents.py            DocumentRef
│   ├── pipeline_spec.py        PipelineSpec, PipelineStep (Pydantic immutable)
│   ├── module_protocol.py      BaseModule (en cours de retrait au profit de StepExecutor)
│   ├── facts.py                Fact, FactType, FactImportance, DetectorRegistry
│   ├── evaluation_spec.py      MetricSpec, EvaluationView, EvaluationSpec
│   ├── projection_spec.py      ProjectionSpec
│   ├── provenance.py           ProvenanceRecord
│   ├── run_manifest.py         RunManifest
│   └── errors.py               PicaronesError, AdapterStepError, ...
│
├── formats/                    Couche 2 — parsing/sérialisation (lxml, defusedxml)
│   ├── alto/, pagexml/         ALTO 4, PAGE XML
│   └── _xml_utils.py           safe_parse_xml
│
├── evaluation/                 Couche 3 — métriques et calcul
│   ├── metrics/                ~30 métriques (CER/WER, MUFI, philological, NER, …)
│   ├── statistics/             Wilcoxon, Friedman/Nemenyi, bootstrap, Pareto, CDD
│   ├── views/, projectors/     EvaluationView (S13+), AltoToText, PageToText, ...
│   ├── corpus.py               Document, Corpus, GTLevel + payloads (legacy en retrait)
│   ├── pipeline.py             PipelineRunner legacy (en cours de convergence)
│   ├── benchmark_result.py     BenchmarkResult, EngineReport, DocumentResult
│   ├── metric_registry.py      MetricSpec, register_metric, compute_at_junction
│   ├── metric_hooks.py         register_document_metric, register_corpus_aggregator
│   ├── metric_result.py        MetricsResult, aggregate_metrics
│   └── _diff_utils.py          compute_word_diff, compute_char_diff, diff_stats
│
├── pipeline/                   Couche 4 — orchestration canonique
│   ├── executor.py             PipelineExecutor (instance-based)
│   ├── planner.py              ExecutionPlan, StepInputBinding
│   ├── protocols.py            StepExecutor Protocol, ExecutionMode
│   ├── types.py                RunContext, StepResult, PipelineResult (Pydantic)
│   └── _legacy_module_adapter.py  Pont BaseModule → StepExecutor (transitoire)
│
├── adapters/                   Couche 5 — adapters externes (libs externes autorisées)
│   ├── ocr/                    Tesseract, Pero, Mistral OCR, Google Vision, Azure DI, Precomputed
│   ├── llm/                    OpenAI, Anthropic, Mistral, Ollama (BaseLLMAdapter)
│   ├── vlm/                    Adapters VLM (zero-shot)
│   ├── corpus/                 IIIF, Gallica, HTR-United, HuggingFace
│   ├── storage/                ArtifactStore, JobStore
│   ├── legacy_engines/         5 OCR engines legacy (BaseModule-based, en retrait)
│   └── legacy_modules/         TextToAltoMonoRegion (en retrait)
│
├── app/                        Couche 6 — services applicatifs
│   └── services/               BenchmarkService, CorpusRunner, RunOrchestrator
│
├── reports_v2/                 Couche 7 — rendu HTML / JSON / CSV
│   ├── html/                   ReportGenerator + 22 renderers + 5 vues + templates Jinja2
│   ├── json/, csv/             exports tabulaires
│   ├── narrative/              moteur narratif (18 détecteurs)
│   ├── glossary/, i18n/        glossaire + i18n FR/EN
│   └── _helpers/               colors, render_helpers, assets
│
├── interfaces/                 Couche 8 — entrées utilisateur
│   ├── cli/                    Click CLI
│   └── web/                    FastAPI (en cours de migration)
│
├── prompts/                    8 fichiers .txt FR+EN
├── data/                       Tables indicatives (pricing.yaml)
│
# Arborescence legacy en cours de retrait (cf. docs/migration/) :
# measurements/, llm/, pipelines/, web/, cli/, extras/
# (engines/, modules/ retirés au Lot E ; report/ au Lot F ;
# core/ entièrement supprimé aux Lots A-G)
└── fixtures.py                 Corpus de test fictifs
```

---

## État des tests et bugs historiques

`pytest tests/` → **4700 passed, 12 skipped, 8 deselected, 0 failed**
(post-S59).  Les deselected sont les markers `live` (5 tests d'intégration
contre vraie API/binaire) + `network` (3 tests qui hit le réseau réel),
opt-in en local via `pytest -m live` ou `pytest -m network`.  Le
compteur en prose est synchronisé automatiquement par
`scripts/gen_readme_tables.py` — toute modification manuelle sera
ré-écrasée au prochain `make lint`.

### Bugs documentés antérieurement — tous résolus

| Bug | Statut | Sprint de résolution |
|-----|--------|---------------------|
| Pipeline OCR+LLM sortie vide (`tesseract → ministral-3b-latest`) | ✅ Résolu | Sprint 15 — adapter Mistral logue `finish_reason`, `completion_tokens`, normalise les ContentChunk |
| CI `python-multipart` manquant | ✅ Résolu | `pyproject.toml` expose `python-multipart>=0.0.9` dans les extras `dev` ET `web`; `ci.yml:71` installe `.[dev,web]` |
| Tests fixtures post-Sprint 10 (counts moteurs, flag `is_pipeline`) | ✅ Résolu | Fixtures mises à jour |
| Test Windows SQLite `test_history_empty_db` | ✅ Résolu | `try/except OSError` + `gc.collect()` avant `unlink` |
| Test HuggingFace `test_search_language_filter` | ✅ Résolu | Assertion corrigée |

En cas de régression sur un de ces bugs, chercher les fichiers de test
correspondants (`test_sprint15_llm_pipeline_bugs.py`, `test_sprint8_escriptorium_gallica.py`,
`test_sprint6_web_interface.py`) avant de ré-ouvrir une enquête.

---

## Règles importantes — ne pas toucher

- **Ne jamais retirer `python-multipart` des dépendances** : FastAPI vérifie sa présence à
  l'import du module (décoration `@app.post` avec `UploadFile`), pas à l'exécution. Ça casse
  tous les tests web au setup.
- **Ne jamais mettre `except Exception: pass`** : remplacer par
  `logger.warning("[module] fonctionnalité dégradée : %s", e)`.
- **Toujours utiliser `logger.warning` avec message explicite** quand une fonctionnalité optionnelle
  échoue (confusion, taxonomy, structure, image_quality, etc.).
- **Avant tout push, lancer `make lint`** (ou `ruff check picarones/ tests/`).
  La config est centralisée dans `pyproject.toml` sous `[tool.ruff]`, donc
  CI, Makefile et invocation directe produisent le même résultat. Le job
  `lint` du CI est bloquant — un F401 (import inutilisé) ou un E741
  (variable ambiguë) fait échouer la PR, par design.
- **Les profils de normalisation** sont dans `picarones/formats/text/normalization.py` — l'endpoint
  `/api/normalization/profiles` doit les lire dynamiquement depuis ce fichier, pas depuis une
  liste statique.

---

## Variables d'environnement

```bash
# Clés API LLM (configurées dans HuggingFace Space Settings → Variables and secrets)
MISTRAL_API_KEY=...
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# OCR cloud (optionnel)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/creds.json
AZURE_DOC_INTEL_ENDPOINT=https://...
AZURE_DOC_INTEL_KEY=...
```

---

## Pipelines OCR+LLM — modes

| Mode | Description |
|------|-------------|
| **zero_shot** | Le LLM reçoit l'image directement et transcrit sans OCR préalable (VLM) |
| **post_correction_texte** | OCR → texte brut → LLM corrige le texte (modèles texte seul) |
| **post_correction_image_texte** | OCR → LLM reçoit image + texte brut pour correction (VLM) |

`ministral-3b-latest` = modèle texte pur → utiliser mode `post_correction_texte` uniquement.

---

## CI/CD

- **CI GitHub Actions** : `.github/workflows/ci.yml` — Python 3.11/3.12, Linux/macOS/Windows
- **Sync HuggingFace** : `.github/workflows/sync_to_huggingface.yml` — push auto sur main
  (nécessite secret `HF_TOKEN` dans GitHub Settings → Secrets → Actions)
- **HuggingFace Space** : Docker sur port 7860

---

## Sprints réalisés

L'historique détaillé des **97+ sprints** du projet (de la fondation
S1 jusqu'au rewrite ciblé S27-S46 puis l'audit institutionnel
S47-S59) est dans le CHANGELOG.md à la racine.  Cette page,
auparavant pléthorique, ne le duplique plus — un seul endroit où
chercher.

Pour le travail courant, ce qui compte :

- **Phase active** : retrait complet du legacy vers le rewrite,
  stratégie 4.B (full migration, sans préservation API).  Le
  projet est en stand-by jusqu'à la fin de la migration
  complète — tests rouges acceptables temporairement, breaking
  changes acceptés.
- **Principe directeur** : suppression agressive.  Pas de shim
  qui survit à son usage.  Dès qu'un caller migre vers le
  canonique, son shim est supprimé.  Tout symbole legacy public
  doit être tracé dans
  [`tests/architecture/test_legacy_canonical_parity.py`](tests/architecture/test_legacy_canonical_parity.py)
  — c'est le journal de bord vivant qui garantit qu'aucune
  fonctionnalité n'est silencieusement perdue.
- **Plan maître** : [`docs/migration/legacy-retirement-plan.md`](docs/migration/legacy-retirement-plan.md)
  — cartographie complète des Phases 0-11 avec statut.
- **Sub-plan convergence pipeline** : [`docs/migration/pipeline-convergence-plan.md`](docs/migration/pipeline-convergence-plan.md)
  — détail de Sub-phases 7.A-7.D (BaseModule → StepExecutor).
- **Documents de référence figés** :
  docs/migration/rewrite-status-s46.md (état du rewrite),
  docs/audits/ (audits historiques figés),
  docs/roadmap/evolution-2026.md (plan stratégique).

### Pour reprendre dans une nouvelle session

**Procédure complète** : lire d'abord
[`docs/migration/SESSION_HANDOVER.md`](docs/migration/SESSION_HANDOVER.md)
qui contient :

- Sources de vérité par ordre de priorité.
- Vérifications à faire avant de toucher au code (branche,
  working tree, tests, lint).
- Pièges connus (architecture des couches, patterns shim,
  test_module_coverage / test_file_budgets / test_doc_paths /
  README généré).
- Plan d'exécution détaillé de la prochaine sub-phase.

Résumé express :

1. `git branch --show-current` → `claude/repo-analysis-cukvm`.
2. `git status` → working tree clean.
3. `pytest tests/ -q --no-header --tb=line` → 4700 passed.
4. `git log -1 --format=%B` → décrit la prochaine sub-phase.

**Règles d'architecture critiques** (apprises à la dure) :

- ``evaluation/`` whitelist externe : ``PIL, annotated_types,
  jiwer, numpy, pydantic, rapidfuzz, scipy, spacy,
  typing_extensions, yaml`` — **pas** ``pytesseract``,
  ``mistralai``, ``azure``, ``google``, ``pero_ocr``.  Tout code
  qui importe ces libs externes va dans ``adapters/`` (qui
  autorise les libs externes par design).
- ``evaluation/`` ne peut pas importer depuis ``pipeline/`` :
  c'est le sens inverse de la dépendance.  Si un module bridge
  les deux contrats, il vit dans ``pipeline/``.
- ``reports_v2/`` ne peut pas importer depuis ``measurements/``
  (legacy) ou ``core/`` (legacy).  Les renderers consomment les
  modules canoniques de ``evaluation/metrics/``.
- ``test_module_coverage::TEST_ONLY_BASELINE`` : ajouter à
  cette frozenset dès qu'un shim ``measurements/X.py`` n'a
  plus de consommateur production (cas typique : un renderer
  est migré vers ``reports_v2/`` et importe directement le
  canonique au lieu du shim).

### Apprentissages des phases précédentes

- **Pattern shim** : pour chaque migration, le chemin legacy
  devient un shim minimal (< 25 lignes) avec
  ``from canonical_path import *  # noqa: F401, F403`` +
  ``DeprecationWarning`` à l'import.  Les noms privés
  (``_FORMATTERS``, ``_PYTESSERACT_AVAILABLE``, etc.) doivent
  être importés explicitement en plus de ``import *`` car
  ``*`` n'exporte pas les ``_`` privés.
- **Pattern docstring** : ajouter en tête du module canonique
  un bloc ``Phase X — module relocalisé depuis Y vers Z``
  avec mention de la suppression en 2.0.
- **Pattern test budgets** : si un fichier dépasse 400 LOC,
  ajouter une entrée dans
  ``tests/architecture/test_file_budgets.py::FILE_BUDGETS``
  avec budget = LOC actuel + ~15 %.
- **Pattern docs paths** : si un sub-plan référence un futur
  chemin Python qui n'existe pas encore (forward reference),
  bumper ``BROKEN_PATHS_BASELINE`` du même montant et noter que la
  référence sera résolue quand le fichier sera créé.

## Moteur narratif

Le modèle de données (`Fact`, `FactType`, `FactImportance`,
`DetectorRegistry`) vit en couche 1 (`domain`) dans
[`picarones/domain/facts.py`](picarones/domain/facts.py). Les détecteurs et
le rendu vivent en couche 7 (`reports_v2`) :

```
picarones/reports_v2/narrative/
├── __init__.py              API publique + pipeline build_synthesis
├── arbiter.py               Tri par importance, non-redondance, anti-contradiction
├── renderer.py              Rendu templates YAML par str.format_map (déterministe)
├── registry.py              Registre par défaut des détecteurs
├── templates/{fr,en}.yaml   18 templates × 2 langues
└── detectors/               18 détecteurs en 6 familles
    ├── ranking.py           5 (global_leader, statistical_tie, significant_gap,
    │                          speed_winner, median_mean_gap_warning)
    ├── pareto.py            2 (pareto_alternative, cost_outlier)
    ├── stratum.py           3 (stratum_winner, stratum_collapse,
    │                          stratification_recommended)
    ├── quality.py           4 (error_profile_outlier, llm_hallucination_flag,
    │                          robustness_fragile, confidence_warning)
    ├── history.py           3 (engine_off_baseline, engine_unstable,
    │                          regression_in_history)
    └── ensemble.py          1 (ensemble_opportunity)
```

**Principe anti-hallucination** : chaque valeur numérique ou nom d'entité
dans le `payload` d'un `Fact` provient du JSON d'entrée. Le test
`test_sprint19_narrative_engine.py` parse la synthèse rendue et vérifie
la traçabilité.

**Règle anti-contradiction** (arbitre) : si `SIGNIFICANT_GAP` (Wilcoxon
non corrigé) et `STATISTICAL_TIE` (Nemenyi corrigé) concernent les mêmes
moteurs, Nemenyi l'emporte.

**Pipeline** : `build_synthesis(benchmark_data, lang, max_facts=5)`
détecte, arbitre, rend.

---

## Contexte développement

- **Environnement** : GitHub Codespaces, Python 3.11+
- **Tests** : `pytest tests/ -q` → 4700 passed, 12 skipped, 24
  deselected, 0 failed (au moment de la pause de session).
- **Plan d'évolution actif** : [`docs/roadmap/evolution-2026.md`](docs/roadmap/evolution-2026.md).
- **Plan retrait du legacy (maître)** : [`docs/migration/legacy-retirement-plan.md`](docs/migration/legacy-retirement-plan.md).
- **Sub-plan convergence pipeline** : [`docs/migration/pipeline-convergence-plan.md`](docs/migration/pipeline-convergence-plan.md).
- **Manifeste architecture** : [`docs/explanation/architecture.md`](docs/explanation/architecture.md).
- **API publique stable** : [`docs/reference/api-stable.md`](docs/reference/api-stable.md).
- **Branche active** : `claude/repo-analysis-cukvm`.

### Statut migration au moment du handover

| Phase   | Statut    | Détails                                                    |
|---------|-----------|------------------------------------------------------------|
| 0-3     | ✅ terminée | Foundation, statistics, narrative engine                   |
| 4       | ✅ terminée | 35 mesures legacy → ``evaluation/metrics/``                |
| 4-bis   | ✅ terminée | ``ArtifactType`` migration + 22 callers                    |
| 4-ter   | ✅ terminée | core/{metric_registry,metric_hooks,metrics,results} → eval |
| 4-quater | ✅ terminée | core/corpus → evaluation/corpus                           |
| 5.A     | ✅ terminée | helpers + glossary + i18n → reports_v2/                    |
| 5.B     | ✅ terminée | (intégré dans 5.A)                                         |
| 5.C     | ✅ terminée | 29 renderers + 5 modules pré-requis → reports_v2/          |
| 5.D     | ✅ terminée | 5 vues thématiques → reports_v2/html/views/                |
| 5.E     | ✅ terminée | generator + comparison + snapshot + data + templates       |
| 7.A     | ✅ terminée | engines/ + modules/ → adapters/legacy_*/                   |
| 7.B.1   | ✅ terminée | _BaseModuleAdapter + _PayloadRegistry (commit b70f12a)     |
| 7.B.2   | ⏳ EN COURS | PipelineRunner.run délègue à PipelineExecutor              |
| 7.B.3   | 📋 à venir | pipeline_benchmark/comparison via canonique                |
| 7.C     | 📋 à venir | Refactor 7 tests axe B (mocks BaseModule → StepExecutor)   |
| 7.D     | 📋 à venir | Suppression BaseModule + PipelineRunner + shims core/      |
| 6, 8-11 | 📋 à venir | pipelines/, importers, web, cli, retirement final          |

**Prochaine sub-phase à exécuter** : 7.B.2 (refactor du corps
de ``PipelineRunner.run`` dans ``evaluation/pipeline.py`` pour
qu'il délègue à ``PipelineExecutor`` via le wrapper
``_BaseModuleAdapter`` créé en 7.B.1).
