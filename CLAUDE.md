# CLAUDE.md — Picarones

Plateforme de benchmark OCR/HTR pour documents patrimoniaux.
Repo : github.com/maribakulj/Picarones
HuggingFace Space : huggingface.co/spaces/Ma-Ri-Ba-Ku/Picarones (Docker, port 7860)

---

## Architecture — 8 couches concentriques

Voir le manifeste complet dans [`docs/explanation/architecture.md`](docs/explanation/architecture.md).

```
domain → formats → evaluation → pipeline → adapters → app → reports → interfaces
```

Règle d'import stricte : les dépendances vont uniquement de l'extérieur
vers l'intérieur.  Vérifié par `tests/architecture/test_layer_dependencies.py`
+ `tests/architecture/test_no_legacy_imports_in_rewrite.py`
(`LEGACY_PACKAGES = ()` à v2.0 — plus aucun paquet legacy).

Tous les paquets legacy (`picarones/{core,measurements,engines,modules,
report,llm,pipelines,cli,web,extras}/` + `adapters/legacy_engines/` +
`adapters/legacy_pipelines/` + `interfaces/{cli,web}/_legacy/`) ont été
**supprimés** au cours des sprints A-H (mai 2026).

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
│   ├── pipeline_spec.py        PipelineSpec, PipelineStep
│   ├── module_protocol.py      BaseModule
│   ├── facts.py                Fact, FactType, FactImportance, DetectorRegistry
│   ├── evaluation_spec.py      MetricSpec, EvaluationView, EvaluationSpec
│   ├── projection_spec.py      ProjectionSpec
│   ├── provenance.py           ProvenanceRecord
│   ├── run_manifest.py         RunManifest
│   └── errors.py               PicaronesError, AdapterStepError, ...
│
├── formats/                    Couche 2 — parsing/sérialisation (lxml, defusedxml)
│   ├── alto/, pagexml/         ALTO 4, PAGE XML
│   ├── text/                   normalisation, profils de comparaison
│   └── _xml_utils.py           safe_parse_xml
│
├── evaluation/                 Couche 3 — métriques et calcul
│   ├── metrics/                ~37 métriques (CER/WER, MUFI, philological, NER, …)
│   ├── statistics/             Wilcoxon, Friedman/Nemenyi, bootstrap, Pareto, CDD
│   ├── views/, projectors/     EvaluationView, AltoToText, PageToText, ...
│   ├── registry/               MetricRegistry (DI)
│   ├── corpus.py               Document, Corpus, GTLevel + payloads
│   ├── benchmark_result.py     BenchmarkResult, EngineReport, DocumentResult
│   ├── metric_registry.py      MetricSpec, register_metric, compute_at_junction
│   ├── metric_hooks.py         register_document_metric, register_corpus_aggregator
│   ├── metric_result.py        MetricsResult, aggregate_metrics
│   ├── synthetic.py            generate_sample_benchmark (pour `picarones demo`)
│   └── _diff_utils.py          compute_word_diff, compute_char_diff, diff_stats
│
├── pipeline/                   Couche 4 — orchestration
│   ├── executor.py             PipelineExecutor (instance-based)
│   ├── planner.py              ExecutionPlan, StepInputBinding
│   ├── protocols.py            StepExecutor Protocol, ExecutionMode
│   ├── runner.py               CorpusRunner (backpressure + timeout + cancel)
│   ├── types.py                RunContext, StepResult, PipelineResult
│   ├── llm_pipeline_builder.py make_ocr_llm_pipeline_spec (3 modes)
│   └── llm_pipeline_config.py  OCRLLMPipelineConfig (container OCR+LLM)
│
├── adapters/                   Couche 5 — adapters externes (libs autorisées)
│   ├── ocr/                    Tesseract, Pero, Mistral OCR, Google Vision, Azure DI, Precomputed + factory
│   ├── llm/                    OpenAI, Anthropic, Mistral, Ollama (BaseLLMAdapter)
│   ├── vlm/                    Adapters VLM (zero-shot)
│   ├── corpus/                 IIIF, Gallica, HTR-United, HuggingFace, eScriptorium
│   └── storage/                ArtifactStore, JobStore
│
├── app/                        Couche 6 — services applicatifs
│   └── services/               BenchmarkService, RunOrchestrator, JobRunner,
│                               benchmark_runner (entry point CLI/web), partial_store
│
├── reports/                    Couche 7 — rendu HTML / JSON / CSV
│   ├── html/                   ReportGenerator + 28 renderers + 5 vues + templates Jinja2
│   ├── json/, csv/             exports tabulaires
│   ├── narrative/              moteur narratif (20 détecteurs)
│   ├── glossary/, i18n/        glossaire + i18n FR/EN
│   └── _helpers/               colors, render_helpers, assets
│
├── interfaces/                 Couche 8 — entrées utilisateur
│   ├── cli/                    Click CLI : run, diagnose, economics, edition,
│   │                           compare, robustness, history, serve, metrics,
│   │                           engines, info, demo, report, import (group)
│   └── web/                    FastAPI : UI Jinja2 + SSE benchmark + ZIP upload
│
├── prompts/                    8 fichiers .txt FR+EN
├── data/                       Tables indicatives (pricing.yaml)
└── i18n.py                     Helper i18n (multi-langue rapports)
```

---

## État des tests et bugs historiques

`pytest tests/` → **4900 passed, 16 skipped, 8 deselected, 2 xfailed, 0 failed**
(post-audit code-quality, mai 2026).  Les deselected sont les markers
`live` (5 tests d'intégration contre vraie API/binaire) + `network`
(3 tests qui hit le réseau réel), opt-in en local via `pytest -m live`
ou `pytest -m network`.  Le compteur ``passed`` est synchronisé
automatiquement par `scripts/gen_readme_tables.py` (CI : job
``sync-counters`` ; local : `make sync-counters-check`).  Le détail
``skipped``/``xfailed`` peut dériver de ±2 entre éditions et n'est
pas verrouillé en CI.

NB : utiliser ``python -m pytest tests/`` plutôt que ``pytest tests/``
directement — l'installation via ``uv tool install pytest`` masque
les deps Picarones et produit ~160 collection errors trompeurs.

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

Trois modes canoniques (typés `Literal[…]` dans `PipelineConfig.pipeline_mode`
depuis Phase 2 du chantier post-rewrite — toute autre valeur est rejetée
en 422 par Pydantic, plus de fallback silencieux) :

| Mode (clé API) | Description |
|---|---|
| **`zero_shot`** | Le VLM reçoit l'image directement et transcrit sans OCR préalable (pas d'OCR amont). |
| **`text_only`** | OCR → texte brut → LLM corrige le texte sans voir l'image (modèles texte seul). |
| **`text_and_image`** | OCR → VLM reçoit image + texte brut pour correction multimodale. |

`ministral-3b-latest` = modèle texte pur → utiliser mode `text_only`
uniquement.  L'UI envoie ces clés en anglais ; les libellés français
"Post-correction texte" / "Post-correction image+texte" sont des
labels d'affichage (i18n), pas des identifiants API.

---

## CI/CD

- **CI GitHub Actions** : `.github/workflows/ci.yml` — Python 3.11/3.12, Linux/macOS/Windows
- **Sync HuggingFace** : `.github/workflows/sync_to_huggingface.yml` — push auto sur main
  (nécessite secret `HF_TOKEN` dans GitHub Settings → Secrets → Actions)
- **HuggingFace Space** : Docker sur port 7860

---

## Sprints réalisés

L'historique détaillé des **97+ sprints** du projet (de la fondation
S1 jusqu'au rewrite ciblé S27-S46, l'audit institutionnel S47-S59,
puis le retrait complet du legacy A-H jusqu'à v2.0) est dans le
CHANGELOG.md à la racine.

**v2.0 (mai 2026)** : la migration vers l'architecture 8 couches
canoniques est terminée.  Plus aucun paquet legacy.  Plus aucun shim.

**Chantier post-rewrite (mai 2026, branche `claude/fix-module-rewiring-MHssX`)** :
réconciliation des chemins UI/API/runner après audit révélant des
options ignorées, moteurs annoncés sans backend, surfaces filesystem
ouvertes et round-trip JSON appauvri (cf. CHANGELOG.md).  Cinq phases
exécutées :
- **Phase 1 (sécurité P0)** : `output_dir` validé (importers HTR-United/HF),
  `db_path` validé (`/api/history/regressions`), ZIP collision de
  basename + validation image extraite.
- **Phase 2 (méthodologie P0)** : `pipeline_mode` strict Literal,
  `BenchmarkResult.from_json_object` (round-trip JSON complet —
  taxonomy/NER/calibration/philological/searchability/hallucination
  préservés), `partial_store` fingerprint SHA-256 (engine_config +
  normalization + char_exclude + corpus mtime/size + code version).
- **Phase 3 (moteurs fantômes)** : adapters `KrakenAdapter` et
  `CalamariAdapter` implémentés (lazy imports, extras `[kraken]` /
  `[calamari]`) + matrice CLI/Web/factory unifiée.
- **Phase 4 (code zombie)** : `upload_purge_task` (RGPD) branchée au
  lifespan + payload corpus dans `JobStore.create_job`,
  `/api/benchmark/start` unifié vers le worker v2,
  `HTRUnitedCatalogue.from_remote` (avec fallback demo + champ
  `is_demo` exposé), endpoints config save/load branchés à l'UI,
  workflows CLI `diagnose`/`economics`/`edition` génèrent le HTML
  automatiquement.
- **Phase 5 (naming)** : `CompetitorConfig` → `PipelineConfig`,
  `ocr_engine` → `engine_name` (rupture API, le field accepte aussi
  des VLMs et `corpus`).

**Règles d'architecture** :

- ``evaluation/`` whitelist externe : ``PIL, annotated_types,
  jiwer, numpy, pydantic, rapidfuzz, scipy, spacy,
  typing_extensions, yaml`` — **pas** ``pytesseract``,
  ``mistralai``, ``azure``, ``google``, ``pero_ocr``.  Tout code
  qui importe ces libs externes va dans ``adapters/``.
- ``evaluation/`` ne peut pas importer depuis ``pipeline/`` :
  c'est le sens inverse de la dépendance.  Un module qui pont
  les deux contrats vit dans ``pipeline/``.
- ``reports/`` consomme uniquement ``evaluation/metrics/``.
- ``interfaces/`` (couche 8) consomme ``app.services`` (couche 6) :
  ``app.services.benchmark_runner`` est l'entry point unique pour
  CLI et web.
- ``test_file_budgets`` : si un fichier dépasse 400 LOC, ajouter
  une entrée avec budget = LOC actuel + ~15 %.

## Moteur narratif

Le modèle de données (`Fact`, `FactType`, `FactImportance`,
`DetectorRegistry`) vit en couche 1 (`domain`) dans
[`picarones/domain/facts.py`](picarones/domain/facts.py). Les détecteurs et
le rendu vivent en couche 7 (`reports`) :

```
picarones/reports/narrative/
├── __init__.py              API publique + pipeline build_synthesis
├── arbiter.py               Tri par importance, non-redondance, anti-contradiction
├── renderer.py              Rendu templates YAML par str.format_map (déterministe)
├── registry.py              Registre par défaut des détecteurs
├── templates/{fr,en}.yaml   20 templates × 2 langues
└── detectors/               20 détecteurs en 6 familles
    ├── ranking.py           5 (global_leader, statistical_tie, significant_gap,
    │                          speed_winner, median_mean_gap_warning)
    ├── pareto.py            3 (pareto_alternative, cost_outlier,
    │                          pricing_staleness_warning)
    ├── stratum.py           3 (stratum_winner, stratum_collapse,
    │                          stratification_recommended)
    ├── quality.py           4 (error_profile_outlier, llm_hallucination_flag,
    │                          robustness_fragile, confidence_warning)
    ├── history.py           4 (engine_off_baseline, engine_unstable,
    │                          regression_in_history, importer_fallback_triggered)
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
- **Tests** : voir « État des tests et bugs historiques » plus haut
  (compteur synchronisé par ``scripts/gen_readme_tables.py``).
- **Manifeste architecture** : [`docs/explanation/architecture.md`](docs/explanation/architecture.md).
- **API publique stable** : [`docs/reference/api-stable.md`](docs/reference/api-stable.md).

### Statut v2.0 — mai 2026

L'architecture 8 couches est complète.  Tous les paquets legacy
top-level (`core/`, `measurements/`, `engines/`, `modules/`,
`report/`, `llm/`, `pipelines/`, `cli/`, `web/`, `extras/`) ainsi
que les sous-paquets transitoires (`adapters/legacy_engines/`,
`adapters/legacy_pipelines/`, `interfaces/{cli,web}/_legacy/`) ont
été supprimés.

| Phase / Sprint | Statut | Détails |
|---|---|---|
| Foundation S1-S26 | ✅ | domain, formats, evaluation, narrative engine |
| Rewrite ciblé S27-S46 | ✅ | pipeline, app.services, adapters/ocr canonique, reports |
| Audit S47-S59 | ✅ | confidences, sécurité web, registry typé, baselines |
| Plan A-H (mai 2026) | ✅ | Retrait complet du legacy : core/measurements/engines/modules/report/llm/pipelines/cli/web/extras supprimés ; interfaces/{cli,web}/_legacy promus au niveau canonique ; v2.0 release |
