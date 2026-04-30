# Architecture Picarones — vue d'ensemble post-chantiers

Ce document décrit l'architecture du projet **après les chantiers 1-5**
du plan d'évolution post-Sprint 97. Il complète le `CLAUDE.md`
historique (qui reste l'historique chronologique des sprints) en
donnant une **vue thématique** de l'organisation actuelle du code.

## Vue d'ensemble

Picarones est un **banc d'essai** pour pipelines OCR/HTR sur documents
patrimoniaux. Le projet livre :

1. **Engines OCR** (5 adapters : Tesseract, Pero OCR, Mistral OCR,
   Google Vision, Azure Document Intelligence).
2. **Adapters LLM** (4 providers : OpenAI, Anthropic, Mistral, Ollama)
   pour les pipelines OCR+LLM (zero-shot, post-correction…).
3. **Modules de référence** (chantier 1) : `TextToAltoMonoRegion`
   pour démonstrer l'extension `BaseModule` (image+texte → ALTO).
4. **Runner** orchestrateur multi-moteurs avec parallélisation
   ProcessPool (CPU-bound) ou ThreadPool (IO-bound).
5. **Rapport HTML** auto-suffisant (Chart.js embarqué) avec 5+ vues
   thématiques composables.
6. **Interface web FastAPI** + **CLI Click** (15 sous-commandes).

## Structure des packages

```
picarones/
├── cli/                     (chantier 5) Package CLI Click
│   ├── __init__.py          Groupe `cli` + helpers + commandes simples
│   ├── _workflows.py        run, diagnose, economics, edition, compare
│   ├── _pipeline.py         pipeline run, pipeline compare
│   ├── _imports.py          import iiif (+ futurs)
│   ├── _serve.py            serve (FastAPI launcher)
│   ├── _history.py          history (consultation SQLite)
│   └── _robustness.py       analyse robustesse
│
├── core/
│   ├── corpus.py            Document, Corpus, GTLevel multi-niveaux
│   ├── modules.py           BaseModule + ArtifactType (Sprint 33)
│   ├── metric_registry.py   Registre typé (Sprint 34)
│   ├── builtin_metrics.py   Métriques scalaires natives
│   ├── alto_metrics.py      (chantier 1) Métriques (ALTO, ALTO)
│   ├── metric_hooks.py      (chantier 2) Profils + registre de hooks
│   ├── builtin_hooks.py     (chantier 2) 12 hooks doc + 12 agrégateurs
│   ├── runner.py            Orchestrateur multi-moteurs
│   ├── pipeline_runner.py   Banc d'essai de pipelines composées
│   ├── narrative/
│   │   ├── facts.py         Modèle Fact + 18+ FactType
│   │   ├── registry.py      Registre déclaratif
│   │   ├── arbiter.py       Arbitrage des Facts (anti-redondance)
│   │   ├── renderer.py      Rendu i18n YAML → str.format_map
│   │   └── detectors/       (chantier 5) 6 sous-modules thématiques
│   │       ├── ranking.py
│   │       ├── pareto.py
│   │       ├── stratum.py
│   │       ├── quality.py
│   │       ├── history.py
│   │       └── ensemble.py
│   └── ... (~60 modules métriques philologiques, statistiques, etc.)
│
├── engines/
│   ├── base.py              (chantier 1) BaseOCREngine factorisée
│   │                        avec hooks _run_with_native +
│   │                        _extract_raw_confidences
│   ├── tesseract.py
│   ├── pero_ocr.py
│   ├── mistral_ocr.py
│   ├── google_vision.py
│   └── azure_doc_intel.py
│
├── llm/
│   ├── base.py              (chantier 4) Helpers normalize_llm_content +
│   │                        log_http_error factorisés
│   ├── mistral_adapter.py
│   ├── openai_adapter.py
│   ├── anthropic_adapter.py
│   └── ollama_adapter.py
│
├── modules/                 (chantier 1) Modules BaseModule de référence
│   └── alto_text_to_mono_region.py
│
├── importers/
│   ├── _http.py             (chantier 4) Helpers HTTP factorisés
│   ├── iiif.py
│   ├── htr_united.py
│   ├── gallica.py           (chantier 4) Délègue à _http
│   ├── huggingface.py
│   └── escriptorium.py
│
├── pipelines/               Pipelines OCR+LLM (zero_shot, post_correction)
│
├── prompts/                 8 templates .txt FR+EN
│
├── report/                  Rapport HTML
│   ├── generator.py         Orchestrateur Jinja2
│   ├── views/               (chantier 3) 5 vues thématiques composables
│   │   ├── economics.py     throughput + cost projection
│   │   ├── advanced_taxonomy.py  taxonomy comparison + lexical_modernization
│   │   ├── diagnostics.py   levers + image_predictive + baseline + worst_lines
│   │   ├── pipeline.py      DAG + error_absorption + incremental + audit
│   │   └── robustness.py    robustness projection
│   ├── *_render.py          26 renderers atomiques
│   ├── templates/           Jinja2 (10 vues HTML + partials)
│   ├── i18n/{fr,en}.json    410 clés
│   ├── glossary/            25 entrées YAML bilingues
│   └── vendor/              Chart.js embarqué
│
└── web/
    ├── app.py               FastAPI (2065 lignes — découpage reporté)
    ├── security.py
    ├── templates/, static/
    └── ...
```

## Fluxes principaux

### Bench OCR classique

```
CLI: picarones run --corpus DIR --engines tess,pero --profile standard
  ↓
load_corpus_from_directory(DIR)  → Corpus
_engine_from_name("tess")        → TesseractEngine (BaseOCREngine)
                                   (chantier 1 : refondu sur _run_with_native
                                    + _extract_raw_confidences)
  ↓
run_benchmark(corpus, engines, profile="standard")
  ↓ (profil active 12 hooks doc + 12 agrégateurs via builtin_hooks)
ProcessPoolExecutor / ThreadPoolExecutor
  ↓
_compute_document_result(doc, profile)
  ↓ (run_document_hooks itère sur les hooks actifs du profil)
DocumentResult (avec confusion, taxonomy, calibration, …)
  ↓ (run_corpus_aggregators)
EngineReport (avec aggregated_*)
  ↓
BenchmarkResult
  ↓
ReportGenerator.generate()
  ↓ (build_economics_view_html + build_advanced_taxonomy_view_html
     + build_diagnostics_view_html selon profil)
report.html (autonome, ~450 Ko)
```

### Bench pipeline composée (axe B, chantier 1 livré bout-en-bout)

```
CLI: picarones pipeline run examples/pipelines/ocr_to_alto.yaml --corpus DIR
  ↓
load_pipeline_spec_from_yaml()
  ↓
PipelineSpec :
  step "ocr"  : TesseractEngine          (IMAGE → TEXT)
  step "alto" : TextToAltoMonoRegion     (IMAGE+TEXT → ALTO)
  ↓
PipelineRunner.run(spec, document)
  ↓ (compute_at_junction((TEXT,TEXT)) → cer/wer/mer/wil)
  ↓ (compute_at_junction((ALTO,ALTO)) → alto_text_cer/wer/...)
PipelineResult avec junction_metrics par étape
  ↓
build_pipeline_report_html()  (rapport pipeline composée autonome)
```

## Documents complémentaires

- [`docs/profiles.md`](profiles.md) — les 7 profils de calcul du chantier 2.
- [`docs/cli-workflows.md`](cli-workflows.md) — les 15 commandes CLI.
- [`docs/views.md`](views.md) — les vues HTML disponibles dans le rapport.
- [`docs/user/reading-a-report.md`](user/reading-a-report.md) — guide
  utilisateur pour lire un rapport.
- [`docs/user/writing-a-pipeline-module.md`](user/writing-a-pipeline-module.md)
  — guide pour brancher un module tiers (`BaseModule`).
- [`docs/developer/narrative-engine.md`](developer/narrative-engine.md)
  — détecteurs narratifs : architecture, comment en ajouter.
- [`docs/developer/module-policy.md`](developer/module-policy.md) — manifest
  + audit pour modules contribués (Sprint 97).
- [`docs/case-studies/`](case-studies/) — 2 cas d'école (registres
  paroissiaux, édition critique).
- [`docs/roadmap/evolution-2026.md`](roadmap/evolution-2026.md) — plan
  d'évolution (axe A métrique + axe B pipelines composées).
