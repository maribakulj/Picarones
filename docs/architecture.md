# Architecture Picarones — manifeste

Picarones est un **banc d'essai** pour pipelines OCR/HTR sur documents
patrimoniaux. Le code est organisé en **3 cercles concentriques** avec
une règle de dépendance stricte : les flèches d'import vont uniquement
de l'extérieur vers l'intérieur.

```
   Cercle 3 (extras, report, cli, web)
   │
   ▼
   Cercle 2 (measurements, engines, llm, pipelines, modules)
   │
   ▼
   Cercle 1 (core)
```

## Cercle 1 — `picarones/core/` : abstractions de domaine

Pas de logique métier, pas d'I/O. Uniquement des **contrats** que les
cercles supérieurs implémentent.

| Module | Contenu |
|---|---|
| `modules.py` | `BaseModule`, `ArtifactType`, `validate_inputs`/`validate_outputs` |
| `corpus.py` | `Document`, `Corpus`, `GTLevel`, payloads typés (`TextGT`, `AltoGT`, `PageGT`, `EntitiesGT`, `ReadingOrderGT`) |
| `results.py` | `DocumentResult`, `EngineReport`, `BenchmarkResult` |
| `metric_registry.py` | `MetricSpec`, `register_metric`, `select_metrics`, `compute_at_junction` |
| `metric_hooks.py` | `register_document_metric`, `register_corpus_aggregator`, profils de calcul |
| `pipeline.py` | `PipelineRunner`, `PipelineSpec`, `PipelineStep` (DAG de modules) |
| `facts.py` | `Fact`, `FactType`, `FactImportance`, `DetectorRegistry` |

**Règle** : un module du cercle 1 peut importer un autre module du
cercle 1. Il ne peut **rien** importer des cercles 2 ou 3.

## Cercle 2 — implémentations officielles

Les implémentations distribuées par défaut dans le package `picarones`.

### `picarones/measurements/` — métriques (~50 modules)

| Catégorie | Modules |
|---|---|
| Coeur | `metrics.py`, `statistics/` (sous-package), `runner.py`, `builtin_hooks.py`, `builtin_metrics.py`, `normalization.py` |
| Erreurs | `confusion.py`, `taxonomy.py`, `taxonomy_comparison.py`, `taxonomy_cooccurrence.py`, `taxonomy_intra_doc.py` |
| Lignes/structure | `line_metrics.py`, `structure.py`, `worst_lines.py`, `char_scores.py` |
| Calibration/fiabilité | `calibration.py`, `reliability.py`, `hallucination.py` |
| Image | `image_quality.py`, `image_predictive.py`, `difficulty.py` |
| Robustesse | `robustness.py`, `robustness_projection.py` |
| Inter-moteurs | `inter_engine.py`, `specialization.py` |
| Statistique avancée | `baseline_comparison.py`, `longitudinal.py`, `incremental_comparison.py` |
| Contenu | `searchability.py`, `numerical_sequences.py`, `rare_tokens.py`, `readability.py` |
| Structure ALTO | `layout.py`, `reading_order.py`, `ner.py`, `ner_backends.py`, `error_absorption.py` |
| Économie | `cost_projection.py`, `marginal_cost.py`, `pricing.py`, `throughput.py` |
| Philologie historique | `mufi.py`, `abbreviations.py`, `unicode_blocks.py`, `early_modern_typography.py`, `modern_archives.py`, `roman_numerals.py`, `lexical_modernization.py`, `philological_runner.py` |
| Pipelines composées | `pipeline_benchmark.py`, `pipeline_comparison.py`, `pipeline_spec_loader.py`, `alto_metrics.py` |
| Divers | `equivalence_profile.py`, `levers.py`, `module_policy.py`, `history.py` |
| Runners adaptifs | `readability_runner.py`, `searchability_runner.py`, `numerical_sequences_runner.py` |
| Narratif | `narrative/` (arbiter, renderer, registry, 18 détecteurs en 6 familles) |

### `picarones/engines/` — adapters OCR (5)

`tesseract.py`, `pero_ocr.py`, `mistral_ocr.py`, `google_vision.py`,
`azure_doc_intel.py`. Tous héritent de `picarones.core.engine.BaseOCREngine`
(qui vit dans `engines/base.py` pour la lisibilité).

### `picarones/llm/` — adapters LLM (4)

`mistral_adapter.py`, `openai_adapter.py`, `anthropic_adapter.py`,
`ollama_adapter.py`. Interface commune dans `base.py`.

### `picarones/pipelines/` — pipelines OCR+LLM intégrés

`base.py` (`OCRLLMPipeline`, qui hérite de `BaseOCREngine`),
`over_normalization.py`.

### `picarones/modules/` — modules `BaseModule` officiels

Démonstrateurs qui prouvent l'axe B du plan d'évolution :
`alto_text_to_mono_region.py`.

## Cercle 3 — extensions et présentation

### `picarones/extras/importers/` — connecteurs corpus

`iiif.py`, `gallica.py`, `htr_united.py`, `huggingface.py`,
`escriptorium.py`, `_http.py`. Plugins pluggable, certains expérimentaux.

### `picarones/report/` — rendu HTML

| Sous-dossier | Contenu |
|---|---|
| `generator.py` | Orchestration Jinja2 |
| `views/` | 5 vues thématiques (economics, advanced_taxonomy, diagnostics, pipeline, robustness) |
| `templates/` | Jinja2 (base, header, footer, vues, partials) |
| `i18n/` | FR/EN |
| `glossary/` | 25 entrées bilingues |
| `vendor/` | Chart.js |
| `*_render.py` | ~22 renderers (calibration, NER, Pareto, Sankey, etc.) |

Pas de sous-dossier `extras/render/` — tout le rendu est ici.

### `picarones/cli/` — Click (7 fichiers)

Point d'entrée `picarones.cli:cli` (référencé dans `pyproject.toml`).
15 sous-commandes : `run`, `diagnose`, `economics`, `edition`,
`compare`, `metrics`, `engines`, `info`, `report`, `demo`, `serve`,
`history`, `robustness`, `pipeline run/compare`, `import`.

### `picarones/web/` — FastAPI

Interface web (`app.py`).

## Données

| Dossier | Rôle |
|---|---|
| `picarones/prompts/` | Prompts LLM versionnés (8 fichiers, FR + EN) |
| `picarones/data/` | Tables indicatives (pricing, etc.) |
| `picarones/fixtures.py` | Corpus de démonstration |

## Règles de migration

1. **Pas de shim** : un module a un seul emplacement physique. Les
   imports pointent directement vers la vraie source.
2. **Pas de double API** : une fonction a un seul nom canonique. Les
   alias historiques sont supprimés et les tests mis à jour.
3. **Frontières strictes** : si un module Y du cercle N importe le
   module X, alors le cercle de X est ≤ N. Une exception
   pragmatique : `engines/base.py` est conceptuellement cercle 1
   mais physiquement dans `engines/` pour rester avec ses
   implémentations.
4. **Les dépendances optionnelles** (`scipy`, `spacy`, etc.) sont
   gérées par try/except à l'import — pas par shim.

## Tests

Organisés par cercle : `tests/core/`, `tests/measurements/`,
`tests/engines/`, `tests/extras/`, `tests/report/`,
`tests/integration/` (tests E2E croisant plusieurs cercles).

Un test du cercle N **n'importe pas** les implémentations des
cercles > N (sauf `tests/integration/`).

## Convention de découpage des modules > 400 lignes

Quand un module Python dépasse 400 lignes ET contient plusieurs
responsabilités disjointes, le découper en **sous-package** plutôt
qu'en plusieurs modules à plat. Modèle de référence :
[`picarones/measurements/statistics/`](../picarones/measurements/statistics/)
issu du sprint « découpage de statistics.py » (mai 2026).

Convention :

1. **Renommer** `X.py` en `X/__init__.py` via `git mv` (préserve
   l'historique du fichier original).
2. **Créer** dans `X/` un sous-module par famille fonctionnelle
   (`bootstrap.py`, `wilcoxon.py`, `friedman_nemenyi.py`, etc.).
   Chaque sous-module doit faire moins de ~400 lignes ; sinon
   re-décomposer.
3. **`X/__init__.py`** ne contient QUE des ré-exports rétrocompat —
   tous les symboles publics de l'ancien `X.py` doivent rester
   importables via `from picarones.X import …`. Les symboles privés
   ré-exportés doivent être ceux **réellement** consommés par les
   tests (vérifié par grep, pas par supposition).
4. **`__all__`** explicite dans chaque sous-module et dans le
   `__init__.py`.
5. **Tests architecture** (`tests/architecture/test_*.py`) doivent
   continuer à passer : si nécessaire, étendre `_measurements_modules()`
   ou `_imports_target_*` pour reconnaître les sous-packages.
6. **Préfixer les modules de rendu** par leur domaine
   (`cdd_render.py` plutôt que `render_cdd.py`) pour cohérence avec
   `picarones/report/*_render.py`.

**Quand NE PAS découper** : si les responsabilités sont fortement
couplées (ex: un orchestrateur qui appelle 12 sous-fonctions au
même endroit), le maintien dans un seul fichier > 400 lignes est
acceptable. Le budget par fichier (`tests/architecture/test_file_budgets.py`)
documente ces dérogations conscientes.
