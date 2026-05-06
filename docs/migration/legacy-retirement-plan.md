# Plan de retrait complet du legacy — vers la 2.0

> **Décision stratégique** : pas de cohabitation legacy + rewrite à
> long terme.  La 2.0 est livrée **sans aucune ligne legacy**.
> L'arborescence cible (domain → formats → evaluation → pipeline →
> adapters → app → reports_v2 → interfaces) est unique.
>
> **Critère absolu** : zéro bricolage, zéro semi-rendu, zéro
> régression de comportement éditorial.  Une institution comme la
> BnF ne tolère pas un *partial rewrite*.
>
> **Pas de contrainte de date** : on livre quand tout est propre.
>
> **Document vivant** : ce plan est mis à jour à chaque phase
> achevée.  Toute exception ou découverte doit être inscrite ici.

## Définition de « done » universelle

Chaque phase est terminée quand **tous** les critères suivants sont
remplis :

1. **Code** : les modules legacy de la phase ont été soit migrés,
   soit déclarés sans équivalent et supprimés (avec justification).
2. **Tests** : tous les tests qui pointaient vers le legacy sont
   migrés vers le rewrite ; les nouveaux tests couvrent le rewrite
   à un niveau ≥ celui du legacy.
3. **Régression** : le harness `tests/regression/legacy_vs_rewrite/`
   prouve que le rewrite produit les mêmes résultats que le legacy
   sur les corpus de référence (tolérance ε explicite par métrique).
4. **Doc** : la doc utilisateur, opérationnelle et architecturale
   ne mentionne plus le legacy de la phase ; les chemins cassés
   `tests/architecture/test_doc_paths.py` baseline diminue.
5. **Lint** : `ruff check picarones/ tests/` clean.
6. **Suite complète** : `pytest tests/` 100 % vert sur 3 OS × 3
   versions Python (3.11, 3.12, 3.13).
7. **Coverage** : ≥ 85 %, pas de dégradation > 0,5 pt vs. la phase
   précédente.

## Phases

### Phase 0 — Foundation (en cours)

**Objectif** : poser les garde-fous qui rendent les 11 phases
suivantes **vérifiables** sans introduire de régression invisible.

**Livrables** :

- [x] `docs/migration/legacy-retirement-plan.md` (ce document) —
  inventaire complet, phases, acceptance criteria.
- [ ] `tests/regression/legacy_vs_rewrite/` — harness qui exécute
  legacy + rewrite sur 3 corpus de référence et compare bit-for-bit
  (avec ε explicite par métrique).
- [ ] `docs/migration/regression-tolerances.md` — table des
  tolérances acceptables par métrique (ex : CER ε = 0, narrative
  templates ε = 0 mais ordre des facts non-significatif, etc.).
- [ ] Test architectural `test_no_legacy_imports_in_rewrite.py` qui
  garantit qu'un module rewrite ne réintroduit jamais d'import
  legacy.

**Critère de fin** : harness vert sur 3 corpus de référence pour
les fonctionnalités déjà migrées (5 OCR, 4 LLM, 4 VLM, vues
canoniques).  Toute migration future doit ajouter son corpus de
régression.

### Phase 1 — Foundation conceptuelle (`core/`, `domain/`)

**Modules à migrer** :

| Legacy | Cible rewrite | Note |
|--------|---------------|------|
| `core/results.py` (677 LOC, `BenchmarkResult`/`EngineReport`/`DocumentResult` + 30 champs agrégés) | `domain/run_result.py` + champs agrégés en `Artifact` typés | Le plus critique |
| `core/facts.py` | déjà dans `domain/facts.py` | Vérifier parité |
| `core/pipeline.py` (legacy `PipelineSpec` + `BaseModule`) | `domain/pipeline_spec.py` + `domain/module_protocol.py` | Migration des callers |
| `core/modules.py` (`BaseModule`, `ArtifactType` 6 valeurs) | `domain/artifacts.py` (déjà 10 valeurs) + `domain/module_protocol.py` | Le superset |
| `core/metric_registry.py` + `metric_hooks.py` | `evaluation/registry/` (déjà migré) | Vérifier callers |
| `core/corpus.py` (`Document`, `Corpus`, `GTLevel`) | `domain/corpus.py` + `domain/documents.py` (déjà migré) | Modèle différent — convertisseur |
| `core/diff_utils.py` | `evaluation/utils/diff.py` (à créer) | Pure utility |
| `core/xml_utils.py` | `formats/xml_utils.py` (à créer) | Pure utility |
| `core/metrics.py` | `evaluation/metrics/_base.py` (à créer) | API helpers |

**Effort** : 5-8 jours-personnes.

**Acceptance** : aucun fichier `picarones.{adapters,evaluation,pipeline,
app,reports_v2,interfaces}` n'importe plus `picarones.core`.  Le
package `core/` peut être vidé en gardant uniquement des shims
`DeprecationWarning` (ou directement supprimé si aucun caller
externe ne le lit).

### Phase 2 — Statistics (`measurements/statistics/`)

**Modules** : `wilcoxon.py`, `friedman_nemenyi.py`, `bootstrap.py`,
`pareto.py`, `clustering.py`, `correlation.py`, `distributions.py`,
`cdd_render.py`.

**Cible** : nouveau sous-package `picarones/evaluation/statistics/`.

**Effort** : 5-7 jours.  Code mathématique pur, pas de couplage
applicatif.

**Acceptance** : régression bit-for-bit sur les outputs des 8 tests
statistiques + le rendu CDD SVG.

### Phase 3 — Narrative engine (`measurements/narrative/`)

**Modules** : `arbiter.py`, `renderer.py`, `registry.py` + 18
détecteurs en 6 familles (`ranking/`, `pareto/`, `stratum/`,
`quality/`, `history/`, `ensemble/`) + 36 templates YAML FR/EN +
`core/facts.py`.

**Cible** : `picarones/reports_v2/narrative/` (le narratif est de
la **présentation**, pas du domaine — il vit du côté rapport, pas
de l'évaluation).

**Effort** : 8-12 jours.  Le bloc le plus délicat — 18 détecteurs
chacun avec garde-fous anti-hallucination, arbitre de cohérence
inter-détecteurs, renderer Jinja par templates YAML.

**Acceptance** : régression bit-for-bit sur la synthèse rendue
pour chaque cas-test legacy `test_sprint{16,19,29,36,39,42,44,46,
51,56,73,81,90,92}_*.py` (les sprints qui ont introduit chaque
détecteur).

### Phase 4 — 35 mesures legacy (`measurements/*.py`)

**Modules** (ordre par cluster thématique, chaque cluster = un PR/sprint) :

#### 4.A — Philologique (8 modules) — 5 jours

`mufi.py`, `abbreviations.py`, `early_modern_typography.py`,
`modern_archives.py`, `roman_numerals.py`, `unicode_blocks.py`,
`equivalence_profile.py`, `philological_hooks.py`.

#### 4.B — NER + readability (6 modules) — 4 jours

`ner.py`, `ner_backends.py`, `readability.py`, `readability_hooks.py`,
`searchability.py`, `searchability_hooks.py`.

#### 4.C — Structurel (5 modules) — 3 jours

`reading_order.py`, `structure.py`, `alto_metrics.py`, `line_metrics.py`
(legacy), `numerical_sequences.py` + `numerical_sequences_hooks.py`.

#### 4.D — Robustness, reliability, history (4 modules) — 4 jours

`robustness.py`, `reliability.py`, `history.py`, `specialization.py`.

#### 4.E — Pipeline metrics (3 modules) — 3 jours

`pipeline_benchmark.py`, `pipeline_comparison.py`,
`pipeline_spec_loader.py` — opèrent sur le `PipelineSpec` legacy.
À fondre dans la couche `app/services/` du rewrite.

#### 4.F — Utility / hooks (9 modules) — 4 jours

`builtin_hooks.py`, `builtin_metrics.py`, `metrics.py`, `char_scores.py`,
`cost_projection.py`, `difficulty.py`, `taxonomy.py`,
`taxonomy_intra_doc.py`, `runner/*` (sous-package).

**Cible** : chaque mesure migre vers `evaluation/metrics/<thème>/`
ou un sous-package adapté.  Enregistrement via la `MetricRegistry`
typée (signature `(input_type, output_type) → score`).

**Effort total Phase 4** : 23-28 jours.

**Acceptance** : régression bit-for-bit sur les ~5000 tests existants
qui pointent vers ces métriques.

### Phase 5 — Reports HTML (`report/`)

**Modules** :

- 22 renderers thématiques : `baseline_render.py`, `calibration_render.py`,
  `difficulty_render.py`, `error_absorption_render.py`,
  `image_predictive_render.py`, `incremental_comparison_render.py`,
  `inter_engine_render.py`, `levers_render.py`,
  `lexical_modernization_render.py`, `longitudinal_render.py`,
  `marginal_cost_render.py`, `module_audit_render.py`,
  `multirun_stability_render.py`, `ner_render.py`,
  `numerical_sequences_render.py`, `philological_render.py`,
  `pipeline_dag_render.py`, `pipeline_render.py`,
  `rare_token_recall_render.py`, `readability_render.py`,
  `robustness_projection_render.py`, `searchability_render.py`,
  `specialization_render.py`, `stratification_render.py`,
  `taxonomy_comparison_render.py`, `taxonomy_cooccurrence_render.py`,
  `taxonomy_intra_doc_render.py`, `throughput_render.py`,
  `worst_lines_render.py`.
- 5 vues : `views/{advanced_taxonomy,diagnostics,economics,
  pipeline,robustness}.py`.
- `generator.py` (orchestrateur), `comparison.py`, `snapshot.py`,
  `assets.py`, `colors.py`, `render_helpers.py`, `report_data/`.
- `templates/` (~10 fichiers Jinja2), `glossary/` (2 YAML, 25
  entrées), `i18n/`, `vendor/`.

**Cible** : `picarones/reports_v2/html/views/<theme>.py` + helpers
partagés dans `reports_v2/html/_helpers/`.  Glossaire dans
`reports_v2/html/glossary/`.  Templates Jinja2 dans
`reports_v2/html/templates/`.

**Effort** : 12-18 jours.

**Acceptance** : régression bit-for-bit sur le HTML produit pour
les 3 corpus de référence.  Aucun renderer legacy laissé.

### Phase 6 — Pipelines OCR+LLM (`pipelines/`)

**Modules** : `pipelines/base.OCRLLMPipeline` (3 modes), `pipelines/
over_normalization.detect_over_normalization`.

**Cible** :

- Les 3 modes deviennent des `PipelineSpec` YAML composés (OCR
  adapter → LLM adapter avec `inputs_from`).
- `over_normalization` devient une métrique enregistrée dans
  `evaluation/metrics/over_normalization.py`.

**Effort** : 3-5 jours.

**Acceptance** : les 3 callers internes (`web/benchmark_utils.py`,
`measurements/runner/document.py`, `fixtures.py`) consomment des
`PipelineSpec` YAML rewrite.

### Phase 7 — Modules officiels (`modules/`)

**Module** : `modules/alto_text_to_mono_region.TextToAltoMonoRegion`
(310 LOC) — baseline TEXT → ALTO.

**Cible** : `picarones/formats/alto/baseline_reconstruction.py` ou
`picarones/evaluation/projectors/text_to_alto.py` (selon où la
sémantique colle le mieux).

**Effort** : 1 jour.

### Phase 8 — Importers (`extras/importers/`)

**Modules** : `iiif.py`, `gallica.py`, `escriptorium.py`, `_http.py`,
`_fallback_log.py`.

**Cible** : `picarones/adapters/corpus/{iiif,gallica,escriptorium}.py`
+ helpers partagés dans `adapters/corpus/_http.py`.

**Effort** : 3-5 jours.

### Phase 9 — Web UI riche (`web/`)

**Modules** : 9 routers (`config`, `engines`, `history`, `home`,
`importers`, `normalization`, `reports`, `synthesis`, `system`) +
utilitaires (`benchmark_utils.py`, `engine_utils.py`,
`corpus_utils.py`, `config_utils.py`, `state.py`, `security.py`,
`models.py`, `jobs.py`, `maintenance.py`, `app.py`) + templates
Jinja2.

**Cible** : `picarones/interfaces/web/routers/<router>.py` + utils
partagés dans `interfaces/web/_utils/` + templates dans
`interfaces/web/templates/`.

**Effort** : 8-12 jours.

**Acceptance** : régression sur tous les `tests/web/test_sprint*.py`
existants.  L'UI riche (sélecteur moteurs dynamique, gallery,
stratification, narrative inline, browse corpus) doit produire les
mêmes pages HTML.

### Phase 10 — CLI complète (`cli/`)

**Commandes** : 13 commandes legacy non couvertes (`metrics`,
`engines`, `info`, `demo`, `diagnose`, `economics`, `edition`,
`compare`, `import` group, `serve`, `history`, `robustness`,
`pipeline` group avec sous-commandes `run` et `compare`).

**Cible** : `picarones/interfaces/cli/<command>.py`.  L'entry point
`console_scripts` du `pyproject.toml` doit pointer sur
`picarones.interfaces.cli:cli` (à la place de `picarones.cli:cli`).

**Effort** : 4-6 jours.

### Phase 11 — Retrait final + release 2.0

- Suppression des 10 packages legacy.
- Suppression des shims `DeprecationWarning` introduits aux phases
  précédentes.
- Mise à jour du `pyproject.toml` (`console_scripts`,
  `[project.urls]`).
- Rédaction du CHANGELOG 2.0 final avec liste exhaustive des
  breaking changes (les utilisateurs externes ont eu
  `DeprecationWarning` à chaque phase).
- Génération SBOM + signature SLSA Level 3 (cf.
  `docs/operations/supply-chain.md`).
- Bump `_version.py` et tag `v2.0.0`.

**Effort** : 3-5 jours.

## Estimation totale

| Phase | Effort min | Effort max |
|-------|------------|------------|
| 0 | 2 j | 3 j |
| 1 | 5 j | 8 j |
| 2 | 5 j | 7 j |
| 3 | 8 j | 12 j |
| 4 | 23 j | 28 j |
| 5 | 12 j | 18 j |
| 6 | 3 j | 5 j |
| 7 | 1 j | 1 j |
| 8 | 3 j | 5 j |
| 9 | 8 j | 12 j |
| 10 | 4 j | 6 j |
| 11 | 3 j | 5 j |
| **Total** | **77 j** | **110 j** |

Soit **3,5 à 5 mois** d'effort focalisé en mode développeur unique.
Aucune contrainte de date — on livre quand c'est propre.

## Stratégie de régression — invariant non négociable

À chaque phase :

1. **Avant** : exécuter le harness legacy sur 3 corpus de référence
   (small / medium / large) → capture des outputs en JSON / HTML
   bit-for-bit.
2. **Pendant** : réécrire la fonctionnalité dans le rewrite.
3. **Après** : exécuter le harness rewrite et **diff** vs. snapshot
   legacy.
4. **Tolérance** : explicite par métrique dans
   `docs/migration/regression-tolerances.md`.  Tout écart non
   tolerance = régression à corriger avant merge.

Cela évite le piège classique du rewrite : *« ça compile, ça tourne,
mais le CER a glissé de 0,002 par doc »*.

## Anti-bricolage — règles

1. **Pas de double API** : pendant la migration d'un module, on ne
   garde **pas** le legacy en parallèle dans le code de production.
   Soit on importe l'ancien, soit le nouveau.  Le harness de
   régression suffit pour valider.
2. **Pas de shim sans date de retrait** : tout `DeprecationWarning`
   introduit doit être inscrit dans le CHANGELOG avec date de
   retrait (la 2.0).
3. **Pas de TODO dans le code mergé** : un TODO = une issue ouverte
   référencée par numéro.
4. **Pas de copié-collé** : si une logique apparaît dans deux
   modules, extraire en helper partagé dès la deuxième occurrence.
5. **Pas de god-module** : `tests/architecture/test_file_budgets.py`
   reste l'autorité.

## Statut

| Phase | Statut |
|-------|--------|
| 0 | 🟡 En cours |
| 1-11 | ⚪ À démarrer |

**Dernière mise à jour** : 2026-05.
