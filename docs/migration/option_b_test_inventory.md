# Inventaire des tests à migrer — Option B (RunOrchestrator)

Document de référence pour la Phase B4 du chantier de migration
`run_benchmark_via_service` → `RunOrchestrator.execute(RunSpec)`.

Établi en Phase B0 par parcours exhaustif de `tests/` :
- `grep -rln "run_benchmark_via_service" tests/` → 10 fichiers (catégorie A)
- `grep -rln "BenchmarkResult" tests/` → 20 fichiers, dont 10 en intersection avec A et 10 en catégorie B seule

Aucun fichier ne **patche** `run_benchmark_via_service` via `monkeypatch.setattr`
ou `mock.patch` — vérifié en Phase B0. Tous les call sites sont des appels
directs. Cela simplifie la stratégie de migration : pas de cibles indirectes
à recâbler.

---

## Catégorie A — Appellent `run_benchmark_via_service` directement

Ces tests doivent être migrés vers `RunOrchestrator.execute(spec)` ou vers le
helper `build_run_spec_from_engines()` (Phase B3.1) pour les cas qui partent
d'instances d'adapter en mémoire.

| # | Fichier | Taille | Occurrences | Priorité B4 | Notes |
|---|---|---|---|---|---|
| A1 | `tests/app/test_sprint_d2b_partial_dir_resume.py` | 506 LOC | 12 | **Haute** | Teste le resume `partial_dir`. Doit valider le port vers `_orchestrator_partial.py` (Phase B2.3). Cœur de la non-régression. |
| A2 | `tests/app/test_sprint_d2cdef_features.py` | 473 LOC | 14 | **Haute** | Teste les 7 paramètres étendus (`profile`, `entity_extractor`, `cancel_event`, etc.). Doit valider chaque feature portée en Phase B2. |
| A3 | `tests/web/test_sprint6_web_interface.py` | 1392 LOC | 10 | **Haute** | Test d'intégration web. Confirmera que la migration `run_benchmark_thread_v2` ne casse rien côté UI. |
| A4 | `tests/app/test_character_analysis_in_runner.py` | 246 LOC | 12 | Moyenne | Teste l'analyse caractère par engine. Conversion mécanique. |
| A5 | `tests/app/test_sprint_h2b_canonical_in_runner.py` | 191 LOC | 9 | Moyenne | Teste l'extraction du `CANONICAL_DOCUMENT`. À adapter au nouveau ViewExecutor. |
| A6 | `tests/evaluation/test_public_api.py` | — | 7 | Moyenne | API publique. Inclura un test de présence pour `RunOrchestrator`. |
| A7 | `tests/evaluation/metrics/test_sprint12_nouvelles_fonctionnalites.py` | 288 LOC | 4 | Basse | Conversion mécanique. |
| A8 | `tests/evaluation/metrics/test_sprint_a14_s1_normalization_propagation.py` | — | 2 | Basse | Vérifie `normalization_profile` — valide la Phase B2.5 (propagation via `EvaluationView`). |
| A9 | `tests/evaluation/test_metric_hooks.py` | — | 1 | Basse | Trivial. Conversion en 1 ligne. |
| A10 | `tests/architecture/test_file_budgets.py` | — | (référence uniquement) | Basse | Budgets des modules `_benchmark_*.py` à actualiser après Phase B2/B7. |

**Total catégorie A** : 10 fichiers, ~3500 LOC, ~71 occurrences.

---

## Catégorie B — Consomment `BenchmarkResult` mais n'appellent pas le runner

Ces tests **ne nécessitent aucun changement** tant que le converter
`RunResult → BenchmarkResult` (`_benchmark_converter.py`) reste en place
après la migration. Ils consomment soit un `BenchmarkResult` construit
manuellement (fixture), soit un `BenchmarkResult` issu d'un appel au runner
fait dans une fixture partagée.

| # | Fichier | Rôle |
|---|---|---|
| B1 | `tests/golden/test_s5_benchmark_result_json_stable.py` | Round-trip JSON stable. Inchangé tant que `BenchmarkResult.from_json_object`/`to_dict` restent. |
| B2 | `tests/reports/test_report.py` | Rendu HTML. Inchangé tant que `ReportGenerator(result)` accepte `BenchmarkResult`. |
| B3 | `tests/reports/test_extra_metrics.py` | Métriques additionnelles attachées au rapport. |
| B4 | `tests/reports/test_sprint72_worst_lines.py` | Worst-N lines (consomme `BenchmarkResult` non-compacté). |
| B5 | `tests/evaluation/metrics/test_results.py` | API `MetricsResult` / `aggregate_metrics`. |
| B6 | `tests/evaluation/metrics/test_sprint36_ensemble_narrative.py` | Narrative engine. Lit `benchmark_data` dict. |
| B7 | `tests/evaluation/metrics/test_sprint44_median_default.py` | Médiane/Pareto. |
| B8 | `tests/evaluation/metrics/test_sprint45_stratification.py` | Stratification du corpus. |
| B9 | `tests/evaluation/test_sprint14_robust_filtering.py` | Filtre robustesse. |
| B10 | `tests/adapters/corpus/test_sprint8_escriptorium_gallica.py` | Importer eScriptorium / Gallica. |
| B11 | `tests/integration/test_importer_fallback_wiring.py` | Fallback importer. Test d'intégration. |
| B12 | `tests/integration/test_s5_disk_full_simulation.py` | Disque plein. |
| B13 | `tests/security/test_phase1_post_rewrite_wiring.py` | Sécurité post-rewrite. |
| B14 | `tests/security/test_s1_xss_in_reports.py` | XSS dans rapports. |
| B15 | `tests/test_minimal_install.py` | Installation minimale (smoke test). |
| B16 | `tests/web/test_sprint28_ux_save_compare.py` | UX save/compare web. |

**Total catégorie B** : 16 fichiers — **AUCUN changement requis** pour Option B.

---

## Catégorie C — Tests qui utilisent déjà `RunOrchestrator`

Pour information : ces tests servent de modèle/template pour la migration.

| # | Fichier | Rôle |
|---|---|---|
| C1 | `tests/app/test_run_orchestrator.py` | Tests unitaires complets de `RunOrchestrator`. Modèle de référence. |
| C2 | `tests/integration/test_runner_profiles.py` | Profils de hooks via le `RunOrchestrator`. |
| C3 | `tests/integration/test_html_views.py` | Vues HTML via `RunOrchestrator`. |
| C4 | `tests/integration/test_narrative_and_views.py` | Narrative engine + vues. |
| C5 | `tests/integration/test_engines_and_llm.py` | Engines + LLM via `RunOrchestrator`. |

---

## Stratégie globale Phase B4

1. **Étape 1** (0.5 j) — Fixture partagée dans `tests/conftest.py` :
   - `make_minimal_corpus_zip()` — corpus zip 2 docs déterministe
   - `run_orchestrator_factory(tmp_path)` — `RunOrchestrator(workspace)`
   - `build_minimal_run_spec(corpus_zip, output_dir, adapters)` — helper
   - `assert_benchmark_results_equal(a, b, *, ignore=("started_at", "completed_at"))`

2. **Étape 2** (2 j) — Migration catégorie A en commençant par la priorité haute :
   - A1 (resume partial) + A2 (features étendues) + A3 (web) → 1 j
   - A4, A5, A6 → 0.5 j
   - A7, A8, A9 → 0.5 j (trivial)

3. **Étape 3** (0.5 j) — Mise à jour `tests/architecture/test_file_budgets.py` (A10) :
   - Marquer les modules `_benchmark_*.py` comme deprecated
   - Augmenter le budget de `run_orchestrator.py` (~+300 LOC après Phase B2)

4. **Étape 4** (1 j) — Run complet de la suite + ajustements :
   - `pytest tests/ -q --tb=short`
   - Snapshot d'invariance (`test_migration_invariance.py`) doit rester vert
   - Feature parity (`test_run_orchestrator_feature_parity.py`) toutes vertes

**Estimation totale Phase B4** : 3-4 jours, conforme au plan.

---

## Validation post-migration

À la fin de la Phase B4 :

```bash
# Tous verts
python -m pytest tests/ -q --tb=short

# Snapshot d'invariance inchangé
python -m pytest tests/integration/test_migration_invariance.py -v

# 7 features de parity portées
python -m pytest tests/app/services/test_run_orchestrator_feature_parity.py -v

# Aucune occurrence résiduelle de run_benchmark_via_service hors module legacy
grep -rln "run_benchmark_via_service" tests/ picarones/ | \
  grep -v "benchmark_runner.py\|_benchmark_" | \
  wc -l
# Attendu : 0 (ou 1 si on garde un export public deprecated dans __init__.py)
```
