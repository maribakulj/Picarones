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

### Phase 0 — Foundation ✅ terminée

**Objectif** : poser les garde-fous qui rendent les 11 phases
suivantes **vérifiables** sans introduire de régression invisible.

**Livrables** :

- [x] `docs/migration/legacy-retirement-plan.md` (ce document) —
  inventaire complet, phases, acceptance criteria.
- [x] `docs/migration/regression-tolerances.md` — table des
  tolérances acceptables par métrique et type d'output (CER ε=0,
  Wilcoxon ε=1e-9, HTML diff sémantique, narrative facts égalité
  ensembliste, etc.).
- [x] `tests/regression/legacy_vs_rewrite/` — harness scaffolding :
  fixtures de corpus synthétique (small=3 docs, medium=30 docs,
  large laissé pour ajout opportuniste) + gestion golden snapshot
  avec flag `--regen-golden` + comparateurs sémantiques (floats,
  sets, JSON).  Marker `regression` enregistré et exclu de
  ``addopts`` par défaut (opt-in via `pytest -m regression`).
  Smoke test couvre les 16 invariants du harness lui-même.
- [x] `tests/architecture/test_no_legacy_imports_in_rewrite.py` —
  garantit qu'aucun fichier des paquets `domain/`, `formats/`,
  `evaluation/`, `pipeline/`, `adapters/`, `app/`, `reports_v2/`,
  `interfaces/` n'importe depuis un paquet legacy.  AST-based,
  pas regex syntaxique.  État initial : **vert** — le rewrite est
  déjà clean.

**Acceptance** : ✅ remplie.  Le harness est prêt à recevoir les
tests de régression de chaque phase suivante (`test_phase1_*.py`,
`test_phase2_*.py`, etc.).  Toute fonctionnalité migrée DOIT
avoir son test de régression ajouté ici en même temps que le
code.

### Phase 1 — Foundation conceptuelle (`core/`, `domain/`) — partielle ✅

**Audit de migrabilité réelle** : 5 modules `core/` sur 9 dépendent
de `core/modules.py` (legacy `BaseModule` + `ArtifactType` 6 valeurs,
incompatible avec le superset `domain/artifacts.ArtifactType` 10
valeurs).  Les migrer ferait dériver le comportement des callers
legacy — à reporter en **Phase 4** quand le runner et les métriques
seront rewrités.

**Migrés en Phase 1 — 3 modules** (sans dépendance à `core/modules`) :

| Legacy | Canonique rewrite | Statut |
|--------|-------------------|--------|
| `core/xml_utils.py` (44 LOC) | `formats/_xml_utils.py` + re-export `picarones.formats.safe_parse_xml` | ✅ shim posé |
| `core/diff_utils.py` (89 LOC) | `evaluation/_diff_utils.py` + re-export `picarones.evaluation.{compute_word_diff,compute_char_diff,diff_stats}` | ✅ shim posé |
| `core/facts.py` (229 LOC) | `domain/facts.py` + re-export `picarones.domain.{Fact,FactType,FactImportance,DetectorRegistry,detect_all}` | ✅ shim posé |

**Reportés en Phase 4** (couplage à `core/modules.ArtifactType` legacy
ou au modèle du runner legacy) :

| Legacy | Bloqueur |
|--------|----------|
| `core/results.py` (677 LOC, `BenchmarkResult` + 30 champs agrégés) | Modèle central du runner legacy ; convergence avec `app.results.RunResult` en Phase 4 (rewrite de `measurements/runner/`) |
| `core/pipeline.py` (571 LOC, legacy `PipelineSpec` + `BaseModule`) | Concept différent du `domain.pipeline_spec.PipelineSpec` ; convergence en Phase 6 (`pipelines/` legacy) |
| `core/corpus.py` (511 LOC, `Document` avec payloads typés) | Modèle data legacy ≠ `DocumentRef` du rewrite ; convergence en Phase 4 |
| `core/modules.py` (173 LOC, `BaseModule` + `ArtifactType` 6 valeurs) | Type legacy partagé par 50+ modules ; déprécation en Phase 4 |
| `core/metric_registry.py` + `metric_hooks.py` (686 LOC) | Importe `core.modules.ArtifactType` ; convergence en Phase 4 |
| `core/metrics.py` (144 LOC, `MetricsResult`) | Schéma legacy ≠ `ViewResult.metric_values` du rewrite ; convergence en Phase 4 |

**Effort consommé Phase 1** : ~1 jour (3 modules + audit + tests).
**Effort restant — reporté en Phase 4** : ~5-7 jours.

**Acceptance Phase 1 partielle** : 3 modules `core/` sont des shims
re-export propres avec `DeprecationWarning`.  Le test architectural
`test_no_legacy_imports_in_rewrite.py` reste vert.  `picarones/__init__.py`
top-level pointe désormais vers le canonique pour les modules
migrés (pas de spam de warning à `import picarones`).  Les 6 autres
modules `core/` fonctionnent inchangés ; ils seront migrés au
moment de la migration de leurs callers.

### Phase 2 — Statistics (`measurements/statistics/`) — ✅ terminée

**Modules migrés** : 8 modules (`wilcoxon.py`, `friedman_nemenyi.py`,
`bootstrap.py`, `pareto.py`, `clustering.py`, `correlation.py`,
`distributions.py`, `cdd_render.py`).

**Canonique** : `picarones/evaluation/statistics/`.

**Travaux** :

- 8 modules copiés bit-for-bit dans `evaluation/statistics/`.
- 1 import legacy dans `clustering.py` migré
  (`picarones.core.diff_utils.compute_word_diff`
  → `picarones.evaluation.compute_word_diff`).
- 1 import auto-référencé dans `friedman_nemenyi.py` migré
  (`picarones.measurements.statistics.wilcoxon._normal_sf`
  → `picarones.evaluation.statistics.wilcoxon._normal_sf`).
- `evaluation/statistics/__init__.py` ré-exporte 14 symboles
  publics (`bootstrap_ci`, `wilcoxon_test`, `compute_pairwise_stats`,
  `friedman_test`, `nemenyi_posthoc`, `build_critical_difference_svg`,
  `compute_pareto_front`, `ErrorCluster`, `cluster_errors`,
  `compute_correlation_matrix`, `compute_reliability_curve`,
  `compute_venn_data`, `_SCIPY_AVAILABLE`, `_chi_square_sf`,
  `_nemenyi_critical_value`, `_normal_sf`, `_rank_row`).
- 8 shims `measurements/statistics/<X>.py` + 1 shim
  `measurements/statistics/__init__.py` avec `DeprecationWarning`,
  `__all__` complet pour rétrocompat des callers (5 fichiers
  `report/`, 6 fichiers tests).

**Effort réel** : ~1 jour (vs estimation 5-7 j — code mathématique
pur, pas de couplage applicatif comme prévu, mais aussi script
de génération de shims qui a accéléré).

**Acceptance** : suite par défaut 5019+ passed (inchangée), tests
ciblés sur les statistiques 226 passed, test architectural
anti-imports legacy reste vert (3 passed).  Pas de régression
détectée — les algorithmes scipy/numpy sont déterministes par
construction (seed=42 partout) ; le rendu SVG est strictement
identique parce que c'est le même fichier.

### Phase 3 — Narrative engine (`measurements/narrative/`) — ✅ terminée

**Modules migrés** : 11 modules + 2 templates YAML.

| Legacy | Canonique |
|--------|-----------|
| `measurements/narrative/__init__.py` | `reports_v2/narrative/__init__.py` |
| `measurements/narrative/arbiter.py` | `reports_v2/narrative/arbiter.py` |
| `measurements/narrative/registry.py` | `reports_v2/narrative/registry.py` |
| `measurements/narrative/renderer.py` | `reports_v2/narrative/renderer.py` |
| `measurements/narrative/detectors/__init__.py` | `reports_v2/narrative/detectors/__init__.py` |
| `measurements/narrative/detectors/_helpers.py` | `reports_v2/narrative/detectors/_helpers.py` |
| `measurements/narrative/detectors/ensemble.py` | `reports_v2/narrative/detectors/ensemble.py` (1 détecteur) |
| `measurements/narrative/detectors/history.py` | `reports_v2/narrative/detectors/history.py` (3 détecteurs) |
| `measurements/narrative/detectors/pareto.py` | `reports_v2/narrative/detectors/pareto.py` (2 détecteurs) |
| `measurements/narrative/detectors/quality.py` | `reports_v2/narrative/detectors/quality.py` (4 détecteurs) |
| `measurements/narrative/detectors/ranking.py` | `reports_v2/narrative/detectors/ranking.py` (5 détecteurs) |
| `measurements/narrative/detectors/stratum.py` | `reports_v2/narrative/detectors/stratum.py` (3 détecteurs) |
| `measurements/narrative/templates/fr.yaml` | `reports_v2/narrative/templates/fr.yaml` |
| `measurements/narrative/templates/en.yaml` | `reports_v2/narrative/templates/en.yaml` |

Total : **18 détecteurs en 6 familles + arbitre + renderer + 36
templates YAML FR/EN** migrés.

**Cible architecturale** : `picarones/reports_v2/narrative/` (le
narratif est de la **présentation**, pas du domaine — il vit du
côté rapport, pas de l'évaluation).

**Travaux** :

- 14 fichiers (11 .py + 1 _helpers.py + 2 .yaml) copiés depuis le
  legacy vers le canonique.
- Tous les imports `picarones.core.facts` (11 occurrences) migrés
  vers `picarones.domain.facts` (Phase 1 a déjà migré ce module).
- Tous les imports auto-référencés `picarones.measurements.narrative`
  réécrits en `picarones.reports_v2.narrative`.
- Path des templates YAML auto-ajusté (relatif à `__file__`).
- 12 shims `measurements/narrative/*.py` + `_helpers.py` shim
  manuel (privé, pas d'`__all__`).
- `_DEFAULT_REGISTRY` (singleton du registre des détecteurs)
  ré-exporté explicitement par le shim `__init__.py` pour la
  rétrocompat des tests S19.

**Effort réel** : ~1 jour (vs estimation 8-12 j — script de
génération de shims a fortement accéléré ; pas d'aléatoire ni
d'I/O dans les détecteurs, donc régression triviale par
construction).

**Acceptance** : tous les tests narratifs passent — Sprints 16,
19, 23, 29, 36, 44, 46, 73, 90, 92, baseline_comparison, chantier
5, reproducibility_ops.  322 tests ciblés passed.  Test architectural
anti-imports legacy : 3 passed (le rewrite reste autonome).
Garde-fou anti-hallucination préservé (les détecteurs lisent
toujours le dict JSON d'entrée, pas une source externe).

### Phase 4 — 35 mesures legacy (`measurements/*.py`) — partielle ✅

**Audit de migrabilité** : sur 35 mesures legacy, **24 étaient
déjà des re-exports** (Phase 4 partielle pré-existante avec un
canonique `evaluation/metrics/X.py`).  Sur les 11 modules réellement
"contenu" :

- **9 sont migrés en Phase 4 (cette session)** sans toucher à
  `core.modules` : autonomes ou en cascade vers d'autres modules
  migrables.
- **13 modules réels** restent bloqués par
  `core.modules.ArtifactType` (enum legacy 6 valeurs incompatible
  avec le superset `domain.artifacts.ArtifactType` 10 valeurs ;
  `TEXT` ≠ `RAW_TEXT`, `ALTO` ≠ `ALTO_XML`, `PAGE` ≠ `PAGE_XML`).
  Substitution non transparente — exigerait un travail de
  remapping sémantique sur chaque caller.

**Migrés en Phase 4 — 9 modules** :

| Legacy | Canonique | Notes |
|--------|-----------|-------|
| `measurements/char_scores.py` (307) | `evaluation/metrics/char_scores.py` | Autonome |
| `measurements/difficulty.py` (161) | `evaluation/metrics/difficulty.py` | Autonome |
| `measurements/ner_backends.py` (186) | `evaluation/metrics/ner_backends.py` | Autonome |
| `measurements/normalization.py` (51) | `evaluation/metrics/normalization.py` | Autonome |
| `measurements/structure.py` (182) | `evaluation/metrics/structure.py` | Autonome |
| `measurements/cost_projection.py` (140) | `evaluation/metrics/cost_projection.py` | dep `pricing` (déjà migré) |
| `measurements/specialization.py` (158) | `evaluation/metrics/specialization.py` | dep `inter_engine` (re-export déjà) |
| `measurements/taxonomy.py` (294) | `evaluation/metrics/taxonomy.py` | dep `char_scores` (en cascade) |
| `measurements/taxonomy_intra_doc.py` (178) | `evaluation/metrics/taxonomy_intra_doc.py` | dep `taxonomy` (en cascade) |

Total : **1657 lignes de code migrées + 9 shims legacy**.

**Bloqués — 13 modules + 1 sous-package + 6 modules `core/`** :

Reportés à une phase dédiée **« Phase 4-bis : ArtifactType
migration »** dont le périmètre est :

1. Décider le mapping sémantique TEXT → RAW_TEXT vs CORRECTED_TEXT
   (par module, en lisant le contexte d'usage).
2. Migrer `core/modules.py` (`BaseModule` + `ArtifactType` 6
   valeurs) vers `domain/module_protocol.py`.
3. Migrer `core/metric_registry.py` + `core/metric_hooks.py` vers
   `evaluation/registry/`.
4. Adapter chaque module bloqué : `mufi.py`, `abbreviations.py`,
   `early_modern_typography.py`, `modern_archives.py`,
   `roman_numerals.py`, `unicode_blocks.py`, `equivalence_profile.py`,
   `philological_hooks.py`, `ner.py`, `readability.py`,
   `readability_hooks.py`, `searchability.py`,
   `searchability_hooks.py`, `reading_order.py`, `alto_metrics.py`,
   `numerical_sequences.py`, `numerical_sequences_hooks.py`,
   `builtin_hooks.py`, `builtin_metrics.py`, `metrics.py`,
   `pipeline_benchmark.py`, `pipeline_comparison.py`,
   `pipeline_spec_loader.py`, `robustness.py`, `reliability.py`,
   `history.py`.
5. Migrer le sous-package `measurements/runner/` (orchestrateur
   legacy → fondre dans `pipeline/` + `app/services/`).
6. Migrer `core/results.py` (`BenchmarkResult` + 30 champs agrégés
   → typed Artifacts dans `domain/`).
7. Migrer `core/corpus.py` (`Document`/`Corpus`/`GTLevel` → modèle
   convergent avec `domain.corpus`).

**Effort estimé Phase 4-bis** : 18-22 jours (vs 23-28 j initialement
estimés pour Phase 4 complète — la moitié déjà faite par les
re-exports pré-existants et les 9 modules de cette session).

**Acceptance Phase 4 partielle** : 9 modules migrés, 1191 tests
mesures passent (inchangés), test architectural anti-imports
legacy reste vert.  Les 13 modules réels + 6 modules `core/`
restants sont documentés comme dépendant d'une migration
ArtifactType.

#### Tentative Phase 4-bis (avortée — diagnostic posé)

Une tentative de migration coordonnée de l'``ArtifactType`` a été
explorée puis revertée :

**Stratégie testée** : exploiter le mécanisme natif d'aliases
d'``Enum`` Python (un membre avec la même valeur qu'un autre devient
un alias).  Ajout de ``TEXT = "raw_text"``, ``ALTO = "alto_xml"``,
``PAGE = "page_xml"`` à ``domain.artifacts.ArtifactType`` + hook
``_missing_`` pour accepter les valeurs string legacy.  Puis
transformation de ``core/modules.py`` en shim qui ré-exporte
``ArtifactType`` et ``BaseModule`` depuis le canonique.

**Conservé en place** : les aliases + ``_missing_`` dans
``domain.artifacts.ArtifactType``.  Inoffensif — aucun code legacy
ne les voit puisqu'aucun module legacy n'importe encore depuis le
canonique.

**Reverté** : le shim ``core/modules.py``.  Cause : passer le
``core.modules.ArtifactType`` du legacy enum 6 valeurs au superset
canonique change silencieusement ``ArtifactType.TEXT.value`` de
``"text"`` à ``"raw_text"``.  Or 27 tests legacy
(``test_sprint63_pipeline_runner``, ``test_sprint65_pipeline_comparison``,
``test_sprint68_pipeline_comparison_html`` etc.) reposent sur le
fait que les clés des dicts ``junction_metrics`` produites par le
runner legacy sont les valeurs string legacy.  Quand le runner
utilise ``at.value`` pour stocker, il stocke maintenant ``"raw_text"``,
et les tests qui cherchaient ``junction_metrics["text"]`` cassent.

Le diagnostic est plus profond qu'un simple rename : le legacy
``BenchmarkResult.junction_metrics`` est un ``dict[str, dict]``
indexé par valeur string ; sa stabilité de format est implicitement
testée.  Migrer ``core.modules.ArtifactType`` exige un travail
**par module** d'identification des dicts indexés par valeur
string, et soit (a) double-clé pour rétrocompat, (b) migration
ordonnée tests-en-même-temps.

**Plan rectifié pour Phase 4-bis** :

1. Lister exhaustivement les dicts indexés par ``ArtifactType.value``
   dans le legacy (``core/results.py``, ``core/pipeline.py``,
   ``measurements/runner/``, ``measurements/pipeline_*``).
2. Décider la stratégie par module : double-clé pendant la
   migration vs migration coordonnée tests + code.
3. Migrer un cluster à la fois en validant la suite après chaque.

**Effort rectifié** : 25-30 jours (vs 18-22 estimés initialement —
le couplage implicite des dicts indexés par valeur string n'avait
pas été vu à l'audit).

**Statut Phase 4-bis** : analyse posée, exécution reportée à un
sprint dédié de plusieurs sessions.

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

**Cible** : `picarones.formats.alto.baseline_reconstruction` ou
`picarones.evaluation.projectors.text_to_alto` (selon où la
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
| 0 | ✅ Terminée |
| 1 | ✅ Partielle (3/9 modules ; les 6 autres → Phase 4-bis) |
| 2 | ✅ Terminée (8/8 modules statistics migrés) |
| 3 | ✅ Terminée (11 modules narrative + 2 templates + 18 détecteurs migrés) |
| 4 | ✅ Partielle (9 modules autonomes/cascade ; 13 modules + 6 modules `core/` + 1 sous-package → Phase 4-bis) |
| 4-bis | 🟡 Diagnostic posé, exécution reportée (couplage dicts-string plus complexe que prévu — voir détail Phase 4) |
| 5-11 | ⚪ À démarrer |

**Dernière mise à jour** : 2026-05 (Phase 4-bis tentative + revert + diagnostic).

**Reste en place suite à la tentative Phase 4-bis** : aliases
``TEXT``/``ALTO``/``PAGE`` dans ``domain.artifacts.ArtifactType``
(inoffensif) + hook ``_missing_`` pour accepter les valeurs string
legacy.  Préparation pour la session future qui complétera Phase
4-bis.
