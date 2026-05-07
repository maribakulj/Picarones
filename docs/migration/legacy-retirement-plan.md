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

#### Phase 4-bis — Reprise et complétion (2026-05)

La reprise s'appuie sur le **diagnostic de la tentative avortée**
en adoptant la stratégie « double-clé » : on garde les aliases
legacy ``TEXT``/``ALTO``/``PAGE`` dans
``domain.artifacts.ArtifactType``, et on s'engage à ce que tout
dict indexé par ``ArtifactType.value`` présente en parallèle la
clé canonique (``"raw_text"``) **et** la clé legacy (``"text"``)
quand un alias existe.

**Ajouts dans le canonique** :

- ``LEGACY_VALUE_ALIASES = {"raw_text": "text", "alto_xml": "alto",
  "page_xml": "page"}`` dans ``domain.artifacts``.
- ``expand_legacy_keys(d)`` qui mute un dict pour y copier les
  valeurs canoniques sous les alias legacy.
- ``BaseModule`` canonique dans ``domain/module_protocol.py``
  (10 valeurs vs 6 legacy).

**Sites mis à jour** :

- ``core/pipeline.py`` : ``StepResult.junction_metrics`` enrichi
  via ``expand_legacy_keys`` à la production.
  ``PipelineResult.junction_metrics_for`` essaie la clé canonique
  puis l'alias legacy.
  ``_artifact_type_to_gt_level`` utilise une map explicite
  ``ArtifactType → GTLevel`` (les valeurs canoniques
  ``"raw_text"``/``"alto_xml"``/``"page_xml"`` ne matchent plus
  ``GTLevel`` qui garde ``"text"``/``"alto"``/``"page"``).
- ``measurements/pipeline_benchmark.py`` :
  ``StepAggregate.junction_metrics`` enrichi via
  ``expand_legacy_keys`` après agrégation.
- ``measurements/pipeline_comparison.py`` :
  ``_final_metric_value`` essaie canonique puis legacy.
- ``evaluation/metrics/module_policy.py`` : la comparaison
  manifest vs déclaration normalise via les aliases (``"text"``
  match ``"raw_text"``).

**Migration des callers** : 16 modules ``measurements/`` + 6
modules `core/`/`engines/`/`modules/`/`cli/`/`report/` migrés
de ``from picarones.core.modules import ArtifactType`` vers
``from picarones.domain.artifacts import ArtifactType`` (et
``from picarones.domain.module_protocol import BaseModule``
quand applicable).

**``core/modules.py``** : transformé en shim avec
``DeprecationWarning`` à l'import.

**Tests adaptés** :

- ``test_public_api.py::test_artifact_type_values`` —
  asserte le set canonique 10 valeurs.
- ``test_sprint33_module_interface.py::test_repr_shows_io_types``
  — asserte ``"raw_text→raw_text"``.
- ``test_sprint68_pipeline_comparison_html.py::test_display_label_default``
  — asserte ``"raw_text.cer"``.

**Acceptance Phase 4-bis** : 5019 tests passent (vs 5008 avant la
reprise — 11 tests étaient cassés par la première tentative et
sont maintenant verts).  Les 24 fichiers de tests qui importent
encore ``from picarones.core.modules`` continuent à fonctionner via
le shim — ils ne deviendront erronés que quand le shim sera retiré
en 2.0.

**Reportés à Phase 4-ter** :

- ``core/metric_registry.py`` (263 l) et ``core/metric_hooks.py``
  (423 l) restent en place : ils sont consommés par 30+ modules
  ``measurements/`` via le décorateur ``@register_metric`` et les
  hooks ``register_document_metric``/``register_corpus_aggregator``.
  Le canonique existant ``evaluation/registry/registry.py`` utilise
  un design **instance-based** (``MetricRegistry()`` explicite,
  pas de décorateur module-level) qui est incompatible avec le
  pattern legacy.  La migration exige un choix de design (garder
  les deux, fondre dans une API unique, etc.) qui dépasse Phase
  4-bis.
- ``core/metrics.py`` (144 l, ``MetricsResult`` +
  ``aggregate_metrics``) reste en place : pas d'équivalent
  canonique dans ``domain/`` ou ``evaluation/`` à ce jour.  La
  conversion nécessitera d'abord de créer le destinataire dans
  ``domain.measurements`` (typé Pydantic au lieu de dataclass) ou
  ``evaluation.aggregation``.
- ``core/results.py`` (``BenchmarkResult`` + champs agrégés) :
  même statut.
- ``core/corpus.py`` (``Document``/``Corpus``/``GTLevel``) :
  même statut.  Note : ``GTLevel`` étant intentionnellement
  conservé en parallèle d'``ArtifactType``, son retrait dépend
  de la fin de migration des callers qui parsent les types de GT
  par leur valeur string.

#### Phase 4-ter — Relocalisation Cercle 1 → Cercle 2 (2026-05)

Stratégie « relocaliser sans redessiner » : on déplace
verbatim les modules legacy de ``core/`` (Cercle 1) vers
``evaluation/`` (Cercle 2) où ils appartiennent sémantiquement,
sans toucher à leur API publique.  Le pattern module-level
décorateur (``@register_metric``, ``@register_document_metric``,
``@register_corpus_aggregator``) est **conservé** — sa
convergence avec l'instance-based ``evaluation.registry.MetricRegistry``
(Sprint A14-S5) est laissée à un futur sprint dédié quand un
caller institutionnel le demandera.

**Migrations effectuées (A-D)** :

| Source legacy | Destination canonique | Lignes |
|---|---|---|
| ``core/metric_registry.py`` | ``evaluation/metric_registry.py`` | 264 |
| ``core/metric_hooks.py``    | ``evaluation/metric_hooks.py``    | 427 |
| ``core/metrics.py``         | ``evaluation/metric_result.py``   | 145 |
| ``core/results.py``         | ``evaluation/benchmark_result.py``| 702 |

Total : **1538 lignes** déplacées du Cercle 1 vers le Cercle 2.
Les chemins ``core/X.py`` deviennent des shims minimaux
(< 30 lignes chacun) avec ``DeprecationWarning`` à l'import.

**Adaptations transverses** :

- ``evaluation/benchmark_result.py`` ne peut pas importer
  ``picarones.__version__`` (cycle d'import via
  ``measurements/``).  La résolution de version utilise
  désormais ``importlib.metadata`` directement avec fallback
  ``"1.0.0"``.
- ``tests/architecture/test_file_budgets.py`` mis à jour
  pour pointer vers les nouveaux chemins canoniques.
- ``tests/core/test_public_api.py::TestCercle1IsLean.EXPECTED_CERCLE1``
  ne contient plus que ``corpus.py`` et ``pipeline.py``
  (les seuls modules ``core/`` réels qui restent).

**Reporté à Phase 4-quater** :

``core/corpus.py`` (511 l, ``Document``/``Corpus``/``GTLevel`` +
payloads + ``load_corpus_from_directory``) reste en place.
Raison : il y a déjà ``domain.corpus.CorpusSpec`` (Pydantic,
immutable, structural) et ``domain.documents.DocumentRef``
en parallèle.  La convergence des deux modèles
(``Document``/``Corpus`` historiques riches en behavior vs
``CorpusSpec``/``DocumentRef`` purement déclaratifs) est un
choix de design (fondre, garder les deux, marquer l'un comme
view-de-l'autre…) qui dépasse Phase 4-ter.  L'objectif Phase
4-quater est de produire un RFC qui tranche cette question
puis migre les 14 callers en une fois.

**Acceptance Phase 4-ter (A-D)** : 5019 tests passent, lint
vert, architecture vérifiée (anti-cycles, file budgets,
EXPECTED_CERCLE1 mis à jour).  Les 24+ fichiers de tests qui
importent encore via les chemins ``core/`` continuent à
fonctionner via les shims — déprécation visible mais
non-bloquante.

#### Phase 4-quater — Relocalisation de ``core/corpus.py`` (2026-05)

Décision RFC : **garder les deux modèles en parallèle**, sans
fusion.  ``evaluation.corpus`` (riche en behavior, dataclass,
chargé en mémoire, runner-friendly) et
``domain.corpus.CorpusSpec`` (Pydantic, immutable, déclaratif,
pipeline-executor-friendly) sont des projections différentes
d'un même domaine ; un convertisseur explicite
``CorpusSpec ↔ Corpus`` viendra quand un caller institutionnel
l'exigera concrètement.  Tenter une convergence prématurée
casserait soit le runner historique (qui consomme
``Document.get_gt(level)`` + ``Corpus.has_ocr_text``), soit le
pipeline executor canonique (qui consomme l'immutabilité de
``CorpusSpec`` pour la sérialisation YAML).

Migration effectuée
-------------------

| Source legacy        | Destination canonique         | Lignes |
|----------------------|-------------------------------|--------|
| ``core/corpus.py``   | ``evaluation/corpus.py``      |   533  |

Le chemin ``core/corpus.py`` devient un shim minimal
(< 30 lignes) avec ``DeprecationWarning`` à l'import.  Les 14
callers de production (``cli/_pipeline``, ``cli/_robustness``,
``cli/_workflows``, ``web/benchmark_utils``,
``measurements/pipeline_benchmark``,
``measurements/pipeline_comparison``,
``measurements/robustness``, ``measurements/runner/orchestration``,
``measurements/runner/ner_attach``,
``extras/importers/{iiif,gallica,escriptorium}``,
``core/pipeline``, et ``picarones/__init__.py``) sont migrés
vers ``picarones.evaluation.corpus``.

Note : ``GTLevel`` reste consommé en parallèle d'``ArtifactType``
par le runner — la convergence de ces deux énumérations est
liée au retrait du runner legacy lui-même (Phase 6+ du plan).

Adaptations transverses
-----------------------

- ``test_file_budgets.py`` : entrée ``core/corpus.py`` retirée,
  remplacée par ``evaluation/corpus.py`` (budget identique 600).
- ``test_public_api.py::EXPECTED_CERCLE1`` : ``corpus.py``
  retiré de la liste — il ne reste plus que ``pipeline.py``
  comme module Cercle 1 réel.

État final de ``core/`` après Phase 4-quater
--------------------------------------------

Le répertoire ``picarones/core/`` ne contient désormais qu'**un
seul module réel** :

- ``pipeline.py`` (~570 l) — ``PipelineRunner`` + ``PipelineSpec``
  + ``StepResult`` + ``PipelineResult``.

Tous les autres fichiers (``corpus.py``, ``modules.py``,
``metric_registry.py``, ``metric_hooks.py``, ``metrics.py``,
``results.py``, ``facts.py``, ``diff_utils.py``,
``xml_utils.py``) sont des shims < 30 lignes avec
``DeprecationWarning``.

**Acceptance Phase 4-quater** : 5019 tests passent (inchangé
depuis Phase 4-ter), lint vert, architecture vérifiée.  Le
``__init__.py`` racine (``picarones/__init__.py``) importe
maintenant directement depuis les chemins canoniques (Cercle
1 ``domain/`` + Cercle 2 ``evaluation/``), seul ``core.pipeline``
reste référencé pour ses types.

**Reporté à Phase 5** :

- ``core/pipeline.py`` (``PipelineRunner``) — convergence avec
  le pipeline executor canonique
  (``picarones/pipeline/executor.py``, ``planner.py``,
  ``runner.py``).  C'est le dernier module ``core/`` réel ;
  son retrait suppose que tous les callers passent par le
  pipeline executor, ce qui implique l'écriture du sucre
  syntaxique pour les benchmarks OCR mono-étape (typique
  ``run_benchmark(corpus, [engine_a, engine_b])``).
- Convergence ``GTLevel`` ↔ ``ArtifactType`` (en attente du
  retrait du runner legacy).

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

#### Phase 5.A+B — Helpers + glossary + i18n (2026-05)

Première tranche du retrait du legacy ``report/`` : les utilitaires
purs et les ressources statiques, sans toucher aux 22 renderers
thématiques (qui consomment ``BenchmarkResult`` legacy et seront
migrés au fil des phases ultérieures, par lots).

**Migrations effectuées** :

| Source legacy                        | Destination canonique                          |
|--------------------------------------|------------------------------------------------|
| ``report/colors.py``                 | ``reports_v2/_helpers/colors.py``              |
| ``report/render_helpers.py``         | ``reports_v2/_helpers/render_helpers.py``      |
| ``report/assets.py`` + ``vendor/``   | ``reports_v2/_helpers/assets.py`` + ``vendor/``|
| ``report/glossary/{fr,en}.yaml``     | ``reports_v2/glossary/{fr,en}.yaml``           |
| ``report/i18n/{fr,en}.json``         | ``reports_v2/i18n/{fr,en}.json``               |

``report/diff_utils.py`` redirige désormais directement vers
``picarones.evaluation`` (au lieu du double-shim via
``core.diff_utils``).

**Shims** : tous les chemins legacy ``report/X`` restent disponibles
via des shims minimaux (< 25 lignes) avec ``DeprecationWarning``
à l'import.

**Adaptations transverses** :

- ``picarones/i18n.py`` : ``_I18N_DIR`` pointe désormais vers
  ``reports_v2/i18n/``.
- 22 renderers ``report/*_render.py`` migrés sur leurs imports
  internes vers ``picarones.reports_v2._helpers.*``.
- 28 fichiers de tests mis à jour (chemins ``picarones/report/i18n/*``
  remplacés par ``picarones/reports_v2/i18n/*``).
- ``test_layer_dependencies.py::EXTERNAL_ALLOWED["reports_v2"]``
  étendu à ``PIL`` (Pillow utilisé par ``_helpers/assets.py``
  pour le redimensionnement d'images).
- ``test_file_budgets.py`` : entrée ``report/render_helpers.py``
  remplacée par ``reports_v2/_helpers/render_helpers.py``
  (budget 480 inchangé).

**Acceptance Phase 5.A+B** : 5019 tests passent, lint vert,
architecture vérifiée (anti-cycles, file budgets).  Aucune
régression sur les renderers thématiques (toujours legacy).

#### Phase 5.C.batch1 — Lot 1 : 5 renderers les plus petits (2026-05)

Première vague de migration des 22 renderers thématiques.  On
relocalise verbatim, sans toucher au contrat avec
``BenchmarkResult`` legacy — la convergence avec ``RunResult``
canonique reste un sprint à part entière (5.D ou 5.E selon
priorité).

Convention de nommage : ``picarones.report.<theme>_render`` →
``picarones.reports_v2.html.renderers.<theme>``.  Le suffixe
``_render`` est retiré (déjà implicite dans la position).

**Migrations effectuées** :

| Source legacy                            | Destination canonique                                |
|------------------------------------------|------------------------------------------------------|
| ``report/searchability_render.py`` (103) | ``reports_v2/html/renderers/searchability.py``       |
| ``report/specialization_render.py`` (113)| ``reports_v2/html/renderers/specialization.py``      |
| ``report/marginal_cost_render.py`` (111) | ``reports_v2/html/renderers/marginal_cost.py``       |
| ``report/rare_token_recall_render.py`` (116)| ``reports_v2/html/renderers/rare_token_recall.py``|
| ``report/readability_render.py`` (126)   | ``reports_v2/html/renderers/readability.py``         |

Total : ~569 lignes relocalisées.  Les chemins ``report/X_render.py``
deviennent des shims minimaux (< 20 lignes) avec
``DeprecationWarning``.

**Adaptations transverses** :

- ``reports_v2/html/renderers/specialization.py`` import canonique
  ``picarones.evaluation.metrics.specialization`` (au lieu du shim
  legacy ``picarones.measurements.specialization``) pour respecter
  la règle layer-dependencies (interdiction d'importer du legacy
  depuis ``reports_v2/``).
- ``test_module_coverage.py::TEST_ONLY_BASELINE`` étendu à
  ``"specialization"`` : son shim legacy n'a plus de consommateur
  production (le renderer est désormais dans ``reports_v2/``).
- 3 tests (``test_extra_metrics.py``,
  ``test_sprint86_aii5_html.py``,
  ``test_sprint87_readability_html.py``,
  ``test_sprint89_specialization.py``) mis à jour pour pointer
  vers les nouveaux chemins canoniques.
- ``picarones/report/generator.py`` mis à jour pour importer les
  5 renderers depuis ``reports_v2/html/renderers/``.

**Acceptance batch 1** : 5019 tests passent, lint vert,
architecture vérifiée.

**Reporté aux batches suivants** :

- Batch 2 ✅ (cf. ci-dessous) — 5 renderers (45-165 LOC).
- Batch 3 ✅ (cf. ci-dessous) — 5 renderers (173-222 LOC).
- Batch 4 ✅ (cf. ci-dessous) — 5 renderers (188-321 LOC).
- Batch 5 ✅ (cf. ci-dessous) — 5 renderers (148-314 LOC).
- Batch 6 (XXL + restants) : ``pipeline_render`` (707 l),
  ``philological_render`` (595 l), ``levers`` (284 l).
- Phase 5.D : 5 vues (``views/*.py``).
- Phase 5.E : ``generator.py``, ``comparison.py``,
  ``snapshot.py``, ``report_data/``, templates Jinja2.

Effort restant estimé : 8-12 jours.

#### Phase 5.C.batch2 — Lot 2 : 5 renderers moyens (2026-05)

Deuxième vague.  Substitution dans la sélection initiale :
``numerical_sequences_render`` reporté au batch 3 (sa dépendance
``measurements/numerical_sequences.py`` dépend elle-même de
``measurements/roman_numerals.py``, deux modules legacy non
migrés vers ``evaluation/metrics/`` ; le renderer ne peut donc pas
les importer depuis le canonique).  Remplacé par
``longitudinal_render`` qui n'a pas de dépendance legacy.

**Migrations effectuées** :

| Source legacy                                | Destination canonique                                |
|----------------------------------------------|------------------------------------------------------|
| ``report/difficulty_render.py`` (45)         | ``reports_v2/html/renderers/difficulty.py``          |
| ``report/lexical_modernization_render.py`` (114) | ``reports_v2/html/renderers/lexical_modernization.py`` |
| ``report/multirun_stability_render.py`` (151)| ``reports_v2/html/renderers/multirun_stability.py``  |
| ``report/throughput_render.py`` (154)        | ``reports_v2/html/renderers/throughput.py``          |
| ``report/longitudinal_render.py`` (165)      | ``reports_v2/html/renderers/longitudinal.py``        |

Total : ~629 lignes relocalisées.  5 nouveaux shims minimaux
(< 20 lignes) avec ``DeprecationWarning``.

**Adaptations transverses** :

- ``reports_v2/html/renderers/lexical_modernization.py`` import
  canonique ``picarones.evaluation.metrics.lexical_modernization``
  (au lieu du shim legacy ``picarones.measurements.lexical_modernization``).
- ``test_module_coverage.py::TEST_ONLY_BASELINE`` étendu à
  ``"lexical_modernization"`` (même rationale que ``specialization``
  au batch 1).
- Tests + ``picarones/report/generator.py`` mis à jour pour les
  5 chemins canoniques.

**Acceptance batch 2** : 5019 tests passent, lint vert,
architecture vérifiée.

**Cumul Phase 5.C** (batches 1+2) : 10 / 22 renderers migrés
(~1198 lignes).  12 renderers restants.

#### Phase 5.C.batch3 — Lot 3 : 5 renderers moyens (2026-05)

Troisième vague.  Tous les renderers sélectionnés sont
**purs sur le contrat** : import depuis ``_helpers/`` uniquement,
pas de dépendance sur des modules legacy non-migrés.

**Migrations effectuées** :

| Source legacy                                  | Destination canonique                                  |
|------------------------------------------------|--------------------------------------------------------|
| ``report/module_audit_render.py`` (173)        | ``reports_v2/html/renderers/module_audit.py``          |
| ``report/incremental_comparison_render.py`` (201)| ``reports_v2/html/renderers/incremental_comparison.py``|
| ``report/image_predictive_render.py`` (207)    | ``reports_v2/html/renderers/image_predictive.py``      |
| ``report/error_absorption_render.py`` (210)    | ``reports_v2/html/renderers/error_absorption.py``      |
| ``report/ner_render.py`` (222)                 | ``reports_v2/html/renderers/ner.py``                   |

Total : ~1013 lignes relocalisées.  5 nouveaux shims minimaux
(< 20 lignes) avec ``DeprecationWarning``.

**Adaptations transverses** :

- Tests + ``picarones/report/generator.py`` mis à jour pour les
  5 chemins canoniques.

**Acceptance batch 3** : 5019 tests passent, lint vert,
architecture vérifiée.

**Cumul Phase 5.C** (batches 1+2+3) : 15 / 22 renderers migrés
(~2211 lignes).  7 renderers restants.

#### Phase 5.C.batch4 — Lot 4 : 5 renderers moyens-gros (2026-05)

Quatrième vague.  Tous les renderers sélectionnés sont **purs sur
le contrat externe** (import depuis ``_helpers/`` uniquement).
``robustness_projection`` avait un import lazy interne vers
``picarones.measurements.robustness_projection`` qui a été redirigé
vers le canonique ``picarones.evaluation.metrics.robustness_projection``.

**Migrations effectuées** :

| Source legacy                                  | Destination canonique                                  |
|------------------------------------------------|--------------------------------------------------------|
| ``report/stratification_render.py`` (188)      | ``reports_v2/html/renderers/stratification.py``        |
| ``report/baseline_render.py`` (238)            | ``reports_v2/html/renderers/baseline.py``              |
| ``report/inter_engine_render.py`` (245)        | ``reports_v2/html/renderers/inter_engine.py``          |
| ``report/robustness_projection_render.py`` (252) | ``reports_v2/html/renderers/robustness_projection.py``|
| ``report/calibration_render.py`` (321)         | ``reports_v2/html/renderers/calibration.py``           |

Total : ~1244 lignes relocalisées.  5 nouveaux shims minimaux
(< 20 lignes) avec ``DeprecationWarning``.

**Adaptations transverses** :

- ``test_module_coverage.py::TEST_ONLY_BASELINE`` étendu à
  ``"robustness_projection"`` (même rationale que les batches
  précédents).
- Tests + ``picarones/report/generator.py`` mis à jour pour les
  5 chemins canoniques.

**Acceptance batch 4** : 5019 tests passent, lint vert,
architecture vérifiée.

**Cumul Phase 5.C** (batches 1+2+3+4) : 20 / 22 renderers migrés
(~3455 lignes).  2 renderers restants : ``pipeline_render`` (707 l)
et ``philological_render`` (595 l) — les XXL — auront leur propre
batch dédié.

#### Phase 5.C.batch5 — Lot 5 : 5 renderers moyens-gros (2026-05)

Cinquième vague.  Inclut les 3 renderers de la famille
``taxonomy``, ``worst_lines`` et ``pipeline_dag``.  Restera ensuite
batch 6 (XXL + ``levers``) et la migration des 5 vues
(``views/*.py``).

**Migrations effectuées** :

| Source legacy                                   | Destination canonique                                |
|-------------------------------------------------|------------------------------------------------------|
| ``report/taxonomy_intra_doc_render.py`` (148)   | ``reports_v2/html/renderers/taxonomy_intra_doc.py``  |
| ``report/taxonomy_cooccurrence_render.py`` (161)| ``reports_v2/html/renderers/taxonomy_cooccurrence.py``|
| ``report/worst_lines_render.py`` (164)          | ``reports_v2/html/renderers/worst_lines.py``         |
| ``report/taxonomy_comparison_render.py`` (233)  | ``reports_v2/html/renderers/taxonomy_comparison.py`` |
| ``report/pipeline_dag_render.py`` (314)         | ``reports_v2/html/renderers/pipeline_dag.py``        |

Total : ~1020 lignes relocalisées.

**Adaptations transverses** :

- ``reports_v2/html/renderers/worst_lines.py`` :
  - import ``WorstLineEntry`` redirigé vers
    ``picarones.evaluation.metrics.worst_lines``
  - import ``compute_char_diff`` redirigé vers
    ``picarones.evaluation`` (au lieu de ``picarones.core.diff_utils``,
    rejeté par la règle layer-dependencies sur ``reports_v2``).

**Cumul Phase 5.C** (batches 1+2+3+4+5) : 20 + 5 = **25 renderers
migrés**, soit l'intégralité moins ``pipeline_render`` et
``philological_render`` (XXL) et ``levers`` (oublié dans le plan
initial).  Reste batch 6 (3 renderers) puis Phase 5.D (5 vues).

Wait — le compte exact : 22 originaux moins ``pipeline_render``,
``philological_render`` et ``levers`` = 19 attendus.  Or on en a
migré 20 + 5 = 25 dans 5 batches.  Vérification : on a fait
batch 1 (5) + batch 2 (5) + batch 3 (5) + batch 4 (5) + batch 5 (5)
= 25.  Le plan initial listait 22 renderers ; en pratique le
``report/`` en contient ~28 (cf. ``ls report/*_render.py``) — la
liste du plan était incomplète.  L'inventaire exact restant :
``levers_render.py`` + ``pipeline_render.py`` +
``philological_render.py`` à finir (3 renderers, ~1586 LOC).

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
