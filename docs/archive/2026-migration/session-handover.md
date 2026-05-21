# Handover entre sessions Claude Code

> Ce document est lu en premier par chaque nouvelle session pour
> reprendre le travail sans se tromper.  Il pointe vers les
> sources de vérité, signale les pièges connus, et donne la
> prochaine action concrète.

---

## 0. Principe directeur (mis à jour 2026-05)

**Suppression agressive, pas de shim qui survit à son usage.**

- Le projet est en stand-by jusqu'à la fin de la migration
  complète.  Personne (ni externe ni HuggingFace Space) ne
  consommera l'API legacy avant cette fin.
- Pas de préservation de l'API publique : breaking changes
  acceptés.
- Dès qu'un caller migre vers le canonique, son shim est
  **supprimé** (pas conservé pour un usage hypothétique).
- L'invariant inverse est gardé par
  ``tests/architecture/test_no_legacy_imports_in_rewrite.py`` :
  les paquets rewrite (`domain → formats → evaluation →
  pipeline → adapters → app → reports → interfaces`) ne
  doivent jamais importer depuis un paquet legacy.  Pour le
  retrait final v2.0, ``LEGACY_PACKAGES = ()`` (vide) — tous
  les paquets top-level legacy ont été supprimés.

---

## 1. Sources de vérité (par ordre de priorité)

1. **[`legacy-retirement-plan.md`](legacy-retirement-plan.md)** —
   plan maître des Phases 0-11 du retrait du legacy.  Chaque
   phase a un statut explicite (✅ terminée / ⏳ en cours / 📋 à
   venir).
2. **[`pipeline-convergence-plan.md`](pipeline-convergence-plan.md)** —
   sous-plan détaillé de la convergence ``BaseModule`` /
   ``PipelineRunner`` → ``StepExecutor`` / ``PipelineExecutor``
   (Sub-phases 7.A-7.D).
3. **[`../../tests/architecture/test_no_legacy_imports_in_rewrite.py`](../../tests/architecture/test_no_legacy_imports_in_rewrite.py)** —
   garde-fou architectural : aucun module rewrite n'importe
   depuis un paquet legacy (``LEGACY_PACKAGES`` désormais
   vide à v2.0).
4. **[`../../CLAUDE.md`](../../CLAUDE.md)** — règles d'architecture
   à respecter, statut de la migration, et liens vers le reste.
5. **`git log --oneline -10`** — les 10 derniers commits
   donnent l'état réel.  Le dernier commit message décrit
   souvent la prochaine sub-phase à exécuter.

---

## 2. Vérifications avant de toucher au code

```bash
# 1. Bonne branche ?
git branch --show-current
# → doit retourner: claude/repo-analysis-cukvm

# 2. Working tree propre ?
git status
# → doit retourner: nothing to commit, working tree clean

# 3. Tests verts à l'état initial ?
python -m pytest tests/ -q --no-header --tb=line
# → doit retourner: 5085 passed (au moment de la pause de session)

# 4. Lint vert ?
ruff check picarones/ tests/
# → doit retourner: All checks passed!
```

Si l'une de ces vérifications échoue : **NE PAS** continuer le
sprint.  Investiguer d'abord pourquoi l'état initial diverge de
celui annoncé dans CLAUDE.md.

---

## 3. Pièges connus (apprentissages des phases précédentes)

### 3.A Architecture des couches

Voir CLAUDE.md section « Règles d'architecture critiques ».
Résumé :

- ``evaluation/`` ne peut pas importer ``pipeline.types`` —
  c'est l'autre sens.
- ``evaluation/`` whitelist limitée : pas de pytesseract /
  mistralai / azure / google / pero_ocr.  Ces libs externes
  vont dans ``adapters/``.
- ``reports/`` ne peut importer que les canoniques
  (``evaluation/metrics/``), pas les shims legacy
  (``measurements/X.py``).

### 3.B Pattern shim — UNIQUEMENT TRANSITOIRE

⚠️ **Principe** : un shim n'existe que pour la durée d'un
sprint.  Dès que tous ses consommateurs ont migré, il est
**supprimé**.

Pour un shim minimal (transitoire) :

```python
"""``picarones.X.Y`` — shim re-export (déprécié, suppression imminente).

Canonique : :mod:`picarones.canonical.path`.  Phase X.Y du
retrait du legacy.  Ce shim disparaît dès que tous les callers
auront migré (généralement dans le commit suivant).
"""

from __future__ import annotations

import warnings

from picarones.canonical.path import *  # noqa: F401, F403
# Si des callers consomment des noms privés (_FOO, etc.),
# les ré-exporter explicitement :
from picarones.canonical.path import _FOO  # noqa: F401

warnings.warn(
    "picarones.X.Y is deprecated and will be removed in 2.0.  "
    "Import from picarones.canonical.path instead.",
    DeprecationWarning,
    stacklevel=2,
)
```

**Avant de créer un shim**, demandez-vous : « est-ce que je peux
juste migrer tous les callers maintenant et supprimer le legacy
en bloc ? »  Si oui, faites-le — pas de shim intermédiaire.

### 3.C ``test_module_coverage::TEST_ONLY_BASELINE``

Quand un shim ``measurements/X.py`` n'a plus de consommateur
production (parce qu'un renderer a migré vers le canonique
direct), ajouter ``"X"`` à ``TEST_ONLY_BASELINE`` dans
``tests/architecture/test_module_coverage.py``.  Sinon le test
``test_no_new_test_only_modules`` échoue.

### 3.D ``test_file_budgets``

Tout fichier ≥ 400 LOC doit avoir une entrée dans
``FILE_BUDGETS`` avec budget = LOC actuel + ~15 %.  Quand on
relocalise un fichier, retirer l'entrée du chemin legacy et
en créer une au chemin canonique avec le même budget.

### 3.E ``test_doc_paths::BROKEN_PATHS_BASELINE``

Si un sub-plan ou doc référence un futur chemin Python
(``picarones/X/Y.py``) qui n'existe pas encore, le test
``test_broken_doc_paths_below_baseline`` détecte la
référence cassée.  Soit :

- Bumper ``BROKEN_PATHS_BASELINE`` du même montant.
- Ou reformuler la référence en code/backticks pour échapper
  au pattern (``picarones/X/Y.py``).

Quand le fichier sera créé en réalité, abaisser
``BROKEN_PATHS_BASELINE``.

### 3.F Test parité legacy ↔ canonique (retiré au sprint H.5)

``tests/architecture/test_legacy_canonical_parity.py`` a été
supprimé au sprint H.5 (mai 2026) : la table 3-états était
vidée au fil des Lots A-G (chaque entrée retirée en même temps
que le shim concerné), et tous les paquets legacy top-level
qu'elle scannait (``llm/``, ``measurements/``, ``engines/``,
``modules/``, ``report/``, ``core/``, ``cli/``, ``web/``,
``extras/``, ``pipelines/``) ont été supprimés ou relocalisés
sous ``adapters/legacy_*`` et ``interfaces/*/_legacy/``.  Le
seul invariant qui reste à garder est l'absence d'import legacy
depuis le rewrite (``test_no_legacy_imports_in_rewrite``).

### 3.G README généré

Le compteur de tests dans `README.md` et `CLAUDE.md` est
synchronisé par `scripts/gen_readme_tables.py`.  À chaque
fois que le nombre de tests change (ajout/retrait), lancer :

```bash
python scripts/gen_readme_tables.py
```

Sinon le test ``test_readme_tables_consistent_with_code``
échoue.

---

## 4. Inventaire actuel — quel legacy reste à migrer ?

(Snapshot au moment de la pause de session, mesuré via AST,
fiable.)

### 4.A Imports legacy dans les tests

**62 fichiers** avec **361 statements** d'import depuis les
paquets legacy (``measurements``, ``llm``, ``pipelines``) —
Lots A à G terminés (cf. 4.D ci-dessous).  Les paquets
``engines/``, ``modules/``, ``report/`` et ``core/`` ont été
**entièrement supprimés**.  Restent uniquement
``measurements/`` (~25 modules de catégorie B/C/D),
``llm/``, ``pipelines/`` et les sous-paquets d'interfaces
(``cli/``, ``web/``, ``extras/``).

Top chemins consommés :

| Imports | Chemin legacy                                                 |
|---------|---------------------------------------------------------------|
| 29      | ``from picarones.measurements.runner import run_benchmark``   |
| 18      | ``from picarones.measurements.metrics import MetricsResult``  |
| 16      | ``from picarones.measurements.statistics import wilcoxon_test`` |
| 13      | ``from picarones.measurements.metrics import compute_metrics`` |
| 10      | ``from picarones.measurements.robustness import degrade_image_bytes`` |

**Pourquoi c'est important** : ces tests passent par les shims
au lieu de pointer vers le canonique.  Tant que ces imports
existent, on **ne peut pas supprimer les shims** (le test casse).

**Stratégie** : sed batch par chemin, valider les tests,
commit, avancer.  Shims supprimés dans les Lots A
(``core.modules`` + ``core.facts``), B
(``core.metric_registry`` + ``core.metric_hooks`` +
``core.metrics``), C (``core.results`` + ``core.corpus`` +
``core.pipeline``) et D (34 shims plats de ``measurements/``
vers ``evaluation.metrics/``) sur la branche
``claude/migrate-core-to-domain-8ubIT``.

### 4.B Imports legacy en production (hors shims eux-mêmes)

**12 fichiers** avec **41 statements** dans des paquets
non-legacy qui pointent encore vers le legacy.  À résoudre
sprint par sprint en migrant chaque caller.

### 4.C Symboles legacy non tracés dans la table de parité

**Sans objet à v2.0** : la table de parité a été retirée au
sprint H.5, en même temps que la suppression des derniers
paquets legacy top-level (cf. 3.F).

### 4.D Plan de bataille pour les imports tests

L'ordre recommandé, par lots de symboles cohérents :

1. ✅ **Lot A — domain** (~40 imports migrés, shims supprimés) :
   - ``core.modules.{ArtifactType, BaseModule, ExecutionMode}``
     → ``domain.{artifacts, module_protocol}``
   - ``core.facts.*`` → ``domain.facts.*``
   - Shims ``picarones.core.modules`` + ``picarones.core.facts``
     supprimés ; doc utilisateur (tutorials/, developer/,
     reference/api-stable.md, explanation/narrative-engine.en.md)
     pointe maintenant vers les canoniques.
2. ✅ **Lot B — evaluation/metric_*** (~45 imports migrés, shims
   supprimés) :
   - ``core.metric_registry.*`` → ``evaluation.metric_registry.*``
   - ``core.metric_hooks.*`` → ``evaluation.metric_hooks.*``
   - ``core.metrics.*`` → ``evaluation.metric_result.*``
   - Shims ``picarones.core.metric_registry`` +
     ``picarones.core.metric_hooks`` + ``picarones.core.metrics``
     supprimés ; ``docs/reference/normalization-profiles.md`` et
     ``docs/reference/api-stable.md`` migrés vers les chemins
     canoniques.
3. ✅ **Lot C — evaluation/{benchmark_result, corpus, pipeline}**
   (~75 imports migrés, shims supprimés) :
   - ``core.results.*`` → ``evaluation.benchmark_result.*``
   - ``core.corpus.*`` → ``evaluation.corpus.*``
   - ``core.pipeline.*`` → ``evaluation.pipeline.*``
   - Shims ``picarones.core.{results, corpus, pipeline}``
     supprimés ; sections de ``docs/reference/api-stable.md``
     migrées vers les chemins canoniques ; logger filter dans
     ``test_sprint32_multi_level_gt`` aligné sur
     ``picarones.evaluation.corpus``.
4. ✅ **Lot D — evaluation/metrics/*** (~100 imports + 44
   prod migrés, 34 shims supprimés en bloc) :
   - ``measurements.{baseline_comparison, calibration,
     char_scores, confusion, cost_projection, difficulty,
     error_absorption, hallucination, image_predictive,
     image_quality, incremental_comparison, inter_engine,
     layout, levers, lexical_modernization, line_metrics,
     longitudinal, marginal_cost, module_policy, ner_backends,
     normalization, numerical_sequences, pricing, rare_tokens,
     robustness_projection, roman_numerals, specialization,
     structure, taxonomy, taxonomy_comparison,
     taxonomy_cooccurrence, taxonomy_intra_doc, throughput,
     worst_lines}`` → ``evaluation.metrics.{...}``.
   - L'ancien ``measurements/__init__.py`` réécrit pour
     refléter la nouvelle composition (modules legacy
     restants + `import picarones.evaluation.metrics`
     unique pour déclencher les décorateurs).  Tout le
     sous-package a été supprimé au sprint E.6.
   - ``test_no_flat_files_in_measurements::WHITELIST_FLAT_FILES_S3``
     réduit de 60 → 25 entrées.
   - ``test_module_coverage::TEST_ONLY_BASELINE`` réduit
     de 16 → 4 entrées.
   - ``test_file_budgets::FILE_BUDGETS`` débarrassé des
     entrées orphelines (inter_engine, levers,
     normalization).
5. ✅ **Lot E — adapters/legacy_*** (8 shims supprimés en bloc,
   0 import à migrer) :
   - ``engines.*`` → ``adapters.legacy_engines.*``
   - ``modules.alto_text_to_mono_region`` →
     ``adapters.legacy_modules.alto_text_to_mono_region``
   - Tous les callers tests + production avaient déjà été
     migrés en amont (Lots A-D), donc le Lot E n'a fait que
     supprimer les 8 shims orphelins.
   - ``LEGACY_PACKAGES`` réduit (retrait d'``engines`` et
     ``modules``) dans
     ``test_no_legacy_imports_in_rewrite.py`` et
     ``test_legacy_canonical_parity.py``.
   - ``ENGINES_DIR`` dans
     ``tests/docs/test_readme_consistency.py`` redirigé vers
     ``picarones/adapters/legacy_engines/``.
6. ✅ **Lot F — reports_v2** (37 shims supprimés en bloc, 7
   imports tests à migrer + ``scripts/gen_readme_tables.py``
   redirigé) :
   - ``report.*_render`` → ``reports_v2.html.renderers.*`` (29 shims)
   - ``report.{generator, comparison, snapshot}`` →
     ``reports_v2.html.*`` (3 shims)
   - ``report.{assets, colors, render_helpers}`` →
     ``reports_v2._helpers.*`` (3 shims)
   - ``report.diff_utils`` → ``evaluation._diff_utils`` (1 shim)
   - ``report.glossary`` → ``reports_v2.glossary`` (sous-package)
   - ``scripts/gen_readme_tables.py`` redirigé vers
     ``picarones/adapters/legacy_engines/`` ;
     ``docs/reference/views.md`` migré en place vers
     ``picarones/reports/html/{views, generator, renderers,
     templates}``.
7. ⏳ **Lot G — measurements/runner et co.** (reporté car
   canonique absent — phase 6 du plan maître).
   Réalisé partiellement : suppression des 2 derniers shims
   de ``picarones/core/`` (``diff_utils``, ``xml_utils``).
   Le sous-paquet ``core/`` n'existe plus du tout.

   La part majeure du Lot G originel (``measurements/runner``
   + ``pipelines/``) reste à faire ; elle nécessite **d'abord
   la création** des canoniques ``app/services/run_orchestrator``
   et ``adapters/llm/pipeline`` (couvrant ``OCRLLMPipeline``,
   ``PipelineMode``, ``over_normalization``, ``run_benchmark``,
   ``_compute_document_result``).  Sans ces canoniques, un
   simple sed est impossible — il faudrait migrer les 76
   imports vers des modules qui n'existent pas encore.

8. ✅ **Lot H — measurements.statistics → evaluation.statistics**
   (~70 imports migrés, 9 shims supprimés en bloc) :
   - ``measurements.statistics.{bootstrap, cdd_render,
     clustering, correlation, distributions, friedman_nemenyi,
     pareto, wilcoxon}`` → ``evaluation.statistics.{...}``.
   - ``measurements/statistics/`` (sous-paquet entier)
     supprimé.

9. ✅ **Lot I — extras.importers → adapters.corpus**
   (3 shims supprimés, ~15 imports migrés) :
   - ``extras.importers.htr_united`` →
     ``adapters.corpus.htr_united``.
   - ``extras.importers.huggingface`` →
     ``adapters.corpus.huggingface``.
   - ``extras.importers._fallback_log`` →
     ``adapters.corpus._fallback_log``.

10. ✅ **Lot J — measurements.metrics.{MetricsResult,
   aggregate_metrics} → evaluation.metric_result** (~25
   imports migrés, 0 shim supprimé) :
   - Migration partielle uniquement des symboles canoniquement
     migrés (``MetricsResult``, ``aggregate_metrics``).
   - ``compute_metrics`` reste dans
     ``picarones.measurements.metrics`` car aucun canonique
     n'existe pour cette fonction (sera traité avec le Lot G
     reporté).

À chaque lot : sed → tests → commit.  Les shims devenus
orphelins après le lot peuvent être **supprimés** dans le même
commit (principe « no shim survives its caller »).

---

## 5. Prochaines sub-phases à exécuter (post H.5)

Les sprints A-G + H.1-H.3 + H.5 ont été achevés.  Restent
les chantiers suivants pour atteindre v2.0 :

### H.2.b-d — refonte API ``BaseOCREngine`` → ``BaseOCRAdapter``

- ``adapters/legacy_engines/`` (Tesseract, Pero, Mistral OCR,
  Google Vision, Azure DI) doit être promu en
  ``adapters/ocr/`` aux contrats ``StepExecutor``
  (``execute(inputs, params, context)`` au lieu de
  ``run(image_path)``).
- Suppression de ``OCRLLMPipeline`` + ``adapters/legacy_pipelines/``
  une fois les callers migrés vers la construction directe
  d'une ``PipelineSpec`` via
  ``picarones.pipeline.make_ocr_llm_pipeline_spec``.
- Suppression de ``BaseOCREngine``, ``BaseModule``,
  ``adapters/legacy_engines/``, ``adapters/legacy_pipelines/``.

### H.4 — refonte interfaces

- ``interfaces/cli/_legacy/`` : refondre les commandes Click
  pour consommer directement ``BenchmarkService`` /
  ``RunOrchestrator`` au lieu du runner legacy.
- ``interfaces/web/_legacy/`` : refondre les routes FastAPI
  pour consommer le rewrite pur (sans ``OCRLLMPipeline``).

### H.6 — release v2.0

- Bump version dans ``pyproject.toml`` + ``picarones/_version.py``.
- Section CHANGELOG « 2.0.0 — Legacy retirement complete ».
- Tag ``v2.0.0``.

---

## 6. Commande de démarrage de la nouvelle session

Le user envoie simplement :

```
Reprends la migration. Lis docs/migration/SESSION_HANDOVER.md
en entier d'abord, puis commence par les vérifications de la
section 2.
```

Ou pour aller direct à l'action :

```
Attaque H.2.b — refonte BaseOCREngine → BaseOCRAdapter.
```

(Claude Code va automatiquement lire CLAUDE.md à l'init, qui
pointera vers ce SESSION_HANDOVER.md et les plans détaillés.)
