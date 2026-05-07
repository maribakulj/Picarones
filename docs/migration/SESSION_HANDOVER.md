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
- Tout symbole legacy public doit être tracé dans
  ``tests/architecture/test_legacy_canonical_parity.py`` :
  `canonical: ...` (équivalent canonique existe), `dropped: ...`
  (volontairement abandonné, justifié), ou `unmigrated: ...`
  (cible prévue, en cours).

Le test ``test_legacy_canonical_parity`` garantit qu'**aucune
fonctionnalité legacy n'est silencieusement perdue** au cours
de la migration.  C'est le journal de bord vivant.

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
3. **[`../../tests/architecture/test_legacy_canonical_parity.py`](../../tests/architecture/test_legacy_canonical_parity.py)** —
   journal vivant de la migration : table 3-états des symboles
   legacy avec leur équivalent canonique.  À mettre à jour à
   chaque migration.
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
- ``reports_v2/`` ne peut importer que les canoniques
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

### 3.F Test parité legacy ↔ canonique

``tests/architecture/test_legacy_canonical_parity.py`` maintient
une table 3-états (``LEGACY_PARITY``) :

- ``canonical: <module.symbol>`` — équivalent canonique existe.
  Le test vérifie présence + signatures compatibles.
- ``dropped: <raison>`` — feature volontairement abandonnée
  avec justification écrite.
- ``unmigrated: <cible prévue>`` — migration prévue ; cible
  peut ne pas encore exister.

À chaque migration d'un symbole, **mettre à jour la table**.
Les symboles non trackés sont comptés via
``BOOTSTRAP_BASELINE`` (à diminuer à chaque session).

Limites du test : il ne vérifie que la **présence** et les
**signatures**, pas le comportement réel.  Les différences
sémantiques sont signalées via le champ ``behavior_diff``
optionnel.

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

**91 fichiers** avec **472 statements** d'import depuis les
paquets legacy (``core``, ``measurements``, ``engines``,
``llm``, ``pipelines``, ``report``, ``modules``) — Lots A, B et
C terminés (cf. 4.D ci-dessous).  Le sous-paquet ``core/`` ne
contient plus que ``diff_utils`` et ``xml_utils`` (à migrer en
Lot G ou plus tard).

Top chemins consommés :

| Imports | Chemin legacy                                                 |
|---------|---------------------------------------------------------------|
| 29      | ``from picarones.measurements.runner import run_benchmark``   |
| 18      | ``from picarones.measurements.metrics import MetricsResult``  |
| 16      | ``from picarones.measurements.statistics import wilcoxon_test`` |
| 13      | ``from picarones.measurements.metrics import compute_metrics`` |
| 10      | ``from picarones.measurements.normalization import get_builtin_profile`` |

**Pourquoi c'est important** : ces tests passent par les shims
au lieu de pointer vers le canonique.  Tant que ces imports
existent, on **ne peut pas supprimer les shims** (le test casse).

**Stratégie** : sed batch par chemin, valider les tests,
commit, avancer.  Shims supprimés dans les Lots A
(``core.modules`` + ``core.facts``), B
(``core.metric_registry`` + ``core.metric_hooks`` +
``core.metrics``) et C (``core.results`` + ``core.corpus`` +
``core.pipeline``) sur la branche
``claude/migrate-core-to-domain-8ubIT``.

### 4.B Imports legacy en production (hors shims eux-mêmes)

**12 fichiers** avec **41 statements** dans des paquets
non-legacy qui pointent encore vers le legacy.  À résoudre
sprint par sprint en migrant chaque caller.

### 4.C Symboles legacy non tracés dans la table de parité

**110 symboles** publics dans les paquets legacy ne sont pas
encore dans
``tests/architecture/test_legacy_canonical_parity.py::LEGACY_PARITY``.
Répartition :

- ``measurements/`` : 104
- ``pipelines/`` : 6

Le test ``test_no_untracked_legacy_symbol_above_baseline``
autorise temporairement 110 (``BOOTSTRAP_BASELINE = 110``).
À diminuer à chaque session.

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
4. **Lot D — evaluation/metrics/*** (~80 imports) :
   - ``measurements.{difficulty, taxonomy, calibration, …}`` →
     ``evaluation.metrics.{...}``
5. **Lot E — adapters/legacy_***  (~50 imports) :
   - ``engines.*`` → ``adapters.legacy_engines.*``
   - ``modules.alto_text_to_mono_region`` →
     ``adapters.legacy_modules.alto_text_to_mono_region``
6. **Lot F — reports_v2** (~80 imports) :
   - ``report.*_render`` → ``reports_v2.html.renderers.*``
   - ``report.{generator, comparison, snapshot}`` →
     ``reports_v2.html.*``
7. **Lot G — measurements/runner et co.** (le plus complexe,
   couplé à Phase 6 qui retire ``pipelines/``).

À chaque lot : sed → tests → commit.  Les shims devenus
orphelins après le lot peuvent être **supprimés** dans le même
commit (principe « no shim survives its caller »).

---

## 5. Prochaine sub-phase à exécuter

**Sub-phase 7.B.2** — refactoriser le corps de
``PipelineRunner.run`` dans
``picarones/evaluation/pipeline.py`` (lignes 384-590) pour
qu'il délègue au canonique ``PipelineExecutor`` via le
wrapper ``_BaseModuleAdapter`` créé en 7.B.1.

### Plan d'exécution

1. **Lire** ``picarones/evaluation/pipeline.py:PipelineRunner.run``
   en entier pour comprendre la logique actuelle (résolution
   d'inputs versionnés, exécution chronométrée, capture
   d'erreur, évaluation auto vs GT, conversion outputs).

2. **Lire** ``picarones/pipeline/_legacy_module_adapter.py``
   en entier pour comprendre les outils disponibles
   (``_BaseModuleAdapter``, ``_PayloadRegistry``,
   ``wrap_initial_inputs``).

3. **Écrire** un nouveau corps de ``PipelineRunner.run`` qui :
   - Crée un ``_PayloadRegistry`` par appel.
   - Wrappe les ``initial_inputs`` legacy via
     ``wrap_initial_inputs(...)``.
   - Convertit la ``PipelineSpec`` legacy en ``PipelineSpec``
     canonique (``picarones.domain.pipeline_spec.PipelineSpec``).
     Chaque ``PipelineStep.module: BaseModule`` devient un
     ``adapter_name: str``, et l'adapter est
     ``_BaseModuleAdapter(module, registry)``.
   - Construit un ``adapter_resolver`` qui retourne le
     wrapper de chaque module.
   - Construit un ``RunContext``.
   - Convertit le ``Document`` legacy en ``DocumentRef``.
   - Invoque ``PipelineExecutor.run(canonical_spec,
     document_ref, canonical_inputs, context)``.
   - Reconvertit le ``PipelineResult`` canonique en
     ``PipelineResult`` legacy.
   - Calcule ``junction_metrics`` en post-étape (parcourt
     les ``StepResult.produced_artifacts``, lit le payload
     du registre, appelle ``compute_at_junction`` contre la
     GT du document si ``GTLevel`` correspond).

4. **Tester** : tous les tests existants doivent toujours
   passer (les 7 fichiers axe B + ``test_sprint63_pipeline_runner``,
   etc.).  C'est l'invariant de la sub-phase 7.B.2.

5. **Lint** : ``ruff check picarones/ tests/``.

6. **Commit + push** avec message décrivant ce qui a été
   fait + pointer vers la sub-phase 7.B.3 comme prochaine
   étape.

### Alternative pragmatique

Si le refactor 7.B.2 est trop gros pour une session,
**commencer par le Lot A de la section 4.D** (migrer les ~30
imports tests qui consomment ``core.modules`` et
``core.facts`` vers leur canonique ``domain/``).  Cela vide
une portion de la table de parité et permet de **supprimer les
shims** ``core.modules.py`` et ``core.facts.py`` en bloc —
résultat tangible et bien aligné avec le principe
« suppression agressive ».

Pareil pour Lots B-F : chaque lot est indépendant, fait
progresser la migration, et démontre concrètement la
suppression du legacy.

### Pièges anticipés pour 7.B.2

- **Sémantique différente des inputs entre legacy et canonique** :
  le legacy passe ``Document.image_path`` comme un string
  pur dans ``initial_inputs[ArtifactType.IMAGE]`` ; le canonique
  attend un ``Artifact(uri=...)``.  ``wrap_initial_inputs``
  fait la conversion mais il faut s'assurer que les modules
  consomment bien le ``uri`` côté `_BaseModuleAdapter`.

- **``junction_metrics`` calcul** : le legacy
  ``PipelineRunner.run`` calcule ``junction_metrics`` à
  chaque step (cf. ligne 519-540 actuellement).  Le canonique
  ``PipelineExecutor`` ne le fait pas.  Il faut donc faire
  ce calcul **après** l'exécution canonique, en parcourant
  les artefacts produits et en lisant les payloads via le
  registre.

- **``output_types`` partial** : si un module produit un
  output type non déclaré, le legacy le tolère (on remplit
  ``StepResult.output_types`` avec ce qui est effectivement
  produit, pas ce qui est déclaré).  Le canonique
  ``PipelineExecutor`` rejette en ``error="missing_output: ..."``.
  Vérifier la sémantique attendue par les tests.

- **Spec conversion** : ``PipelineStep`` legacy a
  ``inputs_from: dict[ArtifactType, str]`` (mapping
  type→step_name).  ``PipelineStep`` canonique a
  ``inputs_from: tuple[InputBinding, ...]``.  Conversion
  attentive nécessaire.

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
Continue la sub-phase 7.B.2.
```

(Claude Code va automatiquement lire CLAUDE.md à l'init, qui
pointera vers ce SESSION_HANDOVER.md et les plans détaillés.)
