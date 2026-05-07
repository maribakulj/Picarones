# Audit & sub-plan — Convergence ``PipelineRunner`` legacy ↔ ``PipelineExecutor`` canonique

> **Note** : ce document est l'audit demandé en conclusion de Phase 5
> du plan de retrait du legacy
> (cf. ``docs/migration/legacy-retirement-plan.md``).  Il identifie
> les différences entre les deux designs de pipeline, inventaire les
> callers, propose 3 stratégies de convergence et recommande un
> sub-plan d'exécution.

---

## 1. État des lieux

Deux designs cohabitent :

### 1.A Legacy — ``picarones.evaluation.pipeline`` (ex-``core/pipeline.py``)

Sprint 63 (axe B), 607 lignes.  Relocalisé en Phase 5.C.batch7
mais **non refactoré**.

**Caractéristiques** :

- ``PipelineRunner`` : classe statique avec ``.run(spec, document, initial_inputs) -> PipelineResult``.
- ``PipelineSpec`` : dataclass mutable.
- ``PipelineStep`` : dataclass avec ``module: BaseModule`` (instance Python).
- ``StepResult`` : dataclass avec ``junction_metrics: dict[str, dict[str, Any]]``.
- ``PipelineResult`` : dataclass avec ``steps: list[StepResult]``.
- Modules : ``BaseModule`` ABC consommant des **payloads bruts**
  (``{ArtifactType: str | dict | list | ...}``).
- Évaluation : ``compute_at_junction`` automatique à chaque étape
  contre la GT du document si ``GTLevel`` correspond.
- Pas de cache d'artefacts.
- Pas de ``ExecutionPlan`` séparé — résolution implicite des
  inputs au runtime via un bag versionné.

### 1.B Canonique — ``picarones.pipeline.executor`` + ``planner`` + ``protocols``

Sprints S6-S7-S28, design rewrite ciblé.

**Caractéristiques** :

- ``PipelineExecutor`` : classe instanciable avec
  ``adapter_resolver`` injecté + ``planner`` optionnel +
  ``artifact_store`` optionnel.
- Méthode ``run(spec, document, initial_inputs, context) ->
  PipelineResult`` (compat S7) qui plan-then-execute.
- Méthode canonique ``run_plan(plan, document, initial_inputs,
  context)`` qui consomme un ``ExecutionPlan`` pré-calculé.
- ``PipelineSpec`` : Pydantic immutable
  (``picarones.domain.pipeline_spec``), sérialisable YAML.
- ``PipelineStep`` : Pydantic immutable avec ``adapter_name: str``
  (pas d'instance — résolution applicative).
- ``ExecutionPlan`` : produit du ``PipelinePlanner`` — porte
  ``StepInputBinding`` explicites + ``MetricJunction`` détectées.
- ``StepResult`` : Pydantic immutable avec
  ``produced_artifacts: dict[str, str]`` (map ArtifactType.value →
  Artifact.id).
- ``PipelineResult`` : Pydantic immutable avec
  ``artifacts: tuple[Artifact, ...]``.
- Adapters : ``StepExecutor`` Protocol (runtime-checkable)
  consommant des **``Artifact`` typés**
  (``{ArtifactType: Artifact(uri, content_hash, provenance)}``).
- Cache d'artefacts via ``ArtifactCachePort`` (Sprint S29 + S47).
- ``RunContext`` Pydantic injecté à chaque ``execute()`` —
  document_id, code_version, pipeline_name, workspace_uri.

---

## 2. Différences API détaillées

| Dimension                  | Legacy (``evaluation.pipeline``)         | Canonique (``pipeline.executor``)             |
|----------------------------|------------------------------------------|-----------------------------------------------|
| Construction               | classe statique                          | instance avec deps injectées                  |
| Spec                       | dataclass mutable                        | Pydantic immutable, YAML-sérialisable         |
| Step                       | porte ``module: BaseModule``             | porte ``adapter_name: str``                   |
| Résolution adapters        | implicite (instance dans spec)           | explicite (``adapter_resolver`` callable)     |
| Résolution inputs          | implicite (last-producer-wins)           | explicite (``StepInputBinding``)              |
| Validation spec            | au runtime                               | au planning (``PipelinePlanner``)             |
| Type passé aux modules     | payload brut (str, dict, list…)          | ``Artifact`` typé                             |
| Provenance                 | absente                                  | ``ProvenanceRecord`` automatique              |
| Hash de contenu            | absent                                   | ``Artifact.content_hash`` SHA-256             |
| Cache inter-runs           | absent                                   | ``ArtifactCachePort``                         |
| ``RunContext``             | absent                                   | injecté à chaque step                         |
| Évaluation auto vs GT      | oui, à chaque step                       | non (sortie : artefacts seulement)            |
| ``junction_metrics``       | dans ``StepResult``                      | absent du runtime, calculé à part             |
| Représentation des étapes  | objets Python uniquement                 | YAML + Python                                 |

---

## 3. Inventaire des callers

### 3.A Legacy (``evaluation.pipeline``)

**Production** (4 fichiers) :

- ``picarones/__init__.py`` — re-export de ``PipelineRunner``,
  ``PipelineSpec``, ``PipelineStep``, ``StepResult``,
  ``PipelineResult`` dans l'API publique.
- ``picarones/evaluation/pipeline_benchmark.py`` — orchestre
  l'exécution corpus-wide via ``PipelineRunner.run()``.
- ``picarones/evaluation/pipeline_comparison.py`` — compare N
  ``PipelineSpec`` via ``run_pipeline_benchmark``.
- ``picarones/measurements/pipeline_spec_loader.py`` — charge des
  YAML legacy en ``PipelineSpec`` + ``PipelineStep`` legacy
  (avec instanciation des modules par ``adapter_name``).

**Tests** : 7 fichiers de tests directs (``test_sprint63_*``,
``test_sprint64_*``, ``test_sprint65_*``, ``test_sprint66_*``,
``test_sprint67_*``, ``test_sprint68_*``, etc.).

### 3.B Canonique (``pipeline.executor``)

**Production** : 0 caller production (le rewrite n'a pas encore
de service applicatif qui consomme l'executor canonique en
mode mono-document).

**Tests** : 9 fichiers de tests directs.  Tests-only à ce jour.

**Conclusion sur les callers** : le legacy est en production,
le canonique est test-only.  La convergence doit migrer le
legacy **sans casser les 7+4 = 11 fichiers tests/prod
existants**.

---

## 4. Stratégies de convergence

### 4.A Wrapper legacy → canonique

Le legacy ``PipelineRunner.run(spec, document, initial_inputs)``
devient un **adaptateur** qui :

1. Convertit la ``PipelineSpec`` legacy (dataclass + module
   instance) en ``PipelineSpec`` canonique (Pydantic +
   adapter_name).
2. Wrappe chaque ``BaseModule`` en ``StepExecutor`` Protocol.
3. Convertit les payloads bruts en ``Artifact`` (uri inline,
   content_hash calculé).
4. Injecte un ``adapter_resolver`` ad hoc qui retourne les
   wrappers.
5. Invoque ``PipelineExecutor.run(spec, document, initial_inputs,
   context)``.
6. Reconvertit le ``PipelineResult`` canonique en ``PipelineResult``
   legacy avec ``junction_metrics`` calculées à partir des
   artefacts produits.

**Avantages** :
- Préserve l'API legacy → 0 caller cassé en production.
- Unifie le moteur d'exécution → 1 seul code path à maintenir.
- Cohérent avec la philosophie "no breaking change for callers".

**Inconvénients** :
- 200-400 LOC de glue (conversion bidirectionnelle de types).
- Coût de performance : double conversion à chaque step.
- Le double modèle ``Artifact``/payload reste visible côté
  modules (le wrapper masque mais le concept demeure).

**Effort** : 2-3 sessions.

### 4.B Migration complète

Migrer chaque caller legacy vers l'API canonique :

1. ``pipeline_benchmark`` : passe de ``PipelineRunner.run`` à
   ``PipelineExecutor.run_plan``.  Les ``StepAggregate`` doivent
   accepter la nouvelle structure ``StepResult`` (Pydantic).
2. ``pipeline_comparison`` : idem.
3. ``pipeline_spec_loader`` : produit des ``PipelineSpec`` Pydantic
   au lieu de dataclass.  Plus de ``module`` instance — juste
   ``adapter_name``.
4. ``__init__.py`` : ré-exporte le canonique.
5. Tests : 7 fichiers à refactorer (mock adapters → ``StepExecutor``
   Protocol, payloads → ``Artifact``).

**Avantages** :
- 1 seul design.  Le legacy disparaît complètement.
- Pas de glue ni de double conversion.
- Conforme à la cible architecturale du rewrite.

**Inconvénients** :
- Massive : ~2500 LOC à toucher entre prod + tests.
- Le contrat des modules tiers (``BaseModule`` → ``StepExecutor``)
  change.  Un caller externe (BnF, HF Space) qui utilise
  ``PipelineRunner.run`` casse silencieusement.
- Risque de régression non détectée sur les ~7 tests sprints
  axe B (les fixtures sont volumineuses).
- Évaluation auto vs GT (legacy : à chaque step) doit être
  ré-implémentée comme une post-étape canonique.

**Effort** : 5-7 sessions.

### 4.C Cohabitation documentée

État actuel.  Document explicitement que les deux designs sont
volontaires.  Convergence reportée à un sprint dédié quand un
caller institutionnel l'exigera (BnF demande un YAML déclaratif
non-instanciable, ou HF Space veut le cache d'artefacts).

**Avantages** :
- 0 risque de régression maintenant.
- Permet de continuer le retrait du legacy sur les autres
  paquets (Phases 6-11) sans buter sur ce sujet complexe.
- Le canonique reste prêt pour le jour où il sera vraiment
  nécessaire.

**Inconvénients** :
- 2 designs à maintenir.
- L'objectif "core/ vide" du retrait du legacy n'est pas
  totalement atteint : ``evaluation/pipeline.py`` reste un module
  "legacy-style" en cercle 2.
- Risque que le canonique reste mort-né si personne ne le
  réclame.

**Effort** : 0 (juste documentation).

---

## 5. Recommandation

**Stratégie 4.A — Wrapper legacy → canonique** est la voie
recommandée :

- Préserve l'API publique → pas de breaking change pour les
  callers externes (BnF, HF Space, scripts).
- Unifie le moteur d'exécution → 1 seul path à maintenir.
- L'API legacy ``BaseModule`` reste le contrat utilisateur ;
  elle est juste implémentée par-dessus le canonique.
- Met en lumière les réelles différences sémantiques (cache,
  provenance, hash) qui peuvent être exposées progressivement à
  l'API legacy sans la casser.

La stratégie 4.B (migration complète) sera possible plus tard,
**après** 4.A : une fois le legacy unifié sur le canonique en
interne, les callers peuvent migrer un par un sans risque.

La stratégie 4.C (cohabitation) est pragmatique mais reporte
indéfiniment la dette.  Elle convient si la priorité est de
finir Phases 6-11 d'abord — ce qui est défendable.

---

## 6. Sub-plan d'exécution (stratégie 4.A)

### 6.A Sub-phase 1 — Adaptateur ``BaseModule`` → ``StepExecutor``

Crée un wrapper ``_BaseModuleAdapter`` qui satisfait le Protocol
``StepExecutor`` à partir d'une instance ``BaseModule`` :

- ``name`` ← ``module.name``
- ``input_types`` ← ``frozenset(module.input_types)``
- ``output_types`` ← ``frozenset(module.output_types)``
- ``execution_mode`` ← ``module.execution_mode``
- ``execute(inputs, params, context)`` :
  - convertit ``inputs: dict[ArtifactType, Artifact]`` en
    ``dict[ArtifactType, payload]`` (lit ``artifact.uri`` quand
    le payload est un fichier, sinon le payload brut est passé
    en clair).
  - appelle ``module.process(payload_inputs)``.
  - reconvertit le ``dict[ArtifactType, payload]`` retourné en
    ``dict[ArtifactType, Artifact]`` (calcule ``content_hash``
    + ``provenance`` automatiquement).

**Fichier** : ``picarones/evaluation/_pipeline_adapter.py``
(privé, ~150 LOC).

**Tests** : ``tests/evaluation/test_pipeline_adapter.py`` —
unitaires sur la conversion bidirectionnelle.

**Effort** : 1 session.

### 6.B Sub-phase 2 — ``PipelineRunner`` consomme ``PipelineExecutor``

Remplace le corps de ``PipelineRunner.run`` :

```python
@staticmethod
def run(spec: PipelineSpec_legacy, document: Document, initial_inputs: dict) -> PipelineResult_legacy:
    canonical_spec = _to_canonical_spec(spec)  # conversion
    canonical_inputs = _to_canonical_artifacts(initial_inputs)
    resolver = _LegacyAdapterResolver(spec)  # mappe step.name → wrapper
    executor = PipelineExecutor(adapter_resolver=resolver)
    canonical_result = executor.run(canonical_spec, document_ref, canonical_inputs, context)
    return _to_legacy_result(canonical_result, document, spec)  # reconvertit
```

Évaluation auto vs GT : ré-implémentée dans ``_to_legacy_result``
après l'exécution canonique (parcours des artefacts produits,
appel ``compute_at_junction``).

**Tests** : les 7 fichiers de tests legacy doivent passer
**inchangés** — c'est l'invariant de cette sub-phase.

**Effort** : 1-2 sessions (la conversion des résultats est
non-triviale à cause de ``junction_metrics``).

### 6.C Sub-phase 3 — Suppression du moteur legacy

Une fois la sub-phase 2 stable :

- Le code de ``run`` legacy entre lignes 384-590 de
  ``evaluation/pipeline.py`` (le moteur d'exécution proprement
  dit) est supprimé.
- Seules restent : les data classes (``PipelineSpec``,
  ``PipelineStep``, ``StepResult``, ``PipelineResult``) +
  l'adaptateur + la fonction ``_artifact_type_to_gt_level``.
- Le module passe de 607 LOC à ~250 LOC.

**Effort** : 0.5 session (suppression mécanique + tests qui
doivent toujours passer).

### 6.D Sub-phase 4 — Documentation & deprecation

- Document que ``evaluation.pipeline.PipelineRunner`` est un
  **wrapper de compatibilité** sur ``pipeline.executor.PipelineExecutor``.
- Émet ``DeprecationWarning`` à l'instanciation
  d'un ``PipelineSpec`` legacy si un caller externe l'utilise
  (warning silencieux dans le code interne).
- Pointe vers le canonique pour les nouveaux callers.

**Effort** : 0.5 session.

### 6.E Sub-phase 5 (optionnelle) — Migration des 4 callers internes

Avec le wrapper en place, on peut maintenant migrer
**incrémentalement** les 4 callers internes vers le canonique :

1. ``pipeline_spec_loader`` : produit des
   ``picarones.domain.pipeline_spec.PipelineSpec`` (Pydantic) au
   lieu du legacy.
2. ``pipeline_benchmark`` : consomme directement
   ``PipelineExecutor.run_plan``.
3. ``pipeline_comparison`` : idem.
4. ``__init__.py`` : ré-exporte les canoniques.

À ce stade, le legacy ``evaluation/pipeline`` ne contient plus
que des shims pour les callers externes (BnF, HF Space).

**Effort** : 2-3 sessions, par caller.

---

## 7. Total effort estimé

| Sub-phase | Description                                | Effort           |
|-----------|--------------------------------------------|------------------|
| 6.A       | Adaptateur ``BaseModule`` → ``StepExecutor`` | 1 session        |
| 6.B       | ``PipelineRunner`` consomme l'executor     | 1-2 sessions     |
| 6.C       | Suppression du moteur legacy               | 0.5 session      |
| 6.D       | Documentation & deprecation                | 0.5 session      |
| 6.E (opt) | Migration des 4 callers internes           | 2-3 sessions     |
| **Total** | **(jusqu'à 6.D)**                          | **3-4 sessions** |
| **Total** | **(jusqu'à 6.E)**                          | **5-7 sessions** |

---

## 8. Décision

À l'issue de cet audit, l'utilisateur décide :

1. **Aller** — démarrer la sub-phase 6.A maintenant.
2. **Reporter** — passer à Phase 6 (``pipelines/``) du retrait
   du legacy, garder la cohabitation documentée comme état
   provisoire.
3. **Hybrider** — faire 6.A + 6.B (wrapper en place) puis
   reporter 6.C-6.E.

L'option 2 (reporter) est défendable : Phases 6-11 du retrait du
legacy peuvent toutes se faire **sans avoir résolu cette
convergence**, et le travail accumulé donnera plus de signal sur
la priorité réelle de l'unification du pipeline.
