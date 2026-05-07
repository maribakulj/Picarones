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

> **Mise à jour 2026-05** : l'utilisateur a précisé que le projet
> est en stand-by jusqu'à la fin de la migration complète et que
> la rétrocompat de l'API publique n'est pas une contrainte.  Cela
> élimine l'avantage principal de la stratégie 4.A (wrapper) et
> rend la stratégie 4.B (migration complète) recommandée :

**Stratégie 4.B — Migration complète** est la voie cible.

Bénéfices avec contrainte API levée :

- 1 seul design final, plus de wrapper interne à maintenir.
- Le contrat des modules tiers (``BaseModule`` → ``StepExecutor``)
  peut changer sans gérer la rétrocompat.
- Les ``Artifact`` typés (provenance, content_hash, uri) deviennent
  natifs partout — pas de double conversion.

Risques résiduels :

- ~2500 LOC à toucher entre prod + tests.
- L'évaluation auto vs GT (legacy : à chaque step) doit être
  ré-implémentée comme une post-étape canonique.
- Risque de régression sur les ~7 tests sprints axe B
  (fixtures volumineuses).
- Plusieurs sessions de travail nécessaires (5-7 sessions).

---

## 6. Découvertes additionnelles (audit complémentaire)

L'audit initial parlait de 4 callers de production de
``PipelineRunner``.  Une investigation plus poussée révèle un
écosystème legacy plus large, qui doit être inclus dans le plan :

### 6.A Legacy engines (`picarones/engines/`, ~1500 LOC)

5 modules OCR legacy qui héritent de ``BaseOCREngine`` (lui-même
extension de ``BaseModule``) :

- ``engines/base.py:BaseOCREngine``
- ``engines/tesseract.py:TesseractEngine`` (177 l)
- ``engines/pero_ocr.py:PeroOCREngine`` (182 l)
- ``engines/mistral_ocr.py:MistralOCREngine`` (231 l)
- ``engines/google_vision.py:GoogleVisionEngine`` (256 l)
- ``engines/azure_doc_intel.py:AzureDocIntelEngine``

**Équivalents canoniques existent** dans
``picarones/adapters/ocr/`` (TesseractAdapter, PeroOCRAdapter,
etc.) et implémentent déjà ``StepExecutor``.  Mais les noms de
classes et les APIs publiques **diffèrent** — pas un simple shim.

Callers production des engines legacy :
- ``picarones/web/benchmark_utils.py``
- ``picarones/pipelines/base.py`` (lui-même legacy, Phase 6)

### 6.B Legacy LLM (``picarones/llm/``, ~67 LOC)

**Déjà migré** : tous les fichiers sont des shims qui
ré-exportent depuis ``picarones/adapters/llm/``.  Rien à faire.

### 6.C Legacy modules officiels (``picarones/modules/``)

- ``modules/alto_text_to_mono_region.py:TextToAltoMonoRegion``
  (310 LOC) — extension de ``BaseModule``.

**Pas d'équivalent canonique** à ce jour.  Cible documentée :
``picarones/formats/alto/baseline_reconstruction.py`` ou
``picarones/evaluation/projectors/text_to_alto.py``
(cf. Phase 7 du plan de retrait).

### 6.D Sémantique des payloads vs Artifacts

La conversion ``BaseModule.process`` ↔ ``StepExecutor.execute``
n'est pas triviale parce que :

- Le legacy passe des **payloads bruts** :
  - ``ArtifactType.IMAGE`` → ``str`` (chemin du fichier image)
  - ``ArtifactType.RAW_TEXT`` → ``str`` (contenu textuel inline)
  - ``ArtifactType.ALTO_XML`` → ``str`` (contenu XML inline)
  - ``ArtifactType.ENTITIES`` → ``list[dict]``
- Le canonique passe des ``Artifact`` Pydantic immutables :
  - ``uri`` (filesystem ou URI distant)
  - ``content_hash`` (SHA-256)
  - ``provenance`` (``ProvenanceRecord``)
  - **pas de champ ``content`` direct** — le contenu se lit via
    ``uri``.

Pour les tests legacy qui injectent du contenu inline (mock
modules retournant ``"hello"``), il faut **soit** :

1. Persister le contenu dans un fichier temporaire et pointer
   ``artifact.uri`` dessus.
2. Ajouter une convention ``data:`` URI pour le contenu inline.
3. Étendre ``Artifact`` avec un champ ``inline_payload: bytes |
   None`` optionnel.

Décision recommandée : **option 1** (fichier temporaire), parce
qu'elle préserve la sémantique « un artefact a toujours un
identifiant filesystem » et permet le cache/provenance proprement.

---

## 7. Sub-plan d'exécution révisé (stratégie 4.B)

### Sub-phase 7.A — Migration des adapters concrets

Bouclage de la migration des adapters legacy (engines/llm/modules)
vers les canoniques avant de toucher aux pipeline runners.

**Étapes** :

1. ``engines/`` → shims pointant vers ``adapters/ocr/`` (avec
   alias de classes : ``TesseractEngine = TesseractAdapter``,
   etc.).
2. Mise à jour des callers de ``engines/`` à utiliser
   ``adapters/ocr/`` directement.
3. ``modules/alto_text_to_mono_region.py`` → migré vers
   ``picarones/evaluation/projectors/text_to_alto.py`` (canonique
   en ``StepExecutor``).
4. Suppression du shim ``engines/``.

**Effort** : 2-3 sessions.

### Sub-phase 7.B — Migration des callers ``PipelineRunner``

Une fois les adapters unifiés sur ``StepExecutor`` :

1. ``pipeline_spec_loader`` : produit des ``picarones.domain.pipeline_spec.PipelineSpec``
   (Pydantic) avec ``adapter_name: str`` au lieu d'instances.
2. ``pipeline_benchmark`` : consomme ``PipelineExecutor.run_plan``.
   ``StepAggregate`` accepte ``StepResult`` Pydantic canonique.
3. ``pipeline_comparison`` : idem.
4. ``__init__.py`` : ré-exporte les canoniques.

**Effort** : 2 sessions.

### Sub-phase 7.C — Refactor des tests

Les 7 fichiers de tests legacy axe B (sprints 63-68 + 95) :

- Mocks ``BaseModule`` → mocks ``StepExecutor`` Protocol.
- Payloads bruts → ``Artifact`` (avec helper
  ``make_inline_artifact(content, type_)`` pour réduire le
  boilerplate).
- ``Document`` legacy → ``DocumentRef`` canonique.
- Fixtures ``junction_metrics`` → ré-implémentation via
  post-étape canonique.

**Effort** : 1-2 sessions.

### Sub-phase 7.D — Suppression du legacy

1. Suppression de ``evaluation/pipeline.PipelineRunner``,
   ``PipelineSpec``, ``PipelineStep``, ``StepResult``,
   ``PipelineResult`` (le legacy).
2. Suppression de ``domain/module_protocol.BaseModule``.
3. Le module ``evaluation/pipeline.py`` réduit à
   ``_artifact_type_to_gt_level`` ou supprimé totalement.
4. ``core/pipeline.py`` (shim) supprimé.
5. ``core/modules.py`` (shim) supprimé.

**Effort** : 0.5 session (suppression mécanique).

---

## 8. Total effort révisé (stratégie 4.B)

| Sub-phase | Description                                | Effort           |
|-----------|--------------------------------------------|------------------|
| 7.A       | Migration adapters concrets                | 2-3 sessions     |
| 7.B       | Migration callers PipelineRunner           | 2 sessions       |
| 7.C       | Refactor des tests                         | 1-2 sessions     |
| 7.D       | Suppression du legacy                      | 0.5 session      |
| **Total** | **Migration complète**                     | **5-8 sessions** |

---

## 9. Ordre d'exécution recommandé

L'ordre **bottom-up** est plus sûr : à chaque étape, les tests
restent verts.

```
Sub-phase 7.A (adapters) → Sub-phase 7.B (orchestration) →
Sub-phase 7.C (tests) → Sub-phase 7.D (suppression)
```

L'ordre **top-down** (start by removing PipelineRunner, then
fix everything that breaks) est plus risqué mais plus rapide
si on accepte une période de tests rouges.

Recommandation : **bottom-up**, par étapes verticales testables.
