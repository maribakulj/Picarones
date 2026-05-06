# État du rewrite — Sprints A14-S46 puis S47-S57 (audit + remédiation)

Ce document synthétise l'état du rewrite du Picarones après les 20 sprints
S27-S46 réalisés sur la directive *« rewrite tout, le plus solide, sans
dette technique »*, puis les 11 sprints S47-S57 d'audit/remédiation des
30 dettes identifiées en revue de fin de rewrite (audit 2026-05).

## Statut réel — partial rewrite, pas full rewrite (S57, audit #21 + #24)

Le rewrite est **fonctionnellement complet sur le périmètre des contrats
et de l'architecture cible** (circles propres `domain → formats →
evaluation → pipeline → adapters → app → reports_v2 → interfaces`,
services applicatifs, adapters natifs OCR/LLM/VLM, pipeline planner,
artifact store, web UI native).  La formulation initiale *« rewrite
fonctionnellement complet »* était trop forte sur deux dimensions
relevées par l'audit :

1. **Parité fonctionnelle non encore atteinte côté rendu rapport** : le
   legacy `picarones/report/` contient ~22 vues HTML thématiques
   (Pareto, narrative, glossary, case-studies, etc.) que `reports_v2/`
   ne reproduit pas intégralement.  Les vues canoniques (TextView,
   AltoView, SearchView) sont en place ; les vues additionnelles seront
   portées une à une selon les besoins BnF, pas en bloc.

2. **Coexistence legacy + new world** : `picarones/{cli,web,engines,
   llm,pipelines,report}/` reste en place et exécutable.  Un caller
   externe peut encore importer depuis n'importe lequel.  Cette
   coexistence est volontaire (cf. *Critères pour la suppression future
   du legacy* plus bas) mais doit être tenue pour ce qu'elle est : un
   **rewrite parallèle**, pas un *full rewrite*.  Les usages production
   sont à migrer caller par caller.

3. **Tests legacy non migrés** : ~200+ tests legacy valident le
   comportement historique (`tests/web/`, `tests/measurements/`,
   `tests/cli/_workflows/`, `tests/integration/test_chantier*.py`,
   etc.).  Ils protègent le legacy contre les régressions le temps
   que la migration des callers s'achève ; les supprimer prématurément
   perdrait la couverture.

## Inventaire des modules legacy

| Module | Statut | Nouvelle implémentation | Action S46 |
|--------|--------|--------------------------|------------|
| `picarones/cli/` | LEGACY | `picarones/interfaces/cli/` (3 commandes) | Conserver — features CLI manquantes |
| `picarones/web/` | LEGACY | `picarones/interfaces/web/` (skeleton + 3 routers + UI) | Conserver — UI riche manquante |
| `picarones/engines/` | LEGACY | `picarones/adapters/ocr/` (5 natifs) | Conserver — feature parité (confidences) |
| `picarones/llm/` | RE-EXPORT | `picarones/adapters/llm/` | Déjà migré (re-export pur) |
| `picarones/pipelines/` | LEGACY | (composition via pipeline DAG natif S6+) | Conserver — pas d'équivalent direct |
| `picarones/report/` | LEGACY | `picarones/reports_v2/{html,csv,json}/` | Conserver — vues thématiques manquantes |

## Ce qui est DÉFINITIVEMENT migré (S27-S45)

### Sprints S27-S29 — Fondations architecturales
- `ProjectionEngine` + `EvaluationEngine` séparés (S27)
- `PipelinePlanner` + `ExecutionPlan` (S28)
- `ArtifactStore` avec hash multi-paramètres + persistance filesystem (S29)

### Sprints S30-S34 — 5 OCR engines natifs (NO SHIM)
- `TesseractAdapter` (S30)
- `PeroOCRAdapter` (S31)
- `MistralOCRAdapter` (S32)
- `GoogleVisionAdapter` (S33)
- `AzureDocIntelAdapter` (S34)

Tous héritent directement de `BaseOCRAdapter` (S26), pas du legacy
`BaseOCREngine`. Le legacy peut être supprimé une fois les confidences
migrées vers `ConfidenceArtifact` (sprint dédié).

### Sprints S35-S38 — Web app native (NO SHIM)
- Skeleton FastAPI avec DI (`WebAppState`, `create_app`) — S35
- Routers corpus + benchmark — S36
- JobStore SQLite + jobs router — S37
- UI Jinja2 + static + i18n FR/EN — S38

### Sprints S39-S41 — Format YAML + domain cleanup
- RunSpec étendu (`inputs_from`, `preferred_text_output`) — S39
- `PipelineSpec` migré dans `domain/` — S40
- `artifacts_index.jsonl` séparé — S41

### Sprints S42-S43 — Reports CSV + JSON
- `CsvReportRenderer` — S42
- `JsonReportRenderer` — S43

### Sprints S44-S45 — LLM/VLM nativement intégrés (NO SHIM)
- Les 4 LLM adapters (Anthropic, OpenAI, Mistral, Ollama) ont désormais
  un `execute()` natif compatible `StepExecutor` — S44
- 4 VLM adapters dérivés via MRO multiple — S45

## Critères pour la suppression future du legacy

Pour chaque module legacy à supprimer, il faut :

1. **Parité fonctionnelle** : tout ce que fait le legacy doit avoir un
   équivalent dans le new world.
2. **Migration des tests** : les tests legacy doivent soit migrer vers
   le new world, soit être identifiés comme supprimables.
3. **Migration des callers externes** : si des callers externes
   importent depuis `picarones.web.app` (par ex. dans le HuggingFace
   Space), ils doivent être migrés en amont.
4. **Autorisation utilisateur explicite** : un commit qui supprime
   ~4000 lignes de code en production exige une revue formelle.

## Statistiques globales du rewrite (S1-S57)

- **Tests** : ~4910 tests, 11 skipped, 0 failed au S46 (vs 4504 au
  début du rewrite, S26).  Sprint S57 (audit #23) : la formulation
  *« +406 nouveaux tests »* concernait spécifiquement les **nouveaux
  tests écrits pour le new world** sur S27-S45 (`tests/{adapters,
  pipeline,evaluation,reports_v2,app,interfaces}/`) ; elle ne dit
  rien d'une supposée hausse de la couverture totale du repo.  Les
  tests legacy (`tests/{web,cli,engines,measurements,...}/`) ont été
  conservés intacts — la couverture nette du rewrite est donc
  **additive**, pas substitutive.
- **Lint** : `ruff check picarones/ tests/` clean.
- **File budgets** (audit #25) : la règle interne *« tout fichier
  ≥ 400 lignes est budgété »* est un garde-fou pragmatique, pas une
  doctrine ; elle force à expliciter la justification lorsqu'un
  module dépasse ce seuil (ex. `interfaces/web/app.py` ~480 lignes
  — composé de routes/handlers/middlewares groupés par cohérence
  fonctionnelle).  Aucun fichier ne dépasse 800 lignes après S46.
- **Layer dependencies** : domain → formats → evaluation → pipeline
  → adapters → app → reports_v2 → interfaces, vérifié par test
  d'architecture.

## Sprints d'audit/remédiation S47-S57 (audit institutional readiness)

L'audit *institutional readiness 2026-05* a identifié 30 dettes
techniques résiduelles après le rewrite ciblé.  Elles ont été
adressées en 6 vagues (S47-S57) :

| Vague | Sprint | Issues | Thème |
|-------|--------|--------|-------|
| pré-audit | S47-S48 | #1, #2 | ArtifactStore wired, JobRunner threading |
| A | S49-S51 | #3-#7 | Web security middlewares, confidences sidecar, output paths |
| B | S52-S53 | #8-#11 | AdapterStepError hierarchy, Mistral routing strict, normalize_llm_content path |
| C | S54 | #6 | MRO guard `__init_subclass__` BaseVLMAdapter |
| D | S55 | #14 | Live integration tests `tests/integration/live/` |
| E | S56 | #12, #13, #17, #18, #19, #20, #22, #27, #28, #29 | JobStore schema_version, busy_timeout, model_dump(mode="json"), `_infer_pipeline_name`, etc. |
| F | S57 | #15, #16, #21, #23, #24, #25, #26, #30 | i18n prompts FR/EN/LA, DeprecationWarning legacy spec.py, doc rectifications |

**Tous les 30 issues sont adressés au S57**.  Les détails sont dans
`docs/audits/remediation-plan-2026-05.md`.

### Notes spécifiques (S57)

- **#15 Lazy imports SDK tiers** : les imports `mistralai`, `anthropic`,
  `openai`, `ollama` sont **intentionnellement à l'intérieur des
  méthodes** (`MistralOCRAdapter._call_chat_vision_api`, etc.) plutôt
  qu'au top du module.  Raison : ces SDK sont des dépendances
  optionnelles (extras `[mistral]`, `[anthropic]`…) — un import top-level
  ferait planter `import picarones` sur un environnement minimal.
  Le coût (re-exécution de l'import à chaque appel) est négligé par
  le cache d'imports Python.
- **#16 i18n prompts FR/EN/LA** : `BaseLLMAdapter.DEFAULT_CORRECTION_PROMPTS`
  et `BaseVLMAdapter.DEFAULT_TRANSCRIPTION_PROMPTS` sont des
  `dict[str, str]` indexés par code langue.  Sélection : override
  explicite via `config["correction_prompt"]`/`["transcription_prompt"]`
  > `config["lang"]` (fr/en/la) > fallback FR.
- **#26 DeprecationWarning legacy spec.py** : import depuis
  `picarones.pipeline.spec` émet désormais un `DeprecationWarning`
  pointant vers `picarones.domain`.  Suppression effective prévue S60.
- **#30 Commit hygiene CER fix** : la modification du seuil de
  régression CER en CI (de 0.10 à 0.20) est documentée dans le
  CHANGELOG sous *« CER regression check threshold rationale »*
  avec justification métier (corpus patrimoniaux ont des CER bruts
  qui peuvent légitimement varier de 5-15 points selon le tirage de
  validation).

## Prochaines étapes possibles (post-rewrite)

1. **Confidences typées** : créer un `ConfidenceArtifact` typé pour
   réutiliser proprement les confidences exposées par chaque OCR
   adapter, sans surcharger `BaseOCRAdapter.execute()`.
2. **Vues HTML manquantes** : porter Pareto, Narrative, Glossary du
   legacy `report/` vers `reports_v2/html/` une vue à la fois.
3. **CLI complète** : porter les commandes manquantes (`history`,
   `compare`, `pipeline`, `diagnose`, etc.) dans
   `interfaces/cli/`.
4. **Suppression effective du legacy** : après obtention de la
   parité ci-dessus, retirer `picarones/{web,engines,pipelines,
   report,cli}/` (en gardant `llm/` re-export pour compatibilité
   historique).
