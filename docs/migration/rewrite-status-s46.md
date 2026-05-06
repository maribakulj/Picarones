# État du rewrite — Sprint A14-S46 (clôture phase rewrite ciblé)

Ce document synthétise l'état du rewrite du Picarones après les 20 sprints
S27-S46 réalisés sur la directive *« rewrite tout, le plus solide, sans
dette technique »*.

## Phase 7 (S46) : retraite progressive du legacy

Le rewrite est **fonctionnellement complet** côté contrats et architecture
(circles propres, services applicatifs, adapters natifs OCR/LLM/VLM,
pipeline planner, artifact store, web UI native). Le legacy
(`picarones/{cli,web,engines,llm,pipelines,report}/`) reste néanmoins en
place pour deux raisons :

1. **Parité fonctionnelle non encore atteinte** : le legacy `report/`
   contient ~22 vues HTML thématiques (Pareto, narrative, glossary,
   case-studies, etc.) que `reports_v2/html/` ne reproduit pas
   intégralement. Les vues canoniques (TextView, AltoView, SearchView)
   sont en place ; les vues additionnelles arriveront post-livraison
   selon les besoins BnF.

2. **Tests legacy** : ~200+ tests legacy valident le comportement
   historique (`tests/web/`, `tests/measurements/`, `tests/cli/_workflows/`,
   `tests/integration/test_chantier*.py`, etc.). Les supprimer
   prématurément perdrait la couverture.

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

## Statistiques globales du rewrite (S1-S45)

- **Tests** : ~4910 tests, 11 skipped, 0 failed (vs 4504 au début du
  rewrite, S26).
- **+406 nouveaux tests** sur S27-S45 (rewrite ciblé).
- **Lint** : `ruff check picarones/ tests/` clean.
- **File budgets** : tous les fichiers ≥ 400 lignes surveillés et
  budgétés.
- **Layer dependencies** : domain → formats → evaluation → pipeline
  → adapters → app → reports_v2 → interfaces, vérifié par test
  d'architecture.

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
