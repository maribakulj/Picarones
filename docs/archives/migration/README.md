# Archives — plans de migration legacy → rewrite (2026-04 / 2026-05)

Ces documents décrivent la migration de l'arborescence pré-rewrite
(`picarones/{core,measurements,engines,modules,report,llm,
pipelines,cli,web,extras}/`) vers l'architecture canonique en 8
couches (`domain → formats → evaluation → pipeline → adapters →
app → reports → interfaces`).

**La migration est terminée à v2.0** (mai 2026).  Tous les
paquets legacy ont été supprimés ; ces plans sont archivés ici à
titre historique.

## Contenu

| Document | Statut historique |
|---|---|
| `legacy-retirement-plan.md` | Plan maître des Phases 0-11 + Lots A-G + Sprints H.1-H.6.  Cartographie complète du retrait. |
| `pipeline-convergence-plan.md` | Sub-plan de la Phase 7 (BaseModule → StepExecutor). |
| `rewrite-status-s46.md` | Snapshot de l'état du rewrite ciblé après le Sprint 46 (avant l'audit institutionnel S47-S59). |
| `sprint-D-audit.md` | Audit du Sprint D (façade `run_benchmark_via_service`). |
| `executor-equivalence.md` | Preuve d'équivalence numérique entre `PipelineRunner` legacy et `PipelineExecutor` canonique. |
| `regression-tolerances.md` | Contrat des tolérances pour le harness `tests/regression/legacy_vs_rewrite/` (harness lui-même retiré au cleanup post-v2.0). |
| `SESSION_HANDOVER.md` | Procédure de reprise de session entre instances Claude pendant la migration.  Caduc à v2.0. |

## Pourquoi conserver ces archives ?

1. **Traçabilité institutionnelle** : la BnF et autres consommateurs
   doivent pouvoir auditer la genèse de l'architecture v2.0.
2. **Documentation des choix** : les plans expliquent *pourquoi*
   tel module a été déplacé là plutôt qu'ailleurs.
3. **Référence pour les renames futurs** : si un mainteneur
   futur doit re-déplacer un module, ces plans documentent les
   contraintes architecturales (whitelist, inward-only, layer
   imports legal).

## Lecture pour comprendre l'architecture actuelle

Pour la doc **active**, voir :

- [`docs/explanation/architecture.md`](../../explanation/architecture.md) — manifeste 8 couches.
- [`docs/reference/api-stable.md`](../../reference/api-stable.md) — contrat de stabilité API.
- [`CLAUDE.md`](../../../CLAUDE.md) — quickstart pour mainteneurs.
- [`CHANGELOG.md`](../../../CHANGELOG.md) section [2.0.0] — résumé des suppressions.
