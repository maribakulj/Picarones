# Archive documentaire — Picarones

> **Archived document.** Cette zone contient des documents
> **historiques** : audits, plans de migration, roadmaps obsolètes,
> changelogs antérieurs à v2.0.  Ils sont conservés pour la
> traçabilité et la mémoire institutionnelle du projet, mais ne font
> **pas** partie de la documentation active.
>
> Pour la documentation à jour, voir [`docs/index.md`](../index.md).

---

## Pourquoi cette séparation

À v2.0 (mai 2026), le projet a clôturé son rewrite et la migration
legacy.  Les artefacts qui décrivaient ces chantiers (handovers de
session, plans de retrait, audits institutionnels, status reports)
restent utiles pour comprendre **comment** le code a évolué, mais
ils ne décrivent pas l'état actuel.

Les laisser dans la documentation active créait deux confusions :

1. **Pour les nouveaux contributeurs** : impression que le projet
   était encore en chantier, alors que la posture v2.0 est
   « release ».
2. **Pour les outils CI** (ratchet `test_doc_paths`, validation
   `mkdocs strict`) : ces fichiers contiennent des références à du
   code supprimé qui empêchaient mécaniquement le compteur de dette
   de décroître.

L'archive résout les deux : zone explicite, exclue de la nav active
et du ratchet, bannière en tête de chaque sous-dossier.

---

## Contenu

### [`2026-audits/`](2026-audits/)

Audits institutionnels et plans de remédiation produits en mai 2026,
préalables au tag v2.0.

- `institutional-readiness-2026-05.md` — audit BnF / LoC / BL
- `remediation-plan-2026-05.md` — plan d'action issu de l'audit

### [`2026-migration/`](2026-migration/)

Plans, handovers et inventaires liés au rewrite et à la suppression
du legacy (lots A-H, mai 2026).

- `legacy-retirement-plan.md` — plan de retrait des paquets legacy
- `pipeline-convergence-plan.md` — convergence des deux pipelines
- `rewrite-status-s46.md` — état d'avancement Sprint 46
- `executor-equivalence.md` — preuve d'équivalence des exécutants
- `option-b-user-guide.md` — guide de migration vers Option B
- `option-b-test-inventory.md` — inventaire des tests touchés
- `session-handover.md` — handover entre sessions de travail
- `sprint-D-audit.md` — audit du sprint D
- `regression-tolerances.md` — tolérances de régression mesurées

### [`2026-roadmap/`](2026-roadmap/)

Plans stratégiques pré-v2.0 (le backlog vivant est dans
[`/docs/roadmap/backlog.md`](../roadmap/backlog.md)).

- `evolution.md` — plan d'évolution 2026 (axes A et B)
- `rewrite.md` — plan du rewrite 2026 (S1–S26)

### [`changelog-pre-v2.md`](changelog-pre-v2.md)

Historique de versions 0.1.0 → 1.x + chantiers pré-v2.0.  Le
CHANGELOG actif (post-v2.0) vit à la racine du repo dans
[`/CHANGELOG.md`](../../CHANGELOG.md).

---

## Politique de modification

Les fichiers archivés sont **figés**.  Toute correction substantielle
doit aller dans la documentation active, pas dans l'archive.  Les
seules modifications acceptables ici sont :

- ajouter le bandeau « Archived document » s'il manque ;
- corriger des typos évidentes ;
- ajouter des notes en tête pour préciser le contexte historique.

Les chemins Python obsolètes (`picarones.measurements`, `picarones.core`,
etc.) restent **volontairement** dans les fichiers archivés —
c'est précisément pour cela qu'ils sont archivés.  Le test
`tests/architecture/test_doc_paths.py` exclut `docs/archive/**` de
son périmètre.
