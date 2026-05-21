# Documentation Picarones — index par rôle

> **Architecture documentaire** : ce projet adopte le modèle
> [Diataxis](https://diataxis.fr/) — quatre quadrants :
> *tutorials* (apprendre), *how-to* (résoudre), *reference*
> (consulter), *explanation* (comprendre).  Plus deux dossiers
> institutionnels : *governance* et *operations*.
>
> **Bilingue** : la **langue canonique est le français**.  Une
> surface publique réduite est traduite en anglais — README,
> CONTRIBUTING, SECURITY, ACCESSIBILITY, deux tutoriels clés.
> Le reste reste FR.  Politique assumée plutôt que bilingue partiel
> brouillé.

---

## Je suis…

### …un chercheur ou archiviste qui veut benchmarker un corpus

Vous voulez exécuter Picarones sur vos documents, lire un rapport,
comprendre les chiffres.

1. Installer : [`how-to/install.md`](how-to/install.md)
2. Premier benchmark : [`tutorials/first-benchmark.md`](tutorials/first-benchmark.md)
3. Lire le rapport produit : [`tutorials/reading-a-report.md`](tutorials/reading-a-report.md)
   ([EN](tutorials/reading-a-report.en.md))
4. Cas d'école pédagogiques : [`case-studies/`](case-studies/)
5. Glossaire des métriques : [`reference/normalization-profiles.md`](reference/normalization-profiles.md),
   [`reference/views.md`](reference/views.md)

### …un opérateur qui doit déployer en environnement institutionnel

Vous installez Picarones sur un NAS BnF, un cluster LoC, un serveur BL.

1. Déploiement institutionnel : [`operations/deployment-institutional.md`](operations/deployment-institutional.md)
2. Conformité RGPD : [`operations/data-retention-rgpd.md`](operations/data-retention-rgpd.md)
3. Runbook incidents : [`operations/runbook.md`](operations/runbook.md)
4. Observabilité (logs, métriques, alerting) : [`operations/observability.md`](operations/observability.md)
5. Process de release : [`operations/release-process.md`](operations/release-process.md)

### …un développeur qui veut contribuer du code

Vous ajoutez un adapter, une vue, une métrique, un détecteur narratif.

1. Vue d'ensemble du projet : [`/CONTRIBUTING.md`](../CONTRIBUTING.md)
   ([EN](../CONTRIBUTING.en.md))
2. Architecture en cercles : [`explanation/architecture.md`](explanation/architecture.md)
3. Politique modules contribués : [`developer/module-policy.md`](developer/module-policy.md)
4. Étendre un sous-système :
   [glossaire](developer/extending-glossary.md) ([EN](developer/extending-glossary.en.md)) ·
   [i18n](developer/extending-i18n.md) ([EN](developer/extending-i18n.en.md)) ·
   [moteur narratif](explanation/narrative-engine.md) ([EN](explanation/narrative-engine.en.md))
5. Écrire un module pour le banc d'essai : [`tutorials/writing-a-pipeline-module.md`](tutorials/writing-a-pipeline-module.md)

### …un mainteneur ou auditeur de sécurité

Vous évaluez Picarones avant un déploiement, un audit, une revue.

1. Politique de gouvernance : [`/GOVERNANCE.md`](../GOVERNANCE.md)
2. Politique de sécurité : [`/SECURITY.md`](../SECURITY.md)
   ([EN](../SECURITY.en.md))
3. Threat model STRIDE : [`security/threat-model.md`](security/threat-model.md)
4. API publique stable et politique de versioning : [`reference/api-stable.md`](reference/api-stable.md)
5. Audits historiques : [`audits/`](audits/)
6. État du rewrite et migration : [`archives/migration/rewrite-status-s46.md`](archives/migration/rewrite-status-s46.md)
7. Reproductibilité bit-for-bit : [`reference/reproducibility-snapshots.md`](reference/reproducibility-snapshots.md)

### …un Délégué à la Protection des Données (DPO)

Vous évaluez les implications RGPD avant signature.

1. Politique de rétention RGPD : [`operations/data-retention-rgpd.md`](operations/data-retention-rgpd.md)
2. Modèle d'accord de sous-traitance (DPA) : [`legal/dpa-template.md`](legal/dpa-template.md)
3. Threat model : [`security/threat-model.md`](security/threat-model.md)
4. Liste des sous-traitants potentiels (services cloud) :
   `pricing.yaml` + section *Adapters cloud* dans
   [`reference/api-stable.md`](reference/api-stable.md)

---

## Index thématique

### Tutorials — j'apprends

| Document | Public | Langue |
|----------|--------|--------|
| [`tutorials/first-benchmark.md`](tutorials/first-benchmark.md) | Chercheur découvrant l'outil | FR |
| [`tutorials/reading-a-report.md`](tutorials/reading-a-report.md) | Chercheur lisant un rapport | FR + EN |
| [`tutorials/writing-a-pipeline-module.md`](tutorials/writing-a-pipeline-module.md) | Développeur tiers | FR |

### How-to — je résous un problème concret

| Document | Cible |
|----------|-------|
| [`how-to/install.md`](how-to/install.md) | Installer en local ou serveur |
| [`how-to/cli-workflows.md`](how-to/cli-workflows.md) | Utiliser la CLI au quotidien |

### Reference — je consulte le contrat

| Document | Sujet |
|----------|-------|
| [`reference/api-stable.md`](reference/api-stable.md) | API Python publique + politique semver |
| [`reference/views.md`](reference/views.md) | Vues d'évaluation (text, alto, search) |
| [`reference/normalization-profiles.md`](reference/normalization-profiles.md) | Profils de normalisation textuelle |
| [`reference/reproducibility-snapshots.md`](reference/reproducibility-snapshots.md) | Reproductibilité bit-for-bit |

### Explanation — je comprends pourquoi

| Document | Sujet |
|----------|-------|
| [`explanation/architecture.md`](explanation/architecture.md) | Architecture en cercles, principes |
| [`explanation/narrative-engine.md`](explanation/narrative-engine.md) | Comment le moteur narratif fonctionne |

### Operations — je déploie et j'opère

| Document | Sujet |
|----------|-------|
| [`operations/deployment-institutional.md`](operations/deployment-institutional.md) | Déploiement institutionnel |
| [`operations/runbook.md`](operations/runbook.md) | Réponse aux incidents |
| [`operations/observability.md`](operations/observability.md) | Logs, métriques, alerting |
| [`operations/data-retention-rgpd.md`](operations/data-retention-rgpd.md) | Conformité RGPD |
| [`operations/release-process.md`](operations/release-process.md) | Cycle de release |

### Governance / security / legal

| Document | Sujet |
|----------|-------|
| [`/GOVERNANCE.md`](../GOVERNANCE.md) | Gouvernance |
| [`/SECURITY.md`](../SECURITY.md) | Sécurité (FR + EN) |
| [`/CODE_OF_CONDUCT.md`](../CODE_OF_CONDUCT.md) | Code de conduite |
| [`/ACCESSIBILITY.md`](../ACCESSIBILITY.md) | Accessibilité |
| [`security/threat-model.md`](security/threat-model.md) | Threat model STRIDE |
| [`legal/dpa-template.md`](legal/dpa-template.md) | DPA RGPD §28 |

### Archives et historique

| Document | Sujet |
|----------|-------|
| [`/CHANGELOG.md`](../CHANGELOG.md) | Journal des versions (Keep-a-Changelog) |
| [`audits/`](audits/) | Audits historiques figés |
| [`migration/`](migration/) | Notes de migration entre versions majeures |
| [`roadmap/`](roadmap/) | Plans stratégiques |

---

## Conventions

- **Une seule arborescence canonique (v2.0)** :
  `domain → formats → evaluation → pipeline → adapters → app → reports → interfaces`.
  Les paquets legacy ont été supprimés en mai 2026.
- **Tout chemin `picarones/.../X.py` cité dans la doc doit exister**.
  Vérifié par `tests/architecture/test_doc_paths.py` (ratchet
  strictement décroissant).
- **Les tableaux générés** (engines, CLI, endpoints) sont régénérés
  par `scripts/gen_readme_tables.py` — modifier le code, pas la doc.
  Les compteurs en prose (nombre de tests, etc.) utilisent la
  formulation approximative `N+ tests` pour absorber la dérive
  OS-dépendante ; le chiffre exact vit dans le badge CI.
- **Cohérence FR/EN** : la langue canonique est le français.  Une
  surface EN réduite est listée dans
  `tests/docs/test_translation_parity.py::TRANSLATION_PAIRS` —
  toute paire FR/EN doit y figurer.
