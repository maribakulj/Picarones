# Architecture en 3 cercles — chantier de refonte post-chantier 6

Ce document **fige la cartographie** de chaque module Picarones dans son
cercle d'appartenance. Il sert de référence stable pour les
contributions futures : avant d'ajouter un module, consulter ce
document pour identifier dans quel cercle il doit aller.

## Principe — 3 cercles concentriques

```
┌─────────────────────────────────────────────────────────────┐
│  Cercle 3 — Plugins (extras/)                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Cercle 2 — Modules officiels                       │    │
│  │  ┌──────────────────────────────────────────┐       │    │
│  │  │  Cercle 1 — Noyau invariant (core/)      │       │    │
│  │  │  API publique stable, ~15 modules        │       │    │
│  │  └──────────────────────────────────────────┘       │    │
│  │  Adapters, mesures, rapport, CLI, web               │    │
│  │  ~30 modules métriques + ~15 adapters/UI            │    │
│  └─────────────────────────────────────────────────────┘    │
│  Modules niche, gouvernance préventive, importers exotiques │
│  Distribués via extras pip ou packages séparés à terme      │
└─────────────────────────────────────────────────────────────┘
```

Plus on s'éloigne du cœur, plus c'est optionnel et plus c'est facile
à supprimer/remplacer/externaliser.

## Cercle 1 — Noyau invariant

**Critères** : ce qui définit *ce qu'est* Picarones. API publique
stable. Ne casse pas entre versions mineures.

**Localisation** : `picarones/core/` (après phase E) — strictement
~15 modules.

**Contenu** :

| Module | Rôle |
|---|---|
| `corpus.py` | Document, Corpus, GTLevel multi-niveaux |
| `modules.py` | BaseModule, ArtifactType (contrat unique pour modules tiers) |
| `results.py` | BenchmarkResult, EngineReport, DocumentResult |
| `metrics.py` | CER/WER/MER/WIL via jiwer (métriques de base) |
| `runner.py` | Orchestrateur (parallélisation, reprise, timeout) |
| `pipeline_runner.py` | Banc d'essai mono-doc des pipelines composées |
| `pipeline_benchmark.py` | Orchestration corpus-wide |
| `pipeline_comparison.py` | Comparaison de N pipelines |
| `pipeline_spec_loader.py` | Chargement YAML déclaratif |
| `metric_registry.py` | Registre typé `(input_type, output_type) → metric` |
| `metric_hooks.py` | Profils + registre de hooks document/corpus |
| `builtin_metrics.py` | CER/WER/MER/WIL enregistrés sur registre typé |
| `alto_metrics.py` | Métriques `(ALTO, ALTO)` (chantier 1) |

**Discipline** :
- Toute modification non rétrocompatible exige une **RFC** et bump majeur.
- Test `test_public_api.py` (à créer en phase D) qui échoue si un nom disparaît.
- Aucun import direct depuis `extras/` ou de modules optionnels.

## Cercle 2 — Modules officiels

**Critères** : maintenu par les mainteneurs Picarones, livré par
défaut, mais peut techniquement vivre ailleurs (un fork peut le
remplacer par un équivalent).

**Localisation** :
- `picarones/measurements/` (après phase E) — métriques au-delà du CER de base.
- `picarones/engines/` — adapters OCR.
- `picarones/llm/` — adapters LLM.
- `picarones/modules/` — modules `BaseModule` de référence (chantier 1).
- `picarones/report/` — génération HTML.
- `picarones/cli/` — interface CLI.
- `picarones/web/` — interface web FastAPI.
- `picarones/pipelines/` — pipelines OCR+LLM legacy (à statuer en phase D).

**Métriques officielles** (futur `picarones/measurements/`) :

| Catégorie | Modules |
|---|---|
| Texte | `confusion`, `char_scores`, `taxonomy`, `structure`, `taxonomy_comparison` |
| Lignes | `line_metrics`, `hallucination` |
| Fiabilité | `calibration`, `reliability`, `robustness`, `robustness_projection` |
| Structure ALTO/PAGE | `reading_order`, `layout`, `error_absorption` |
| Recherche | `searchability`, `numerical_sequences`, `rare_tokens` |
| Lisibilité | `readability` (Flesch), `specialization` |
| Inter-moteurs | `inter_engine`, `worst_lines` |
| Économie | `throughput`, `cost_projection`, `marginal_cost`, `pricing` |
| Comparaison | `incremental_comparison` |
| Narrative | `narrative/` (engine + 6 familles de détecteurs) |
| Hooks | `builtin_hooks` |
| Contexte corpus | `history`, `difficulty`, `image_quality`, `normalization` |
| Statistiques | `statistics` |
| Levers | `levers` |

**Discipline** :
- Modification libre sans RFC.
- Nouveau module doit s'enregistrer via `@register_metric` ou
  `@register_document_metric` plutôt qu'imports directs depuis `runner.py`.
- Couvre les 4 axes du produit : viabilité prod, hallucinations VLM,
  pipelines composées, projection coût/vitesse.

## Cercle 3 — Plugins

**Critères** : ne sert pas tout le monde, peut être désactivé sans
amputer le produit principal.

**Localisation** : `picarones/extras/` (sous-package interne pour
l'instant ; packages PyPI séparés possibles à terme).

**Sous-packages** :

### `extras/academic/` — modules techniques sans cas d'usage prod

| Module | Pourquoi en plugin |
|---|---|
| `taxonomy_intra_doc.py` | Heatmap classe×position. Question rare, peu actionnable |
| `taxonomy_cooccurrence.py` | Jaccard inter-classes. Académique, info rare |
| `image_predictive.py` | Score combiné avec poids éditoriaux arbitraires |

### `extras/governance/` — gouvernance préventive

| Module | Pourquoi en plugin |
|---|---|
| `module_policy.py` | Manifest + audit pour modules contribués externes. Inutile tant qu'il n'y a pas 5+ modules tiers réels |

### `extras/historical/` — métriques philologiques (phase B)

| Module | Public spécifique |
|---|---|
| `unicode_blocks.py` | Tous périodes |
| `abbreviations.py` | Médiéval (Capelli) |
| `mufi.py` | Médiéval (PUA) |
| `early_modern_typography.py` | XVIᵉ-XVIIIᵉ siècles |
| `modern_archives.py` | XIXᵉ-XXᵉ siècles |
| `roman_numerals.py` | Toutes périodes |
| `lexical_modernization.py` | Édition critique |
| `philological_runner.py` | Orchestration des 6 modules ci-dessus |

### `extras/importers/` — imports externes (phase C)

| Module | Statut |
|---|---|
| `_http.py` | Helpers HTTP partagés (chantier 4) |
| `iiif.py` | Maintenu |
| `htr_united.py` | Maintenu |
| `gallica.py` | Maintenu |
| `huggingface.py` | Expérimental (à finir ou marqué unstable) |
| `escriptorium.py` | Expérimental (à finir ou marqué unstable) |

### `extras/render/` — renderers correspondants

Renderers atomiques pour les modules `extras/`. Importés
conditionnellement par les vues thématiques du chantier 3 (qui sont
elles-mêmes dans `report/views/`, donc Cercle 2).

## Distinguer un module Cercle 1 vs Cercle 2

Test concret : si on supprime ce module, est-ce que la phrase
*« Picarones est un banc d'essai pour pipelines OCR/HTR/VLM »* reste
vraie ?

- **Oui** → Cercle 2 (le produit existe sans ce module).
- **Non** → Cercle 1 (le module participe à la définition même).

Exemple :
- Sans `corpus.py` : impossible de charger un corpus → Cercle 1.
- Sans `confusion.py` : on a toujours un bench fonctionnel sans
  matrice de confusion → Cercle 2.
- Sans `taxonomy_intra_doc.py` : on a toujours un bench complet et
  utile → Cercle 3.

## Distinguer un module Cercle 2 vs Cercle 3

Test concret : ce module sert-il à répondre à la question
*« peut-on déployer ce moteur en prod sur ce corpus dans nos
contraintes ? »* — soit en mesurant un risque (hallucinations,
stabilité), soit en projetant un coût (throughput, pricing), soit
en évaluant la qualité (CER, calibration, structure) ?

- **Oui** → Cercle 2.
- **Non** → Cercle 3.

Exemple :
- `hallucination.py` : mesure un risque pour la prod VLM → Cercle 2.
- `throughput.py` : projette un coût opérationnel → Cercle 2.
- `taxonomy_intra_doc.py` : décrit une distribution sans implication
  de décision → Cercle 3.

## Disclaimer

Cette cartographie est **une décision produit**, pas une vérité
absolue. Elle peut évoluer si les usages réels d'institutions
révèlent qu'un module Cercle 3 est en fait essentiel, ou
inversement.

Toute remise en cause doit passer par une RFC documentée, pas par
une PR silencieuse.

## Voir aussi

- [`docs/architecture.md`](architecture.md) — vue d'ensemble post-chantiers 1-6.
- [`docs/profiles.md`](profiles.md) — profils de calcul (chantier 2).
- [`docs/views.md`](views.md) — vues HTML du rapport.
- [`docs/cli-workflows.md`](cli-workflows.md) — commandes CLI.
- `docs/api-stable.md` — *à créer en phase D* — engagement API publique du Cercle 1.
