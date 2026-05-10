# Backlog post-livraison

> **Garde-fou de discipline du rewrite ciblé** (cf. `docs/roadmap/rewrite-2026.md`).
>
> Tout ce qui apparaît ici est **explicitement hors scope** des sprints
> S1–S26. Ces items pourront revenir dans le scope après la livraison à
> la BnF, pas avant.
>
> La règle d'or : "à chaque doute pendant le sprint en cours, l'item va
> ici et le sprint continue."

---

## 1. Promesses retirées du README

Items historiquement présentés comme acquis et qui ne sont en réalité
pas tenus au niveau qui justifierait leur affirmation publique.

### 1.1 Scientific publication track

- `CITATION.cff` au format Citation File Format 1.2.
- DOI Zenodo (snapshot release).
- Soumission JOSS (Journal of Open Source Software) avec article
  technique.
- BibTeX généré automatiquement par release.

**Pourquoi retiré du README pour l'instant** : la posture éditoriale
sera difficile à tenir tant que le rewrite ciblé n'est pas livré et
qu'on ne peut pas pointer vers une version 2.0 stable.

**Quand revoir** : après S26.

### 1.2 Conformité RGPD opérationnelle

- Audit DPO interne ou externe.
- Registre des traitements documenté.
- Politique de rétention enforced (pas seulement documentée).
- Mécanisme d'exercice des droits (export, suppression).

**État actuel** : `docs/operations/data-retention-rgpd.md` existe mais
n'a jamais été validé par un DPO ni testé sur un workflow réel BnF.

### 1.3 Gouvernance et COI policies

- Constitution explicite du comité de pilotage.
- Politique de gestion des conflits d'intérêts exercée sur ≥ 1 PR
  externe.
- Processus de release reviews documenté et appliqué.

**État actuel** : `GOVERNANCE.md` et `CONTRIBUTING.md` sont en place
comme documents de référentiel mais aucun de ces processus n'a été
exercé en pratique.

### 1.4 Accessibilité WCAG 2.1 AA

- Audit RGAA externe.
- Tests automatisés axe-core sur la SPA.
- Navigation complète clavier validée par utilisateur empêché.

**État actuel** : `ACCESSIBILITY.md` documente l'intention. Les
améliorations Sprint 25 (extraction du JS inline vers
`web-app.js`) sont un pas dans la bonne direction mais ne suffisent
pas à revendiquer la conformité.

### 1.5 Sécurité — pentest externe

- Pentest opérationnel sur un déploiement institutionnel (pas un
  Space HF public).
- Validation de la CSP sans `'unsafe-inline'`.
- Validation de la sandbox `validated_path` / `compute_workspace_roots`
  par un attaquant compétent.

**État actuel** : Sprint A14-S1 a comblé les 6 P0 connus mais
l'absence d'audit externe nous interdit d'affirmer l'absence d'autres
vecteurs.

---

## 2. Features attendues mais reportées

### 2.1 Features fonctionnelles

- Reprise de benchmark hashée par contenu+config (pas seulement par
  `corpus_name + engine_name`).
- Backpressure réelle dans le runner (limite de futures en vol,
  timeout depuis le début d'exécution réelle).
- Annulation propre qui tue les workers OCR/LLM en cours
  (actuellement `cancel_futures` ne ferme pas un Tesseract en train
  de tourner).
- ZIP upload qui préserve l'arborescence (sans flatten qui écrase).
- Détection des paires `(image, GT)` qui supporte tous les patterns
  réels (`.gt.alto.xml`, `.alto.xml`, `.page.xml`, etc.).

→ Couverts par les Sprints S8, S9, S20 du rewrite ciblé.

### 2.2 Vues d'évaluation explicites

- `TextView` — la vue qui projette toute sortie textuelle vers du
  texte brut comparable.
- `AltoView` — fidélité documentaire ALTO/PAGE.
- `SearchView` — recherchabilité fuzzy plein-texte.
- `LayoutView` — coordonnées et ordre de lecture.
- `HallucinationView` — contrôle d'invention par le modèle.
- `CostView` — coût/temps/CO₂.

→ Sprints S13–S18 du rewrite. Au minimum les 3 premières doivent
exister à la livraison BnF.

### 2.3 Couche service applicative

- `app/services/benchmark_service.py` — orchestration séparée des
  routers FastAPI.
- `app/services/path_security.py` — `WorkspaceManager` qui crée un
  dossier isolé par session/run.
- Schemas DTO (Pydantic) séparés des modèles de domaine.

→ Sprint S19 du rewrite.

### 2.4 Suppression de la dette d'imports magiques

- Plus de `import picarones.evaluation.metrics as _trigger_metric_registration`
  dans `picarones/__init__.py`.
- Registres construits explicitement par un service au démarrage.
- Entry points Python pour les modules tiers (`picarones.metrics`,
  `picarones.adapters`).

→ Sprint S5 + S20 du rewrite.

### 2.5b Migration des adapters restants

Le Sprint S11 a migré 5 LLM (base + openai/mistral/anthropic/ollama)
+ 2 corpus importers (htr_united, huggingface) + 1 helper privé
(_fallback_log).  L'ancien emplacement est un re-export.

**Adapters OCR** (5 fichiers : tesseract, pero_ocr, mistral_ocr,
google_vision, azure_doc_intel) restent dans `picarones/adapters/ocr/`.
Tous importent `engines/base.py` qui hérite de `core.modules.BaseModule`.
Migration différée jusqu'au S20 quand `core.modules` aura disparu
(remplacé par le protocole `StepExecutor` du S6).

**Importers patrimoniaux** (3 fichiers : iiif, gallica, escriptorium)
restent dans `picarones/evaluation/metrics/importers/`.  Tous importent
`core.corpus.{Corpus, Document}`.  Migration différée jusqu'au
déplacement de `core.corpus` vers `domain/` (sprint dédié).

### 2.5c Migration des fichiers `measurements/*.py` restants vers `evaluation/metrics/`

Le Sprint S10 a migré 23 fichiers de calcul autonomes.  17 fichiers
restent dans `picarones/evaluation/metrics/` à migrer.

**Catégorie B — utilisent `@register_metric`** (singleton global
`core.metric_registry` à supprimer au S20) :
  `mufi`, `abbreviations`, `unicode_blocks`, `roman_numerals`,
  `early_modern_typography`, `modern_archives`, `reading_order`,
  `ner`, `readability`, `searchability`, `numerical_sequences`.

→ Migrés au S20 quand le `MetricRegistry` instancié explicitement
(S5) deviendra le seul registre.

**Catégorie C — dépendances vers `core.corpus` / `engines.base` /
`measurements.metrics`** :
  `robustness`.

→ Migré après S11 (déplacement des adapters) et S12 (équivalence
numérique).

**Catégorie D — dépendances inter-fichiers à orchestrer** :
  `cost_projection` (→ pricing, déjà migré),
  `equivalence_profile` (→ formats.text.normalization, déjà migré),
  `specialization` (→ inter_engine, déjà migré),
  `taxonomy_intra_doc` (→ taxonomy),
  `taxonomy` (→ char_scores).

→ Trois de ces fichiers (cost_projection, equivalence_profile,
specialization) peuvent être migrés dès le S11+ puisque leurs deps
sont déjà migrées.

**Fichiers d'orchestration legacy** (à NE PAS migrer en l'état,
remplacés par `pipeline/executor` + `pipeline/runner` au S22) :
  `runner/` (sous-package), `pipeline_benchmark`,
  `pipeline_comparison`, `pipeline_spec_loader`,
  `builtin_hooks`, `builtin_metrics`, `philological_hooks`,
  `readability_hooks`, `searchability_hooks`,
  `numerical_sequences_hooks`, `ner_backends`,
  `metrics`, `history`, `structure`, `difficulty`,
  `char_scores`, `alto_metrics`, `narrative/`, `statistics/`.

### 2.5 Suppression des références "Sprint X" dans le code

Le repo contient ~679 références à "Sprint N" dans les fichiers
Python (commentaires, docstrings, justifications de seuils
éditoriaux). C'est de la stratigraphie archéologique qui rend le
code illisible pour un nouveau contributeur.

→ Nettoyage progressif au fil des Sprints S10–S22 du rewrite (à
chaque déplacement de fichier, on supprime les commentaires de
sprint qui n'apportent plus rien à un lecteur de la version
courante). Pas un sprint dédié.

---

## 3. Idées qui ressortent mais qu'on ne traite pas

À valider après la livraison.

- Cache d'artefacts intermédiaires côté pipeline executor.
- Parallélisation inter-étapes au sein d'une même pipeline.
- Vue HTML drag-and-drop pour composer un pipeline (le DAG render
  Sprint 95 est de l'inspection, pas de la construction).
- Score composite personnel persisté côté serveur (pour l'instant
  uniquement URL state côté client).
- Plugin system PyPI pour modules contribués (`picarones-module-X`).
- Extension corpus levels au-delà de TEXT/ALTO/PAGE/ENTITIES/READING_ORDER
  (par exemple : tableaux, mathématiques, partitions).

---

## 4. Convention d'usage de ce document

- **Ajouter** un item dès qu'on identifie une promesse / feature qui
  doit attendre.
- **Ne pas retirer** un item juste parce qu'on a envie de le faire ;
  attendre que le rewrite l'absorbe officiellement (auquel cas il
  apparaîtra dans `docs/roadmap/rewrite-2026.md`).
- **Référencer** ce fichier dans les PRs qui retirent du scope du
  README ou de la documentation utilisateur.

Dernière revue : Sprint A14-S2 (rewrite ciblé, étape 0).
