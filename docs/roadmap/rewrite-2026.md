# Rewrite ciblé — plan S1 → S26

> **Statut** — démarré au Sprint A14-S1 (mai 2026), livraison cible
> **fin 2026** sur la branche `claude/repo-analysis-cukvm` puis fusion
> sur `main` pour livraison BnF.
>
> **Doctrine** : pas de Big Rewrite. Pas non plus de migration douce
> qui laisserait la dette en place. **Rewrite ciblé** : on réécrit
> from scratch les zones cassées (~5–8 k lignes : runner d'orchestration,
> couche web sécurité, gestion d'artefacts) et on **déplace** les zones
> saines (~30–40 k lignes : calculs purs MUFI / philological /
> statistics / etc.) sans toucher à leur logique.

---

## Pourquoi un rewrite ciblé ?

Trois constats issus de l'audit (`docs/audits/`) et de la conversation
de cadrage de mai 2026 :

1. **Les promesses du README dépassaient la réalité du code.** Six bugs
   P0 vérifiés dans l'audit invalidaient la promesse scientifique
   (notamment : `normalization_profile` côté web silencieusement
   ignoré, `compact()` qui amputait le JSON exporté, `compute_metrics`
   qui retournait `0.0` indistinguable d'un score parfait en cas
   d'erreur).
2. **L'architecture à imports magiques.** `import picarones`
   déclenche une chaîne d'imports par effet de bord qui charge le
   registre de métriques. Une dépendance optionnelle manquante au fond
   de la chaîne fait crasher l'import du package entier.
3. **La dette narrative est trop lourde.** ~679 références à
   "Sprint N" dans les fichiers Python, qui parasitent la lecture du
   code par un nouveau contributeur et empêchent toute prise en main
   par un mainteneur extérieur.

Le rewrite ciblé attaque ces trois problèmes ensemble.

---

## Architecture cible

À la fin du rewrite, l'arborescence Python sera :

```
picarones/
  domain/            # Cercle 1 — types purs (Artifact, PipelineSpec,
                     #   EvaluationSpec, DocumentRef, Provenance)
  evaluation/        # Cercle 2 — vues, projecteurs, métriques
    views/
    projectors/
    metrics/
    registry.py
  pipeline/          # Cercle 2 — exécution
    executor.py
    cache.py
    spec.py
  formats/           # Cercle 2 — ALTO, PAGE, normalisation texte
    alto/
    pagexml/
    text/
  adapters/          # Cercle 3 — moteurs OCR/LLM/VLM, importers, storage
    ocr/
    llm/
    vlm/
    corpus/
    storage/
  app/               # Cercle 4 — services applicatifs
    services/
    schemas/
  interfaces/        # Cercle 5 — CLI, web, reports
    cli/
    web/
  reports/
    html/
    json/
    csv/
```

Pivot mental : l'objet central n'est plus `Engine + BenchmarkResult`,
c'est `Pipeline → Artifacts → Projection → EvaluationView → Metrics`.

---

## Calendrier (26 semaines)

### Phase 0 — Stabilisation de l'existant (S1 → S2)

| Sprint | Objectif | État |
|---|---|---|
| **S1** | Boucher les 6 P0 sur `main` | ✅ Livré (commit `a2bea75`) |
| **S2** | Recadrer le README, env propre, BACKLOG_POST_LIVRAISON | ⏳ En cours |

À la fin de S2, l'outil actuel reste utilisable pour les tests BnF
pendant que le rewrite avance sur `rewrite-2026`.

### Phase 1 — Squelette et règles d'architecture (S3 → S6)

| Sprint | Objectif |
|---|---|
| S3 | Créer les répertoires cibles + tests d'architecture qui interdisent le retour en arrière |
| S4 | Modèle `Artifact` et types fondamentaux dans `domain/` |
| S5 | `EvaluationView`, `EvaluationSpec`, `MetricSpec` typés |
| S6 | `PipelineSpec`, `PipelineStep`, contrats d'exécution |

Critère go/no-go fin de Phase 1 : les tests d'architecture passent,
la BnF continue à utiliser `main`.

### Phase 2 — Pipeline executor et migration des calculs (S7 → S12)

| Sprint | Objectif |
|---|---|
| S7 | Pipeline executor v1 (séquentiel mono-document) |
| S8 | Backpressure + timeout réel + annulation propre |
| S9 | `formats/alto/` et `formats/pagexml/` |
| S10 | Migration des calculs purs vers `evaluation/metrics/` (gros sprint) |
| S11 | Migration des adapters dans `adapters/` |
| S12 | Le nouvel executor reproduit l'ancien runner numériquement |

Critère go/no-go fin de Phase 2 : équivalence CER/WER vérifiée à
1e-9 près sur 5 fixtures + 1 corpus BnF réel.

### Phase 3 — Vues d'évaluation (S13 → S18) — cœur de la valeur ajoutée

| Sprint | Objectif |
|---|---|
| S13 | `EvaluationViewExecutor` et le moteur de vues |
| S14 | `TextView` (vue canonique 1) |
| S15 | `AltoView` (vue canonique 2) |
| S16 | `SearchView` (vue canonique 3) + cohérence inter-vues |
| S17 | Intégration runner + vues + nouveau format de résultat |
| S18 | E2E sur le cas BnF central + recettage interne |

Critère go/no-go fin de Phase 3 : ton cas d'usage central
(Tesseract texte brut vs OCR+LLM+ALTO remappé vs VLM+ALTO reconstruit)
fonctionne bout-en-bout, lisible, avec rapports de projection
explicites.

### Phase 4 — Web sandboxée + recettage (S19 → S24)

| Sprint | Objectif |
|---|---|
| S19 | Couche `app/services/` |
| S20 | Réécriture corpus upload + sandbox ZIP |
| S21 | Nouveau `interfaces/web/` (CSRF on, CSP sans inline) |
| S22 | `interfaces/cli/` + `reports/html/` migration |
| S23 | Recettage BnF complet |
| S24 | Corrections de recettage + documentation finale |

### Buffer (S25 → S26)

Imprévus + livraison. Ces deux semaines sont **non négociables**.

---

## Discipline du rewrite

Quatre invariants permanents, valables pendant les 26 semaines :

1. **`main` reste livrable.** Le rewrite vit sur `rewrite-2026` /
   `claude/repo-analysis-cukvm`. Les P0 vont sur `main`.
2. **Pas de feature nouvelle.** Si l'envie vient, écrire dans
   [`BACKLOG_POST_LIVRAISON.md`](../../BACKLOG_POST_LIVRAISON.md) et
   passer.
3. **Fin de chaque sprint = un commit qui passe `pytest tests/ -q`.**
4. **Chaque sprint a un livrable démontrable** en 5 minutes.

Pour le détail à la semaine de chaque sprint (livrables, tests,
définition de "done", risque principal), voir le plan complet livré
en réponse à la question de cadrage du 2026-05-03 dans la session
[`session_011XQZNitg1rCgia8ZD1a2hP`](https://claude.ai/code/session_011XQZNitg1rCgia8ZD1a2hP).

---

## Ce qui n'est *pas* dans le rewrite

Cf. [`BACKLOG_POST_LIVRAISON.md`](../../BACKLOG_POST_LIVRAISON.md) pour
la liste complète. En résumé :

- Pas de feature nouvelle (NER cloud, VLM extras, etc.).
- Pas de promesses institutionnelles (RGPD opérationnel, JOSS, COI
  exercés).
- Pas de réécriture des calculs purs (MUFI, philological, statistics)
  — on les déplace, point.
- Pas de refonte du rapport HTML au-delà de l'intégration des vues
  (le rendu visuel reste celui d'aujourd'hui pour ne pas allonger).
