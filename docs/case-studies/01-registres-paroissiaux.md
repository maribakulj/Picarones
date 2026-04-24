# 01 — Registres paroissiaux XVIIᵉ-XVIIIᵉ siècle

> 🎓 **Cas d'école** — scénario illustratif. Le corpus, les chiffres et
> l'institution sont fictifs mais conçus pour être réalistes (calibrés
> sur des projets décrits dans la littérature DH).

## Contexte

| | |
|---|---|
| Institution | Service d'archives départementales d'une métropole française |
| Projet | Numérisation et indexation plein-texte de 80 000 pages de registres |
|         | de baptêmes-mariages-sépultures (BMS) du XVIIᵉ au XVIIIᵉ siècle |
| Corpus de benchmark | 200 pages échantillonnées sur 12 paroisses |
| Langue | Français pré-classique, formules latines récurrentes |
| Écriture | Mains de scribes paroissiaux, qualité variable, cursive courante |
| GT disponible | 200 pages transcrites par 3 archivistes vacataires |
| Conventions | Diplomatique strict (préservation `ſ`, `u`/`v`, abréviations) |
| Budget annuel | 15 000 € pour les coûts d'inférence sur 4 ans |

## Question

> Quel moteur (ou pipeline) retenir pour traiter les 80 000 pages dans le
> budget alloué, en privilégiant le rappel des **noms propres** (objectif
> métier : indexer les actes pour la recherche généalogique) ?

## Métriques regardées en priorité

L'équipe a ouvert le rapport Picarones et consulté **dans cet ordre** :

1. **Synthèse factuelle en tête** — pour identifier les moteurs candidats
   dans le groupe de tête statistique. Le CDD (Friedman-Nemenyi) montrait
   que `pero_ocr`, `tesseract → claude-haiku-4-5` et `tesseract` étaient
   indiscernables au seuil α = 0,05.
2. **Vue Pareto coût/qualité** (axe coût €) — pour exclure les options
   trop chères pour le budget. `gpt-4o` en zero-shot, malgré un CER
   compétitif, était à ×8 le coût de `claude-haiku-4-5`.
3. **Score de difficulté** stratifié par paroisse — pour vérifier qu'aucun
   moteur ne s'effondrait sur les paroisses aux mains les plus cursives.
   `pero_ocr` y excellait, `tesseract` seul s'effondrait.
4. **Taxonomie des erreurs (vue Caractères)** — l'équipe a regardé
   spécifiquement la classe `abbreviation_error` puisque les actes BMS
   utilisent des abréviations latines fréquentes (`obijt`, `bapt.`).
   `tesseract → claude-haiku-4-5` produisait 2× moins d'erreurs
   d'abréviation que `tesseract` seul (le LLM les développait correctement).

## Métriques **non** regardées

- Le CER global comme critère unique. L'équipe savait par expérience que
  les actes BMS sont des textes courts et formulaires : un CER de 8 %
  peut être acceptable si les noms propres et les dates sont préservés.
- Le WER : trop sensible à la segmentation, sans valeur ajoutée par
  rapport au CER pour leur usage.

## Verdict

**Pipeline retenu** : `tesseract → claude-haiku-4-5` en mode
`post_correction_texte`.

**Arguments** :
- Coût estimé : 80 000 × 0,80 €/1000 = **64 €** (budget largement
  respecté). À comparer aux 12 000 € de `gpt-4o` en zero-shot.
- CER médian : 4,2 % [3,8–4,7] (IC 95 % bootstrap), dans le groupe de
  tête statistique du CDD.
- Profil d'erreurs favorable aux noms propres.
- Robuste sur la stratification par paroisse (pas d'effondrement sur les
  paroisses aux mains difficiles).

## Limites

- L'évaluation a porté sur 200 pages, soit 0,25 % du corpus cible. Une
  validation sur 1 000 pages additionnelles est prévue après mise en
  production.
- L'IC bootstrap suppose l'indépendance entre documents — peut-être
  optimiste car les pages d'une même paroisse partagent le même scribe.
- Le coût de `claude-haiku-4-5` peut évoluer pendant les 4 ans du projet.
  L'équipe a prévu un avenant tarifaire dans la convention.
- L'empreinte carbone n'a pas été incluse comme critère décisif (mode
  expérimental dans Picarones), mais l'équipe a noté qu'un OCR cloud
  émet ~×30 plus de CO₂ qu'un OCR local côté France.

## Reproductibilité

```yaml
# picarones-config.yml
corpus: ./benchmarks/bms-200pages/
engines:
  - tesseract: { lang: fra+lat, psm: 6 }
  - pero_ocr: { model: medieval-french-2024 }
  - pipeline:
      ocr: tesseract
      llm: claude-haiku-4-5
      mode: post_correction_texte
      prompt: correction_early_modern_french.txt
  - pipeline:
      ocr: null
      llm: gpt-4o
      mode: zero_shot
      prompt: zero_shot_imprime_ancien.txt
normalization: early_modern_french
report:
  lang: fr
  output: rapport-bms.html
```

Reprise possible via `picarones run --partial-dir /tmp/picarones-bms/`.
