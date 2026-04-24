# Études de cas Picarones

Ce répertoire rassemble des **études de cas documentées** illustrant
comment des équipes ont utilisé Picarones pour évaluer des moteurs OCR/HTR
sur des corpus patrimoniaux.

## Statut des études

- 🎓 **Cas d'école** — scénarios illustratifs construits à des fins
  pédagogiques. Les corpus et résultats sont fictifs mais réalistes.
  Les commentaires d'analyse reflètent les pratiques actuelles
  documentées dans la littérature.
- 🏛 **Cas réel** — co-rédigé avec une institution patrimoniale, avec son
  accord, sur la base d'un benchmark réel. **Aucun cas réel publié à ce
  jour** — Picarones est en recherche de premiers partenariats.

## Liste

| Étude | Statut | Corpus | Question |
|-------|--------|--------|----------|
| [01 — Registres paroissiaux XVIIᵉ-XVIIIᵉ](01-registres-paroissiaux.md) | 🎓 Cas d'école | 200 pages d'actes BMS | Quel moteur pour une indexation plein-texte de masse à budget contraint ? |
| [02 — Édition critique d'un manuscrit médiéval unique](02-edition-critique.md) | 🎓 Cas d'école | 180 folios d'un manuscrit unique | Quel pipeline pour une édition diplomatique stricte ? |

## Contribuer une étude

Picarones cherche à publier des études de cas réelles. Si votre
institution a utilisé Picarones pour un projet de numérisation et que
vous accepteriez de co-rédiger une étude (anonymisable), ouvrez une
issue GitHub avec le label `case-study`.

## Format type d'une étude

Chaque étude suit la structure :

1. **Contexte** — institution, projet, corpus (taille, langue, période,
   conventions de transcription, état physique).
2. **Question** — ce que l'équipe cherchait à décider.
3. **Métriques regardées en priorité** — et pourquoi (lien vers le
   glossaire embarqué dans le rapport).
4. **Verdict** — moteur(s) retenu(s) et argument décisif.
5. **Limites** — ce que cette évaluation ne dit pas.
6. **Reproductibilité** — paramètres exacts du benchmark, fichier
   `picarones-config.yml` si applicable.
