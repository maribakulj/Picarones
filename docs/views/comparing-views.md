# Lire les 3 vues canoniques ensemble

Sprint A14-S16 livre la troisième vue canonique du rewrite ciblé :
`SearchView`.  Avec `TextView` (S14) et `AltoView` (S15), on a
maintenant **trois lentilles complémentaires** pour évaluer un
même pipeline.

## Le tableau des 3 vues

| Vue | Question | Métriques | Direction |
|---|---|---|---|
| **TextView** (S14) | Quel pipeline produit le meilleur **texte final** ? | CER, WER, MER, WIL | `lower_is_better` (erreurs) |
| **AltoView** (S15) | Quel pipeline produit le meilleur **ALTO exploitable** ? | alto_validity, line_count_ratio, word_box_coverage | `higher_is_better` (qualité) |
| **SearchView** (S16) | Quel pipeline maximise la **recherchabilité plein-texte** ? | searchability_recall, numerical_sequence_preservation | `higher_is_better` (rappel) |

Aucune des trois vues ne dit toute la vérité sur un pipeline.
**Ensemble, elles racontent l'histoire complète.**

## Pourquoi les trois vues sont nécessaires

Un même pipeline peut être **excellent dans une vue et médiocre
dans une autre**.  C'est précisément ce qui rend la comparaison
hétérogène utile pour la BnF — un seul score (CER global)
masquerait des informations critiques.

### Pattern 1 : CER excellent, recherchabilité numérique catastrophique

Démontré dans le test
`tests/evaluation/test_sprint_a14_s16_views_consistency.py::TestDivergencePattern::test_year_corruption_invisible_to_cer_visible_to_search` :

- **GT** : *"Charte signée à Paris le 14 juillet 1789 en présence du roi"*
- **Hypothèse** : *"Charte signée à Paris le 14 juillet 1798 en présence du roi"*

Le LLM de post-correction a "amélioré" la date (1789 → 1798).
Conséquences :

| Vue | Métrique | Valeur | Lecture |
|---|---|---|---|
| TextView | CER | ~0.03 | Excellent (3 chars sur 58) |
| TextView | WER | ~0.09 | Très bon (1 mot sur 11) |
| SearchView | searchability_recall | ~0.91 | Bon (1798 fuzzy match 1789) |
| SearchView | **numerical_sequence_preservation** | **0.0** | **Catastrophique** |

Pour un historien qui veut indexer ses chartes par date, ce
pipeline est **inutilisable** — l'année 1789 est silencieusement
réécrite en 1798.  Le CER ne le révèle pas.  `SearchView` le
révèle.

### Pattern 2 : Texte parfait, ALTO inexistant

Un OCR Tesseract qui ne produit que du texte brut :

| Vue | Statut | Lecture |
|---|---|---|
| TextView | CER = 0.0 | Pipeline parfait pour la lecture |
| SearchView | recall = 1.0 | Pipeline parfait pour l'indexation |
| **AltoView** | **OMIS** | Pipeline non éligible |

Pour un workflow IIIF / Mirador qui veut surligner les mots dans
l'image, ce pipeline est **inutilisable** — pas de coordonnées.
`AltoView` ne lui attribue pas un score factice à 0 ; le rapport
affiche *"Tesseract texte brut n'est pas évalué dans AltoView
(ne produit pas d'ALTO)"*.

### Pattern 3 : ALTO valide mais texte hallucinant

Un VLM avec module ALTO_reconstruction peut produire un ALTO
structurellement parfait (validity=1, lignes correctes,
coordonnées présentes) mais avec du texte inventé :

| Vue | Métrique | Valeur | Lecture |
|---|---|---|---|
| AltoView | tous | 1.0 | Pipeline parfait structurellement |
| TextView | CER | élevé | Pipeline mauvais textuellement |
| SearchView | recall | bas | Pipeline inutile pour la recherche |

`AltoView` seul ferait passer ce VLM pour le meilleur pipeline.
Lire les trois vues ensemble révèle le vrai problème.

## Recommandation de lecture pour le rapport BnF

Le rapport HTML (S22) présentera les 3 vues côte-à-côte avec
cette grille de lecture :

1. **Tableau de synthèse** : un tableau par vue, chaque ligne =
   un pipeline, chaque colonne = une métrique.  Les pipelines
   omis sont indiqués explicitement (pas de valeur factice).

2. **Encart "divergences notables"** : signale automatiquement
   les pipelines dont le rang change fortement entre vues
   (par exemple "rang 1 en TextView, rang 5 en SearchView").
   C'est un signal pour l'utilisateur d'aller regarder en
   détail ce qui se passe.

3. **Pour chaque vue** : warnings explicites de ce qu'elle
   **n'évalue pas** (cf. `ignored_dimensions` dans chaque
   `ViewResult`).  L'utilisateur ne peut pas conclure
   "TextView dit que X est le meilleur" sans avoir vu ce que
   `TextView.ignored_dimensions` ne dit PAS.

## Critères de choix selon l'usage

| Usage cible | Vue principale | Vues secondaires |
|---|---|---|
| Lecture humaine (édition critique) | TextView | AltoView (si édition diplomatique) |
| Indexation Elastic / Solr / Gallica | SearchView | TextView |
| Réinjection IIIF / Mirador (mots cliquables) | AltoView | TextView |
| Citation académique | TextView + SearchView | AltoView |
| Reproduction d'un fac-similé | AltoView | TextView |

## Statut

- ✅ Sprint S14 — `TextView`
- ✅ Sprint S15 — `AltoView`
- ✅ Sprint S16 — `SearchView` + cohérence inter-vues
- ⏳ Sprint S17 — intégration runner + RunManifest
- ⏳ Sprint S18 — tests E2E sur le cas BnF central
