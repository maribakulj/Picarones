# Comment lire un rapport Picarones

Ce guide est destiné aux utilisateurs qui ouvrent un rapport HTML
Picarones pour la première fois — archivistes, conservateurs, chercheurs
en humanités numériques, prestataires de numérisation.

> **Principe directeur du rapport** : il présente les faits, jamais un
> verdict. Aucun moteur n'est étiqueté « recommandé ». L'utilisateur lit
> et décide ; l'outil l'équipe pour décider.

## Anatomie du rapport

Un rapport Picarones est un fichier HTML autonome (~450 Ko, Chart.js
embarqué). Il s'ouvre dans n'importe quel navigateur récent, sans
serveur ni connexion réseau. Vous pouvez l'envoyer par e-mail,
l'archiver, ou le déposer sur Zenodo (export DOI prévu en Phase 4).

Il s'organise en cinq sections, du général vers le spécifique :

### En-tête — synthèse + diagnostic statistique

Visible dès l'ouverture, sans navigation. Contient :

1. **Synthèse factuelle** — 3 à 5 phrases générées mécaniquement à
   partir des résultats. Aucun LLM dans la chaîne, donc le texte est
   reproductible bit-à-bit. Chaque nombre cité est traçable au JSON
   de résultats. Voir [docs/developer/narrative-engine.md] pour la liste
   complète des faits que le moteur peut détecter.
2. **Critical Difference Diagram** (Friedman-Nemenyi) — un graphique
   horizontal qui place chaque moteur sur un axe de rang moyen. Les
   barres horizontales relient les moteurs **statistiquement
   indiscernables** au seuil α = 0,05. Si deux moteurs sont reliés, leur
   différence de CER moyen n'est pas significative ; le choix entre eux
   doit se faire sur d'autres critères (coût, vitesse, robustesse, profil
   d'erreur).

### Vue Classement

Tableau triable par colonne. Chaque en-tête a un petit `?` cliquable
qui ouvre le **glossaire contextuel** (définition, ce que la métrique
mesure, cas d'usage historique, limites, référence bibliographique).

Colonnes principales :
- **CER exact** vs **CER diplomatique** : le diplomatique fusionne
  `ſ=s`, `u=v`, `i=j`, etc. — utile pour les corpus pré-XIXᵉ.
- **WER, MER, WIL** : variantes mot-à-mot.
- **Ligatures, Diacritiques** : pertinent pour les corpus patrimoniaux.
- **Gini** : concentration des erreurs dans le document. Un Gini élevé
  signale qu'une petite fraction des lignes concentre la majorité des
  erreurs.
- **Ancrage trigrammes** : pour LLM/VLM. Un score bas signale des
  hallucinations probables.

### Vue Galerie / Document

Inspection page par page, avec diff coloré GT/OCR par moteur.
Particulièrement utile pour les pipelines OCR+LLM : vous pouvez voir
**les trois couches** (image, OCR brut, correction LLM) et identifier
les zones où le LLM a apporté de la valeur ou, au contraire, halluciné.

### Vue Caractères

Matrice de confusion Unicode + taxonomie des erreurs en 9 classes
(confusion visuelle, diacritique, casse, ligature, abréviation, hapax,
segmentation, OOV, lacune). Utile pour identifier les **patterns
systématiques** d'un moteur — par exemple : « ce moteur fait 40 %
d'erreurs d'abréviation, donc il ne saura pas lire mon corpus notarié ».

### Vue Analyses

Graphiques Chart.js avancés :
- **Vue Pareto qualité/coût/vitesse/carbone**. Les moteurs sur la
  frontière de Pareto (en vert) sont ceux pour lesquels aucun autre
  n'offre simultanément un meilleur CER ET un meilleur coût. Choisir
  hors du front est toujours sous-optimal ; choisir sur le front
  dépend de vos priorités. Les prix sont indicatifs et datés — voir
  les hypothèses détaillées sous le graphique.
- Histogrammes, radar, scatter plots, courbes de fiabilité.
- Bootstrap CI, tests de Wilcoxon pairwise, clustering d'erreurs,
  matrice de corrélation.

## Le bouton « ⚙ Avancé »

Dans la nav (en haut à droite). Ouvre un panneau latéral avec :

- **Choix de colonnes visibles** — masquez les colonnes qui ne vous
  intéressent pas pour réduire la charge cognitive.
- **Filtres par strate** — quand le corpus expose `script_type` par
  document, vous pouvez filtrer (ex. ne voir que les documents en
  textualis).
- **Score composite personnel** — strictement opt-in. Tous les curseurs
  démarrent à 0. Quand vous ajustez les poids, la formule du score
  s'affiche en clair sous les curseurs et une nouvelle colonne « Score »
  apparaît dans le tableau.

> ⚠️ **Avertissement explicite affiché dans le panneau** :
> « Ces poids reflètent votre cas d'usage. Il n'existe pas de pondération
> universellement valide — Picarones ne suggère aucune pondération par
> défaut. »
>
> Picarones ne fait pas de recommandation à votre place. Le score
> composite est un outil que vous construisez vous-même.

L'état de personnalisation est persisté dans l'URL (`?hidden=…&w=…`),
donc vous pouvez partager une vue avec votre équipe en envoyant le lien.

## Le mode « ⊞ Présentation »

Masque les éléments techniques (boutons `?`, mode avancé, hints) pour
projeter le rapport en réunion ou pour l'imprimer. Les nombres et
graphiques restent identiques.

## Export CSV

Bouton « ⬇ CSV » dans la nav. Exporte la vue courante (avec les filtres
de personnalisation appliqués) en CSV pour réutilisation dans Excel ou
LibreOffice.

## Que faire si vous ne savez pas par où commencer ?

1. Lisez la **synthèse en tête** (3 à 5 phrases) — elle pointe les faits
   saillants.
2. Regardez le **CDD** : si tous les moteurs sont reliés par une seule
   barre horizontale, votre corpus n'est pas assez discriminant pour
   trancher sur la base statistique seule. Cherchez d'autres critères
   (coût, profil d'erreur).
3. Ouvrez la **vue Pareto** : si un moteur du front Pareto est très
   moins cher que le leader CER, c'est probablement votre meilleur
   compromis pratique.
4. Consultez les **études de cas** dans `docs/case-studies/` pour voir
   comment d'autres équipes ont raisonné sur des problèmes similaires.

## Pour aller plus loin

- [Glossaire complet] (intégré dans le rapport, accessible via les `?`)
- [docs/developer/narrative-engine.md] — comment ajouter un détecteur
- [docs/developer/extending-glossary.md] — comment enrichir le glossaire
- [SPECS.md] — spécifications complètes du projet
