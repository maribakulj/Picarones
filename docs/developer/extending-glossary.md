# Étendre le glossaire contextuel

Le glossaire affiché dans le panneau latéral du rapport est défini en
YAML, une entrée par terme, dans `picarones/reports/html/glossary/{lang}.yaml`.

## Ajouter un terme

Éditez **les deux fichiers** `fr.yaml` et `en.yaml` (les tests vérifient
la symétrie) :

```yaml
my_new_metric:
  title: "Titre court — abréviation"
  definition: >-
    2-3 phrases de définition formelle. Tout en pur texte, pas de HTML.
  measures: >-
    Ce que la métrique mesure concrètement.
  usage: >-
    Cas d'usage factuels (pas prescriptifs : « utilisé en X » et non
    « à choisir si vous êtes Y »).
  limits: >-
    Limites connues, pièges fréquents.
  reference: >-
    Référence bibliographique canonique (auteur, année, titre).
```

Tous les champs (`title`, `definition`, `measures`, `usage`, `limits`,
`reference`) sont **obligatoires** et non vides. Les tests
(`test_each_entry_has_required_fields`) le vérifient.

## Brancher l'entrée à une colonne du rapport

Dans `picarones/reports/html/templates/view_ranking.html` (ou un autre fichier
de vue), ajoutez l'attribut `data-glossary-key` sur l'en-tête de
colonne :

```html
<th data-col="my_new_metric" class="sortable"
    data-glossary-key="my_new_metric"
    data-i18n="col_my_new_metric">Ma métrique</th>
```

Au démarrage du rapport, `injectGlossaryButtons()` parcourt tous les
`th[data-glossary-key]` et ajoute un petit bouton `?` cliquable.

## Règles à respecter

1. **Pas de HTML dans le contenu** : le rendu côté JS utilise
   `textContent` pour les paragraphes (sécurité XSS). Si vous mettez
   du HTML, il sera affiché en clair.
2. **Pas de prescription** : le champ `usage` doit être factuel
   (« historiquement utilisé pour X ») et non prescriptif (« à choisir
   si vous voulez Y »). Picarones ne recommande jamais.
3. **Référence vérifiable** : citez un auteur/année réels. Les utilisateurs
   peuvent suivre la référence pour creuser.
4. **Symétrie FR/EN** : les deux fichiers doivent contenir les mêmes
   clés. Le test `test_fr_and_en_have_same_keys` casse en cas
   d'asymétrie.

## Ajouter une langue

Créez `picarones/reports/html/glossary/de.yaml` avec la même structure que
`fr.yaml`. Le loader le détecte automatiquement via
`Path.glob("*.yaml")`.

Pour qu'un utilisateur puisse choisir cette langue :

1. Ajouter `picarones/reports/html/i18n/de.json` (traductions de l'interface).
2. Aucune modification de code requise — `load_glossary("de")` marchera.

## Tests

Le fichier `tests/test_glossary_customize.py` couvre
automatiquement :

- Chargement et fallback (langue inconnue → FR).
- Symétrie FR/EN.
- Présence des champs obligatoires.
- Termes critiques garantis (16 entrées dont CER, WER, Friedman, etc.).
- Pas de HTML dans le contenu.

Ajoutez votre nouvelle entrée à la liste `critical` du test
`test_critical_terms_are_documented` si elle est référencée par une
colonne du rapport — cela garantit qu'un futur refactor ne la
supprime pas par accident.
