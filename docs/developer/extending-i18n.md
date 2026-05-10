# Étendre l'internationalisation (i18n)

Picarones a deux systèmes i18n distincts, qui partagent la même
convention de fichiers JSON par langue :

| Système | Fichier | Contenu |
|---------|---------|---------|
| Interface du rapport HTML | `picarones/reports/html/i18n/{lang}.json` | Libellés des onglets, colonnes, boutons, messages dynamiques |
| Glossaire contextuel       | `picarones/reports/html/glossary/{lang}.yaml` | Définitions des métriques |
| Templates narratifs        | `picarones/domain/narrative/templates/{lang}.yaml` | Phrases de la synthèse factuelle |

## Ajouter une nouvelle clé d'interface

1. Ajoutez la clé dans `picarones/reports/html/i18n/fr.json` ET
   `picarones/reports/html/i18n/en.json` (le test `test_fr_and_en_have_same_keys`
   casse sinon).
2. Côté HTML, utilisez l'attribut `data-i18n="ma_nouvelle_cle"`. Le
   contenu littéral du HTML est le **fallback** ; il est remplacé au
   chargement par `applyI18n()` dans `_app.js`.
3. Côté JS, `I18N.ma_nouvelle_cle` retourne la traduction.

```html
<button data-i18n="my_new_button">Texte par défaut français</button>
```

```javascript
const label = I18N.my_new_button || 'Texte par défaut français';
```

## Ajouter une nouvelle langue

1. Créez `picarones/reports/html/i18n/de.json` (copiez `fr.json` et traduisez).
2. Créez `picarones/reports/html/glossary/de.yaml` (copiez `fr.yaml` et
   traduisez).
3. Créez `picarones/domain/narrative/templates/de.yaml` (copiez `fr.yaml`).
4. Lancez le rapport en spécifiant la langue : `picarones report
   --json results.json --output rapport.html --lang de`.

Les loaders (`get_labels`, `load_glossary`, `_load_templates`) tombent
automatiquement sur `fr` si une langue manque.

## Tests à mettre à jour

- `test_sprint17_jinja2_refactor.py::TestI18nFromJSON::test_fr_and_en_have_same_keys`
  vérifie la symétrie. Pour 3 langues, étendre.
- `test_sprint21_glossary_customize.py::TestGlossaryCompleteness::test_fr_and_en_have_same_keys`
  vérifie la symétrie du glossaire.

## Format YAML pour les templates narratifs

Voir `docs/explanation/narrative-engine.md` pour le détail. En bref :

```yaml
fact_type_value: >-
  Phrase avec des {placeholders} qui correspondent aux clés du
  payload du Fact détecté.
```

Le rendu utilise `str.format_map`, pas Jinja2 — pour empêcher
toute génération arbitraire de contenu (anti-hallucination).

## Référence

| Sprint | Travail i18n |
|---|---|
| 11 | Création du système FR/EN |
| 17 | Migration `i18n.py` (dict Python) → `i18n/{fr,en}.json` |
| 18 | +6 clés CDD (`cdd_*`) |
| 19 | +2 clés synthèse (`synth_*`) + 10 templates narratifs par langue |
| 20 | +9 clés Pareto (`pareto_*`) |
| 21 | +19 clés glossaire/personnalisation + 25 entrées glossaire par langue |
