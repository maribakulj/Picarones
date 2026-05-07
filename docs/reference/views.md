# Vues HTML du rapport — chantier 3 post-Sprint 97

Le rapport HTML Picarones est composé de **5 vues principales** historiques
(navigables via la nav top) et de **3 nouvelles vues thématiques** (chantier 3)
qui s'affichent comme cartes dans la vue `analyses`.

Toutes les vues sont **adaptive** : une vue/section ne s'affiche que si elle
a du signal à montrer (au moins une sous-section avec données exploitables).

## Vues historiques (navigation principale)

| Vue | Template | Contenu |
|---|---|---|
| `ranking` | `view_ranking.html` | Classement moteurs (CER/WER médian/moyen, stratifié) |
| `gallery` | `view_gallery.html` | Galerie d'images du corpus |
| `document` | `view_document.html` | Vue détaillée par document (diff GT vs hyp) |
| `analyses` | `view_analyses.html` | Tableau de bord analytique (graphes Chart.js) |
| `characters` | `view_characters.html` | Analyse caractère par caractère (confusion) |

## Sous-sections de `view_analyses.html`

Cette vue agrège **tous les renderers spécialisés** sous forme de cartes
pleine largeur, avec un patron commun :

```jinja2
{% if some_html %}
<div class="chart-card" style="grid-column:1/-1">
  {{ some_html }}
</div>
{% endif %}
```

Si `some_html` est `""` (adaptive masking parce qu'aucune donnée), la
carte n'apparaît pas.

### Sous-sections historiques (Sprints 5+10+39+42+55-60+86+87+89)

| Bloc | Origine | Données nécessaires |
|---|---|---|
| Distribution CER | Sprint 2 | toujours |
| Radar profil moteur | Sprint 2 | ≥ 1 moteur |
| CER par document | Sprint 2 | toujours |
| Temps d'exécution | Sprint 2 | durations propagées |
| Qualité image ↔ CER | Sprint 5 | `aggregated_image_quality` |
| Taxonomie | Sprint 5 | `aggregated_taxonomy` |
| Courbes de fiabilité | Sprint 7 | `aggregated_calibration` |
| NER (P/R/F1) | Sprint 41 | `aggregated_ner` (opt-in spaCy) |
| Calibration ECE/MCE | Sprint 43 | `aggregated_calibration` |
| Stratification | Sprint 46 | `script_type` par doc |
| Profil philologique | Sprint 62 | `aggregated_philological` |
| Recherchabilité fuzzy | Sprint 86 | `aggregated_searchability` |
| Séquences numériques | Sprint 86 | `aggregated_numerical_sequences` |
| Lisibilité (Δ Flesch) | Sprint 87 | `aggregated_readability` |
| Spécialisation inter-moteurs | Sprint 89 | ≥ 2 moteurs avec taxonomie |
| Analyse inter-moteurs | Sprint 37 | ≥ 2 moteurs |
| Matrice de corrélation | Sprint 7 | toujours |

### Sous-sections du chantier 3 (post-Sprint 97)

3 nouvelles vues thématiques composables qui regroupent les **16 renderers
orphelins** identifiés dans l'audit initial :

#### Vue « Coût et performance » (`build_economics_view_html`)

Module : [`picarones/reports_v2/html/views/economics.py`](../picarones/reports_v2/html/views/economics.py).
Activée si :
- `engine_reports` fournis avec durations non nulles.
- (Optionnel) `extra_html_blocks` pour cost projection / marginal cost.

Sous-sections :
- **Throughput effectif** : pages/h **utilisable** (intégrant 5 s/erreur
  HTR-United), depuis `picarones.core.throughput`.

#### Vue « Taxonomie avancée » (`build_advanced_taxonomy_view_html`)

Module : [`picarones/reports_v2/html/views/advanced_taxonomy.py`](../picarones/reports_v2/html/views/advanced_taxonomy.py).
Activée si ≥ 2 moteurs ont une `aggregated_taxonomy`.

Sous-sections :
- **Comparaison miroir** : leader CER vs runner-up, diagramme miroir
  par classe d'erreur, tableau de récupérabilité éditoriale.
- **Co-occurrence de classes** (opt-in) : heatmap Jaccard inter-classes.
- **Distribution intra-document** (opt-in) : heatmap classe × position.
- **Modernisation lexicale** (opt-in) : top tokens GT modernisés.

#### Vue « Diagnostic approfondi » (`build_diagnostics_view_html`)

Module : [`picarones/reports_v2/html/views/diagnostics.py`](../picarones/reports_v2/html/views/diagnostics.py).
Activée si `detect_levers()` produit au moins un levier (typique sur
un bench standard) ou si données opt-in fournies.

Sous-sections :
- **Leviers d'amélioration** : factuels (jamais prescriptifs), depuis
  `picarones.core.levers`.
- **Comparaison historique** (opt-in) : encart « ce corpus est-il habituel ? ».
- **Profil d'image du corpus** (opt-in) : complexité paléographique +
  homogénéité.
- **Évolution longitudinale** (opt-in) : table par moteur sur historique.
- **Stabilité multi-runs** (opt-in) : variance entre runs successifs.
- **Lignes les pires** (opt-in) : top 20 lignes avec le pire CER.

## Vues spécifiques (rapport autonome)

Deux vues du chantier 3 ne s'intègrent pas au rapport classique mais
servent à composer des **rapports autonomes** :

### Vue « Pipeline composée » (`build_pipeline_view_html`)

Module : [`picarones/reports_v2/html/views/pipeline.py`](../picarones/reports_v2/html/views/pipeline.py).

Utilisée par `picarones pipeline run` (ou par tout outil qui consomme un
`PipelineBenchmarkResult`). Sous-sections :

- **Résumé pipeline** (`build_pipeline_summary_html` + `build_pipeline_steps_table_html`).
- **DAG visuel** (`pipeline_dag_render.py` Sprint 95) — opt-in.
- **Absorption d'erreur** par jonction (`error_absorption_render.py`
  Sprint 94) — opt-in.
- **Comparaison incrémentale** par slot (`incremental_comparison_render.py`
  Sprint 96) — opt-in.
- **Audit des modules** contribués (`module_audit_render.py`
  Sprint 97) — opt-in.

### Vue « Robustesse projetée » (`build_robustness_view_html`)

Module : [`picarones/reports_v2/html/views/robustness.py`](../picarones/reports_v2/html/views/robustness.py).

Utilisée par le workflow `picarones robustness`. Sous-sections :

- **Déficit projeté de robustesse** (`robustness_projection_render.py`
  Sprint 88).

## Convention de rendu partagée

Toutes les 5 vues du chantier 3 utilisent le même shell `_render_view_shell()`
défini dans `economics.py` :

- Titre `<h3>` + note explicative en tête.
- Une `<details>` collapsible par sous-renderer.
- Premier `<details>` ouvert, les autres fermés (réduit le scroll initial).
- Anti-injection HTML systématique via `xml.sax.saxutils.escape`.

## Code source

- [`picarones/reports_v2/html/generator.py`](../picarones/reports_v2/html/generator.py)
  — orchestrateur Jinja2 qui appelle les renderers et passe leurs sorties
  au template.
- [`picarones/reports_v2/html/views/`](../picarones/reports_v2/html/views/) — 5 modules de
  composition (chantier 3).
- [`picarones/reports_v2/html/renderers/`](../picarones/reports_v2/html/renderers/) — 26 renderers
  atomiques.
- [`picarones/reports_v2/html/templates/view_analyses.html`](../picarones/reports_v2/html/templates/view_analyses.html)
  — template Jinja2 qui inclut les blocs.
- [`tests/test_views.py`](../tests/test_views.py) — tests d'intégration
  des 5 vues du chantier 3.
