"""Vues HTML thématiques — orchestrateurs des renderers du rapport.

Vues
----
- :mod:`economics`         — throughput effectif + (cost projection si fourni)
- :mod:`advanced_taxonomy` — taxonomy_comparison + cooccurrence + intra_doc + lexical_modernization
- :mod:`diagnostics`       — levers + image_predictive + baseline + longitudinal + multirun_stability + worst_lines
- :mod:`pipeline`          — pipeline_dag + error_absorption + incremental_comparison + module_audit
- :mod:`robustness`        — robustness_projection (workflow CLI séparé)

Convention API
--------------
Chaque vue expose une fonction publique
``build_<name>_view_html(report_data, labels, **opts) -> str`` qui :

1. **Prend** ``report_data`` (dict construit par le generator),
   ``labels`` (i18n) et des options spécifiques à la vue (ex. fixtures
   externes que l'utilisateur peut fournir).
2. **Calcule** les données dont chaque renderer a besoin à partir de
   ``report_data`` quand c'est possible.
3. **Compose** le HTML des sous-renderers en blocs ``<details>``
   collapsibles (premier ouvert par défaut).
4. **Retourne** la chaîne HTML complète, ou ``""`` si aucune
   sous-section n'a de contenu (adaptive masking corpus-wide).

Ne pas confondre
----------------
``views/<name>.py`` = orchestrateur (composition + adaptive masking).
``renderers/<name>.py`` = rendu HTML d'un seul bloc atomique.
Les renderers atomiques restent inchangés, l'orchestrateur les
combine.
"""

from picarones.reports.html.views.advanced_taxonomy import build_advanced_taxonomy_view_html
from picarones.reports.html.views.diagnostics import build_diagnostics_view_html
from picarones.reports.html.views.economics import build_economics_view_html
from picarones.reports.html.views.pipeline import build_pipeline_view_html
from picarones.reports.html.views.robustness import build_robustness_view_html

__all__ = [
    "build_advanced_taxonomy_view_html",
    "build_diagnostics_view_html",
    "build_economics_view_html",
    "build_pipeline_view_html",
    "build_robustness_view_html",
]
