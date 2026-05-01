"""Vues HTML thématiques — orchestrateurs des renderers du rapport.

Chantier 3 du plan d'évolution post-Sprint 97.

Pourquoi ce package
-------------------
Avant ce chantier, ``picarones/report/`` exposait 26 modules
``*_render.py``, dont **16 étaient orphelins** : testés mais jamais
importés par ``generator.py`` ni inclus dans aucun template Jinja2.

Le chantier 3 résout ce déséquilibre **par regroupement** : chaque
renderer orphelin trouve une **adresse** dans une vue thématique,
qui est elle-même branchée conditionnellement au rapport principal
si elle a du contenu à afficher.

Vues livrées par ce chantier
----------------------------
- :mod:`economics`         — throughput effectif + (cost projection si fourni)
- :mod:`advanced_taxonomy` — taxonomy_comparison + cooccurrence + intra_doc + lexical_modernization
- :mod:`diagnostics`       — levers + image_predictive + baseline + longitudinal + multirun_stability + worst_lines
- :mod:`pipeline`          — pipeline_dag + error_absorption + incremental_comparison + module_audit
- :mod:`robustness`        — robustness_projection (workflow CLI séparé)

Convention API
--------------
Chaque vue expose une fonction publique
``build_<name>_view_html(report_data, labels, **opts) -> str`` qui :

1. **Prend** ``report_data`` (dict construit par
   :func:`picarones.report.generator._build_report_data`),
   ``labels`` (i18n) et des options spécifiques à la vue (ex. fixtures
   externes que l'utilisateur peut fournir).
2. **Calcule** les données dont chaque renderer a besoin à partir de
   ``report_data`` quand c'est possible.
3. **Compose** le HTML des sous-renderers en blocs ``<details>``
   collapsibles (premier ouvert par défaut).
4. **Retourne** la chaîne HTML complète, ou ``""`` si aucune
   sous-section n'a de contenu (adaptive masking corpus-wide).

Le générateur principal (``generator.py``) appelle ces fonctions et
passe leur retour au template Jinja2 ``view_analyses.html`` qui les
inclut sous forme de cartes pleine largeur derrière un en-tête
identifiant la famille.

Ne pas confondre
----------------
``views/<name>.py`` = orchestrateur (composition + adaptive masking).
``<name>_render.py`` = rendu HTML d'un seul bloc atomique.
Les renderers atomiques restent inchangés, l'orchestrateur les
combine.
"""

from picarones.report.views.advanced_taxonomy import build_advanced_taxonomy_view_html
from picarones.report.views.diagnostics import build_diagnostics_view_html
from picarones.report.views.economics import build_economics_view_html
from picarones.report.views.pipeline import build_pipeline_view_html
from picarones.report.views.robustness import build_robustness_view_html

__all__ = [
    "build_advanced_taxonomy_view_html",
    "build_diagnostics_view_html",
    "build_economics_view_html",
    "build_pipeline_view_html",
    "build_robustness_view_html",
]
