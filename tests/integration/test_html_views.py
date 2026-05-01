"""Index thématique : tests des vues HTML du chantier 3 et des
renderers historiques.

Chantier 6 du plan d'évolution post-Sprint 97. Couche d'index pour
trouver rapidement les tests liés aux vues HTML, au générateur du
rapport et aux renderers atomiques.

Tests couvrant cette feature
----------------------------
- :mod:`tests.report.test_views` (chantier 3) — 5 vues thématiques,
  adaptive masking, anti-injection, câblage générator → vues.
- :mod:`tests.integration.test_alto_baseline` (chantier 1) — métriques ALTO
  + reconstructeur baseline, partagé avec la vue pipeline.
- :mod:`tests.test_chantier3_views` — alias.

Sprints d'origine pour les renderers atomiques
----------------------------------------------
- Sprint 17 : ``test_sprint17_jinja2_refactor.py`` — refactor
  monolithe ``_HTML_TEMPLATE`` en partials.
- Sprint 41 : ``test_sprint41_ner_html.py`` — NER summary + heatmap.
- Sprint 43 : ``test_sprint43_calibration_html.py`` — ECE/MCE +
  reliability diagrams.
- Sprint 46 : ``test_sprint46_stratification_html.py`` — vue
  stratifiée par script_type.
- Sprint 62 : ``test_sprint62_philological_html.py`` — profil
  philologique 6 sections.
- Sprint 67-68 : ``test_sprint67_pipeline_html.py`` /
  ``test_sprint68_pipeline_comparison_html.py`` — vue pipeline.
- Sprint 86 : ``test_sprint86_aii5_html.py`` — recherchabilité +
  séquences numériques.
- Sprint 87 : ``test_sprint87_readability_html.py`` — Flesch.
- Sprint 89 : ``test_sprint89_specialization.py`` — spécialisation
  inter-moteurs.
- Sprint 88 : ``test_sprint88_robustness_projection_html.py`` —
  déficit projeté.

Pour exécuter tous les tests de rendu HTML :

.. code-block:: bash

    pytest tests/test_views.py tests/test_sprint*_html.py
"""

# Index documentaire — pas de tests propres.
