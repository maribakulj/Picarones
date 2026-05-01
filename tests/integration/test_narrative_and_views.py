"""Index thématique : moteur narratif (détecteurs + arbitre + renderer)
et vues HTML qui le consomment.

Chantier 6 du plan d'évolution post-Sprint 97.

Tests couvrant cette feature
----------------------------
- :mod:`tests.test_chantier5` (classe ``TestDetectorsPackage``) —
  package thématique des détecteurs (chantier 5).
- :mod:`tests.test_views` (chantier 3) — vue diagnostics qui consomme
  les leviers calculés depuis ``picarones.measurements.levers``.

Sprints d'origine du moteur narratif
------------------------------------
- Sprint 16 : ``test_sprint16_narrative_foundations.py`` — modèle
  ``Fact``, ``FactType``, ``DetectorRegistry``.
- Sprint 19 : ``test_sprint19_narrative_engine.py`` — pipeline
  ``build_synthesis``, traçabilité anti-hallucination.
- Sprint 23 : ``test_sprint23_anti_hallucination.py`` — garde-fou
  anti-hallucination.
- Sprint 29 : ``test_sprint29_detector_registry.py`` — registre
  déclaratif.

Sprints d'origine des détecteurs spécifiques
--------------------------------------------
- Sprint 4 : 10 détecteurs (`global_leader_cer`, `statistical_tie`,
  `significant_gap`, `pareto_alternative`, `stratum_winner`,
  `stratum_collapse`, `error_profile_outlier`,
  `llm_hallucination_flag`, `robustness_fragile`,
  `confidence_warning`).
- Sprint 18 : `statistical_tie` enrichi avec Friedman+Nemenyi
  (``test_sprint18_friedman_nemenyi_cdd.py``).
- Sprint 19 : `pareto_alternative` activé avec données de coût.
- Sprint 36 : `ensemble_opportunity` — couplé aux métriques
  inter-moteurs (``test_sprint36_ensemble_narrative.py``).
- Sprint 44 : `median_mean_gap_warning` — couplé à la médiane par
  défaut (``test_sprint44_median_default.py``).
- Sprint 45/46 : `stratification_recommended`
  (``test_sprint46_stratification_html.py``).
- Sprint 73 : `engine_off_baseline`
  (``test_sprint73_baseline_comparison.py``).
- Sprint 90 : `engine_unstable`
  (``test_sprint90_engine_unstable.py``).
- Sprint 92 : `regression_in_history`
  (``test_sprint92_longitudinal.py``).

Distribution actuelle
---------------------
18 détecteurs dans 6 sous-modules (chantier 5) :

.. code-block:: text

    picarones/core/narrative/detectors/
    ├── ranking.py    (5)  global_leader_cer, statistical_tie,
    │                      significant_gap, speed_winner,
    │                      median_mean_gap_warning
    ├── pareto.py     (2)  pareto_alternative, cost_outlier
    ├── stratum.py    (3)  stratum_winner, stratum_collapse,
    │                      stratification_recommended
    ├── quality.py    (4)  error_profile_outlier,
    │                      llm_hallucination_flag,
    │                      robustness_fragile, confidence_warning
    ├── history.py    (3)  engine_off_baseline, engine_unstable,
    │                      regression_in_history
    └── ensemble.py   (1)  ensemble_opportunity

Pour exécuter tous les tests narratifs :

.. code-block:: bash

    pytest tests/test_sprint{16,19,23,29}*.py \\
           tests/test_sprint{36,44,46,73,90,92}*.py \\
           tests/test_chantier5.py::TestDetectorsPackage
"""

# Index documentaire — pas de tests propres.
