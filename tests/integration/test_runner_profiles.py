"""Index thématique : tests des profils de calcul + registre de hooks.

Chantier 6 du plan d'évolution post-Sprint 97. Ce fichier ne contient
**pas de tests propres** — il sert d'entrée thématique pour les
nouveaux contributeurs qui cherchent les tests liés aux profils du
runner et au registre central de métriques (chantier 2).

Tests couvrant cette feature
----------------------------
- :mod:`tests.test_metric_hooks` (chantier 2) — registre,
  décorateurs, sélection par profil, exécution.
- :mod:`tests.test_chantier4` (sous-classes
  ``TestNormalizeLlmContent``, ``TestLogHttpError``,
  ``TestLlmAdaptersInheritEnvVar``) pour les helpers LLM partagés.
- :mod:`tests.test_chantier5` (classe ``TestRunnerStillReachable``)
  pour la rétrocompat des fonctions privées du runner.

Sprints d'origine
-----------------
- Sprint 13 : ``test_sprint13_parallelisation_stats.py`` —
  ``_aggregate_confusion`` importée directement.
- Sprint 42 : ``test_sprint42_calibration_runner.py`` —
  ``_aggregate_calibration`` et ``_calibration_from_engine_result``.
- Sprint 87 : ``test_sprint87_readability_html.py`` — propagation
  ``corpus_lang`` au runner.

Pour exécuter tous les tests liés :

.. code-block:: bash

    pytest tests/test_metric_hooks.py \\
           tests/test_sprint13_parallelisation_stats.py \\
           tests/test_sprint42_calibration_runner.py \\
           tests/test_sprint87_readability_html.py \\
           tests/test_chantier4.py::TestLlmAdaptersInheritEnvVar \\
           tests/test_chantier5.py::TestRunnerStillReachable
"""

# Ce fichier est volontairement vide : il sert d'index documentaire,
# pas de fichier de tests.
