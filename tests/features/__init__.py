"""Couche d'index thématique des tests — chantier 6 post-Sprint 97.

Les fichiers ``tests/test_sprintNN_*.py`` historiques sont conservés
intégralement comme régressions chronologiques. Ce sous-package
``tests/features/`` regroupe les tests par **fonctionnalité métier**
pour aider les nouveaux contributeurs à trouver les tests pertinents
sans avoir à parcourir 95+ fichiers sprint.

Convention
----------
Chaque ``test_<feature>.py`` est soit :

1. un **fichier d'index documentaire** qui pointe (via docstring) vers
   les tests réels disséminés dans ``tests/test_sprintNN_*.py`` ;
2. ou un fichier de **vrais tests d'intégration** transversaux (ex.
   ``test_pipeline_ocr_to_alto.py`` créé au chantier 1).

Index disponibles
-----------------
- :mod:`test_pipeline_ocr_to_alto`     — bench pipeline composée
  (BaseModule + PipelineRunner) — tests E2E réels.
- :mod:`test_runner_profiles`          — profils de calcul + registre
  de hooks (chantier 2) — index documentaire.
- :mod:`test_html_views`               — vues HTML du chantier 3 +
  renderers historiques — index documentaire.
- :mod:`test_engines_and_llm`          — adapters OCR et LLM, fix
  Sprint 15 généralisé (chantier 4) — index documentaire.
- :mod:`test_narrative_and_views`      — moteur narratif + 18
  détecteurs en 6 familles — index documentaire.

Voir aussi ``docs/architecture.md`` pour la cartographie globale du
projet après les chantiers 1-5.
"""

