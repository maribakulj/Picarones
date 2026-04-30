"""Couche d'index thématique des tests (chantier 6 du plan d'évolution).

Les fichiers ``tests/test_sprintNN_*.py`` historiques sont conservés
intégralement comme régressions chronologiques.  Ce sous-package
``tests/features/`` regroupe les tests par **fonctionnalité métier**
pour aider les nouveaux contributeurs à trouver les tests pertinents
sans avoir à parcourir 95 fichiers sprint.

Convention : un fichier par feature, qui peut soit ajouter ses propres
tests, soit ré-exporter / paramétrer des tests sprint existants.
"""
