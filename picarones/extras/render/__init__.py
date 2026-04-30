"""Renderers atomiques pour les modules ``extras/``.

Importés conditionnellement par les vues thématiques du chantier 3
(``picarones.report.views.advanced_taxonomy``, etc.) qui restent
dans le Cercle 2. Si les modules ``extras/academic/`` ou
``extras/governance/`` sont absents, ces renderers ne sont pas
sollicités et la vue masque la sous-section.

Rétrocompat
-----------
Imports historiques ``from picarones.report.taxonomy_intra_doc_render
import ...`` continuent à fonctionner via des fichiers-shims.
"""
