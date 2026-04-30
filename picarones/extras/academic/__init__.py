"""Modules techniques sans cas d'usage prod direct.

Ces 3 modules calculent des distributions intéressantes pour la
recherche académique mais ne participent pas à la décision
*« peut-on déployer ce moteur en prod ? »*.

Modules
-------
- :mod:`taxonomy_intra_doc`   — heatmap classe×position intra-document.
- :mod:`taxonomy_cooccurrence` — matrice Jaccard inter-classes au niveau document.
- :mod:`image_predictive`     — score de complexité paléographique (poids éditoriaux).

Rétrocompat
-----------
Les imports historiques ``from picarones.core.taxonomy_intra_doc import
...`` continuent à fonctionner via des fichiers-shims laissés à
l'ancien emplacement.
"""
