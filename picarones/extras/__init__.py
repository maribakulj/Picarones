"""Plugins Picarones — Cercle 3 de l'architecture.

Modules optionnels, niche, ou préventifs qui ne servent pas
directement la question centrale du produit (« peut-on déployer ce
moteur en prod sur ce corpus ? »). Ils sont **séparables** : leur
absence ne casse pas le bench standard.

À terme, certains de ces sous-packages pourront être distribués comme
packages PyPI séparés (``picarones-historical``, ``picarones-importers``).
Pour l'instant ils vivent comme sous-packages internes pour limiter le
churn.

Convention de rétrocompat
-------------------------
Pour chaque module déplacé depuis ``picarones/core/`` ou
``picarones/report/`` vers ``picarones/extras/``, un fichier-shim est
laissé à l'ancien emplacement qui réexporte les noms publics. Les
imports historiques (``from picarones.core.taxonomy_intra_doc import
...``) continuent à fonctionner sans modification.

Voir :doc:`docs/architecture-cercles.md` pour la cartographie complète
et les critères d'assignation au Cercle 3.
"""
