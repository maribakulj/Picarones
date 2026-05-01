"""Plugins Picarones — Cercle 3 de l'architecture.

Modules optionnels et **séparables** : leur absence ne casse pas
le bench standard. Ils étendent Picarones avec des fonctionnalités
qui ne servent pas directement la question centrale (« peut-on
déployer ce moteur en prod sur ce corpus ? ») et qui dépendent
typiquement de sources externes (IIIF, eScriptorium, HuggingFace…).

Sous-packages
-------------
- :mod:`importers` — connecteurs corpus (IIIF, Gallica, HTR-United,
  HuggingFace, eScriptorium). Les modules ``huggingface`` et
  ``escriptorium`` émettent un ``UserWarning`` à l'import car ils
  n'ont pas été validés sur des instances de production.

À terme, ces sous-packages pourront être distribués comme packages
PyPI séparés (``picarones-importers``…). Pour l'instant ils vivent
comme sous-packages internes pour limiter le churn.

Voir :doc:`docs/architecture.md` pour la cartographie complète
et les critères d'assignation au Cercle 3.
"""
