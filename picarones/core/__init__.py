"""Picarones — couche legacy ``core/`` en cours de retrait.

Ce package, historiquement nommé « Cercle 1 — abstractions de
domaine », est progressivement vidé au profit des canoniques en
8 couches concentriques (cf. ``docs/explanation/architecture.md``).

Modules restants
----------------
- :mod:`diff_utils`       compute_word_diff, compute_char_diff (helpers
  utilisés par quelques tests historiques ; canonique :
  :mod:`picarones.evaluation._diff_utils`).
- :mod:`xml_utils`        safe_parse_xml (canonique :
  :mod:`picarones.formats._xml_utils`).

Modules retirés (Phase 4-bis et suivantes du retrait du legacy)
---------------------------------------------------------------
- ``modules``         → ``picarones.domain.{artifacts, module_protocol}`` (Lot A).
- ``facts``           → ``picarones.domain.facts`` (Lot A).
- ``metrics``         → ``picarones.evaluation.metric_result`` (Lot B).
- ``metric_registry`` → ``picarones.evaluation.metric_registry`` (Lot B).
- ``metric_hooks``    → ``picarones.evaluation.metric_hooks`` (Lot B).
- ``results``         → ``picarones.evaluation.benchmark_result`` (Lot C).
- ``corpus``          → ``picarones.evaluation.corpus`` (Lot C).
- ``pipeline``        → ``picarones.evaluation.pipeline`` (Lot C).

Voir :doc:`docs/explanation/architecture.md` pour le manifeste complet et
:doc:`docs/reference/api-stable.md` pour le contrat de stabilité de chaque
nom exporté.
"""
