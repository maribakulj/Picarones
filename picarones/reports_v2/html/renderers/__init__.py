"""Renderers HTML thématiques — un module par vue analytique.

Phase 5.C+ du retrait du legacy : les renderers historiques de
``picarones.report.*_render`` sont relocalisés ici par lots, en
préservant leur API (signatures publiques, formats de sortie, IDs
i18n) pour garantir la non-régression des rapports existants.

Convention de nommage : ``picarones.report.<theme>_render`` →
``picarones.reports_v2.html.renderers.<theme>``.  Le suffixe
``_render`` du legacy est retiré ici car déjà implicite dans la
position dans l'arborescence.

Chaque module exporte une fonction ``build_<theme>_html(...)`` qui
prend un ``BenchmarkResult`` (ou un sous-dict pré-calculé) et
retourne une chaîne HTML autonome.
"""

from __future__ import annotations

__all__: list[str] = []
