"""Renderers HTML thématiques — un module par vue analytique.

Chaque module exporte une fonction ``build_<theme>_html(...)`` qui
prend un ``BenchmarkResult`` (ou un sous-dict pré-calculé) et
retourne une chaîne HTML autonome.
"""

from __future__ import annotations

__all__: list[str] = []
