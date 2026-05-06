"""Routers FastAPI du nouveau monde — Sprints S36-S38.

Chaque router est mince : valide DTO Pydantic, appelle un service
de ``app/services``, retourne une réponse.  Pas de logique métier
dans les routers.

Routers livrés
--------------
- ``corpus.py`` (S36) : import ZIP + analyse de structure.
- ``benchmark.py`` (S36) : listing/lecture des runs.
- (S37) ``jobs.py`` : queue + persistance SQLite + cancellation.
- (S38) ``ui.py`` : templates HTML Jinja2 + i18n.
"""

from __future__ import annotations

from picarones.interfaces.web.routers.benchmark import router as benchmark_router
from picarones.interfaces.web.routers.corpus import router as corpus_router

__all__ = [
    "benchmark_router",
    "corpus_router",
]
