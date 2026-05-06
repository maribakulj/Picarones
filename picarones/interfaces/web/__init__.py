"""Interface web FastAPI — Sprints S35-S38.

Squelette FastAPI **natif** au nouveau monde, écrit pour consommer
directement les services applicatifs du Sprint S17+ via DI explicite.
**Pas un shim** sur le legacy ``picarones.web.app``.

Architecture
------------
- ``app.py`` (S35) : factory ``create_app(WebAppState)`` qui
  produit une instance FastAPI consommant les services injectés.
  Endpoints squelette ``/health`` et ``/version``.
- (S36) routers/corpus.py : import ZIP, listing, validation.
- (S36) routers/benchmark.py : démarrage/lecture d'un run.
- (S37) routers/jobs.py : queue + persistance SQLite + cancellation.
- (S38) ui.py : Jinja2 templates + static + i18n.

Le legacy ``picarones.web.app`` reste exposé jusqu'au S46.
"""

from __future__ import annotations

from picarones.interfaces.web.app import (
    HealthResponse,
    VersionResponse,
    WebAppState,
    create_app,
)

__all__ = [
    "HealthResponse",
    "VersionResponse",
    "WebAppState",
    "create_app",
]
