"""Interface web Picarones — orchestrateur FastAPI.

Lance avec :

.. code-block:: bash

    picarones serve [--port 8000] [--host 127.0.0.1]
    # ou directement :
    uvicorn picarones.web.app:app --reload --port 8000

L'application est intentionnellement minimaliste : elle se contente
d'instancier ``FastAPI``, de monter le middleware de sécurité (CSP,
en-têtes durcis), de servir les fichiers statiques, puis d'inclure
les 11 routers thématiques de :mod:`picarones.web.routers`. Toute la
logique métier vit dans les sous-modules :

- :mod:`picarones.web.state` — singletons et helpers transverses
- :mod:`picarones.web.models` — Pydantic schemas
- :mod:`picarones.web.corpus_utils` — parsing XML, analyse corpus
- :mod:`picarones.web.engine_utils` — détection moteurs, capacités
- :mod:`picarones.web.benchmark_utils` — workers threadés
- :mod:`picarones.web.config_utils` — validation config utilisateur
- :mod:`picarones.web.routers.*` — 11 ``APIRouter`` thématiques
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from picarones import __version__
from picarones.web import state as _state
from picarones.web.routers import (
    benchmark as _benchmark_router,
    config as _config_router,
    corpus as _corpus_router,
    engines as _engines_router,
    history as _history_router,
    home as _home_router,
    importers as _importers_router,
    normalization as _normalization_router,
    reports as _reports_router,
    synthesis as _synthesis_router,
    system as _system_router,
)
from picarones.web.security import csp_middleware

_logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Lifespan
# ──────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Hook de démarrage : marque les jobs orphelins comme ``interrupted``.

    Sprint 26 — au démarrage d'un nouveau processus, tous les jobs
    encore en statut ``pending`` ou ``running`` en base sont
    forcément orphelins (le processus précédent est mort sans les
    finir). On les bascule en ``interrupted`` pour ne pas laisser
    d'état mensonger sur le tableau de bord.
    """
    try:
        _state.JOB_STORE.mark_orphaned_jobs_interrupted()
    except Exception as exc:  # pragma: no cover — défense en profondeur
        _logger.warning("[jobs] mark_orphaned_jobs_interrupted échoué : %s", exc)
    yield


# ──────────────────────────────────────────────────────────────────────────
# Instance FastAPI
# ──────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Picarones",
    description=(
        "Plateforme de comparaison de moteurs OCR/HTR pour documents patrimoniaux"
    ),
    version=__version__,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=_lifespan,
)

# Sprint 24 — middleware CSP + en-têtes durcis (X-Frame-Options, etc.)
app.middleware("http")(csp_middleware)


# ──────────────────────────────────────────────────────────────────────────
# Fichiers statiques
# ──────────────────────────────────────────────────────────────────────────

_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ──────────────────────────────────────────────────────────────────────────
# Routers thématiques
# ──────────────────────────────────────────────────────────────────────────

# Ordre indifférent fonctionnellement, mais regroupé par domaine
# pour la lisibilité (info → données → processus → présentation).
app.include_router(_system_router.router)
app.include_router(_engines_router.router)
app.include_router(_corpus_router.router)
app.include_router(_normalization_router.router)
app.include_router(_config_router.router)
app.include_router(_synthesis_router.router)
app.include_router(_history_router.router)
app.include_router(_reports_router.router)
app.include_router(_importers_router.router)
app.include_router(_benchmark_router.router)
app.include_router(_home_router.router)
