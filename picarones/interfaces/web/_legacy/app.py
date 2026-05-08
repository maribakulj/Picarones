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
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

# Sprint F (plan v2.0) — interfaces/ ne peut pas importer ``picarones`` racine.
_picarones = __import__("importlib").import_module("picarones")
__version__ = getattr(_picarones, "__version__", "unknown")
from picarones.interfaces.web._legacy import state
from picarones.interfaces.web._legacy.routers import (
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
from picarones.interfaces.web._legacy.security import csp_middleware, csrf_middleware

_logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Lifespan
# ──────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Hook de démarrage : marque les jobs orphelins comme ``interrupted``.

    Au démarrage d'un nouveau processus, tous les jobs
    encore en statut ``pending`` ou ``running`` en base sont
    forcément orphelins (le processus précédent est mort sans les
    finir). On les bascule en ``interrupted`` pour ne pas laisser
    d'état mensonger sur le tableau de bord.
    """
    # NB : on accède via ``state.JOB_STORE`` (pas un import direct) pour
    # que les fixtures de tests qui ré-affectent ``state.JOB_STORE`` à
    # un store isolé soient effectivement vues par le lifespan.
    try:
        state.JOB_STORE.mark_orphaned_jobs_interrupted()
    except sqlite3.Error as exc:  # pragma: no cover — défense en profondeur
        # Si la base de jobs est cassée au démarrage, on log en ``error``
        # (pas ``warning``) — c'est un signal opérationnel : l'app
        # tourne dans un état dégradé, le tableau de bord va être incorrect.
        _logger.error(
            "[jobs] mark_orphaned_jobs_interrupted ÉCHOUÉ — "
            "base SQLite inaccessible (%s) : le tableau de bord "
            "affichera des jobs zombies.", exc,
        )
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

# Middleware CSP + en-têtes durcis (X-Frame-Options, etc.)
app.middleware("http")(csp_middleware)

# Sprint A4 (B-11) — protection CSRF, gated par PICARONES_CSRF_REQUIRED.
# En mode public (HuggingFace Space) : bypass complet, pas de cookie.
# En mode institutionnel : double-submit cookie + signature HMAC-SHA256.
# Le middleware s'enregistre toujours mais devient no-op si la variable
# d'env n'est pas activée — coût ~0 en mode public.
app.middleware("http")(csrf_middleware)


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
