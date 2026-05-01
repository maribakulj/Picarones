"""Configuration pytest globale.

Picarones expose dans ``picarones.web.state`` trois états globaux
partagés :
  - un sémaphore borné (``JOBS_SEMAPHORE``) pour les benchmarks
    concurrents,
  - un rate limiter par IP (``RATE_LIMITER``),
  - une liste cachée de browse roots dérivée à l'import dans
    ``picarones.web.routers.corpus._BROWSE_ROOTS``.

Ces états peuvent polluer des tests indépendants. Ce conftest :

1. Force des défauts test-friendly via env vars **avant** l'import
   du module ``picarones.web.app`` — sinon le sémaphore est déjà
   créé avec la valeur prod (2) au moment où le premier test
   l'utilise.
2. Restaure l'état entre chaque test via une fixture autouse qui
   purge le rate limiter, ré-injecte un sémaphore frais, et recalcule
   les browse roots selon l'environnement courant.
"""

from __future__ import annotations

import os
import threading

import pytest


# ---------------------------------------------------------------------------
# Defaults test-friendly à appliquer AVANT tout import de l'app FastAPI.
# ---------------------------------------------------------------------------

# Plafond très large pour ne jamais bloquer une suite de tests qui démarre
# rapidement plusieurs benchmarks daemon (Sprint 6 + Sprint 24).
os.environ.setdefault("PICARONES_MAX_CONCURRENT_JOBS", "32")
# S'assurer qu'on est en mode dev pour les tests existants ; les tests
# Sprint 24 qui valident le mode public le forcent eux-mêmes via monkeypatch.
os.environ.pop("PICARONES_PUBLIC_MODE", None)
# Rate limit désactivé en dev (déjà le cas, mais explicite).
os.environ.setdefault("PICARONES_RATE_LIMIT_PER_HOUR", "0")


# ---------------------------------------------------------------------------
# Fixture d'isolation : purge l'état global avant chaque test.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_web_app_state():
    """Réinitialise sémaphore, rate limiter et browse roots entre tests.

    Sans cette fixture :
      - une suite séquentielle de 32+ tests qui font ``POST /api/benchmark/*``
        peut épuiser le sémaphore (les threads daemon ne libèrent qu'à la
        fin du benchmark Tesseract, qui prend une fraction de seconde mais
        plusieurs tests consécutifs se bousculent) ;
      - un test Sprint 24 qui mute ``_BROWSE_ROOTS`` localement laisse une
        liste pointant vers un ``tmp_path`` purgé ;
      - les tests qui appellent le rate limiter directement laissent des
        timestamps dans son bucket.
    """
    try:
        from picarones.web import app as web_app
        from picarones.web import security as web_sec
        from picarones.web import state as web_state
        from picarones.web.routers import corpus as web_corpus_router
    except ImportError:
        # Tests non-web : aucun état à restaurer.
        yield
        return

    # Sauvegarde
    original_browse_roots = list(web_corpus_router._BROWSE_ROOTS)

    # Sémaphore frais à chaque test (capacité large, voir conftest top-level).
    # Le module ``state`` détient le singleton ; ``app`` en a juste une
    # référence (importée comme ``_JOBS_SEMAPHORE``) — on synchronise les deux.
    new_sem = threading.Semaphore(web_sec.get_max_concurrent_jobs())
    web_state.JOBS_SEMAPHORE = new_sem
    web_app._JOBS_SEMAPHORE = new_sem
    web_state.RATE_LIMITER.reset()
    web_state.RATE_LIMITER.max_per_hour = web_sec.get_rate_limit_per_hour()

    yield

    # Restauration
    web_corpus_router._BROWSE_ROOTS[:] = original_browse_roots
    web_state.RATE_LIMITER.reset()
