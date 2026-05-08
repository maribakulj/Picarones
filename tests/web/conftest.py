"""Fixtures locales aux tests de l'interface web FastAPI.

Le serveur Picarones expose dans :mod:`picarones.web.state` trois
états globaux partagés entre routes :

- ``JOBS_SEMAPHORE`` — sémaphore borné pour les benchmarks concurrents,
- ``RATE_LIMITER`` — rate limiter par IP,
- ``picarones.web.routers.corpus._BROWSE_ROOTS`` — répertoires
  autorisés à la navigation, calculés au chargement.

Ces états peuvent polluer des tests indépendants. Cette fixture
réinitialise les trois entre chaque test web — purge le rate
limiter, ré-injecte un sémaphore frais, recalcule les browse roots
selon l'environnement courant.

Discipline : la fixture est ``autouse=True`` mais n'est définie
que dans ``tests/web/conftest.py`` — les tests des cercles 1 et 2
(``tests/core/``, ``tests/measurements/``, etc.) ne paient pas le
coût de l'import de ``picarones.web.*`` à chaque test.
"""

from __future__ import annotations

import threading

import pytest


@pytest.fixture(autouse=True)
def _isolate_web_app_state():
    """Réinitialise sémaphore, rate limiter et browse roots entre tests.

    Sans cette fixture :

    - une suite séquentielle de 32+ tests qui font ``POST /api/benchmark/*``
      peut épuiser le sémaphore (les threads daemon ne libèrent qu'à la
      fin du benchmark Tesseract, qui prend une fraction de seconde mais
      plusieurs tests consécutifs se bousculent) ;
    - un test qui mute ``_BROWSE_ROOTS`` localement laisse une liste
      pointant vers un ``tmp_path`` purgé ;
    - les tests qui appellent le rate limiter directement laissent des
      timestamps dans son bucket.
    """
    from picarones.interfaces.web._legacy import app as web_app
    from picarones.interfaces.web._legacy import security as web_sec
    from picarones.interfaces.web._legacy import state as web_state
    from picarones.interfaces.web._legacy.routers import corpus as web_corpus_router

    # Sauvegarde
    original_browse_roots = list(web_corpus_router._BROWSE_ROOTS)

    # Sémaphore frais à chaque test (capacité large, voir les env vars
    # par défaut dans ``tests/conftest.py``). Le module ``state``
    # détient le singleton ; ``app`` en a une référence importée comme
    # ``_JOBS_SEMAPHORE`` — on synchronise les deux.
    new_sem = threading.Semaphore(web_sec.get_max_concurrent_jobs())
    web_state.JOBS_SEMAPHORE = new_sem
    web_app._JOBS_SEMAPHORE = new_sem
    web_state.RATE_LIMITER.reset()
    web_state.RATE_LIMITER.max_per_hour = web_sec.get_rate_limit_per_hour()

    yield

    # Restauration
    web_corpus_router._BROWSE_ROOTS[:] = original_browse_roots
    web_state.RATE_LIMITER.reset()
