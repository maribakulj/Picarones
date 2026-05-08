"""Tests Sprint A4 — endpoint /health (item M-3 de l'audit).

Garantit le contrat de l'endpoint *liveness* utilisé par le HEALTHCHECK
Docker et tout orchestrateur (Kubernetes, systemd, supervisord).
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from picarones import __version__
from picarones.interfaces.web._legacy.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_returns_200(client: TestClient) -> None:
    """``GET /health`` doit toujours répondre 200, même sans état initialisé."""
    r = client.get("/health")
    assert r.status_code == 200


def test_health_payload_contract(client: TestClient) -> None:
    """Le payload doit contenir au minimum ``status`` et ``version``."""
    r = client.get("/health")
    body = r.json()
    assert body.get("status") == "ok"
    assert body.get("version") == __version__


def test_health_is_fast(client: TestClient) -> None:
    """Le endpoint doit répondre en moins de 100 ms (5× le seuil cible 20 ms)
    même sur le runner CI le plus lent.

    Ce check protège contre une régression où le endpoint commencerait
    à toucher la BD ou à introspecter des engines.
    """
    # Premier appel = chauffe (TestClient cold start). Mesuré sur le 2ᵉ.
    client.get("/health")
    start = time.perf_counter()
    r = client.get("/health")
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert r.status_code == 200
    assert elapsed_ms < 100, (
        f"GET /health a pris {elapsed_ms:.1f} ms (seuil : 100 ms). "
        "Vérifier qu'aucune I/O n'a été ajoutée."
    )


def test_health_does_not_open_db(client: TestClient, tmp_path) -> None:
    """``GET /health`` ne doit pas ouvrir la BD jobs.sqlite.

    Vérification indirecte : on appelle le endpoint et on s'assure
    qu'il fonctionne même quand la BD est rendue inaccessible (chemin
    inexistant via env var). Si le endpoint touchait la BD, il
    lèverait une exception SQLite.
    """
    import os

    bogus_db = str(tmp_path / "nonexistent" / "jobs.sqlite")
    old = os.environ.get("PICARONES_JOBS_DB")
    os.environ["PICARONES_JOBS_DB"] = bogus_db
    try:
        r = client.get("/health")
        assert r.status_code == 200, (
            "GET /health a échoué quand PICARONES_JOBS_DB pointe vers un "
            "chemin invalide — le endpoint touche probablement la BD."
        )
    finally:
        if old is None:
            os.environ.pop("PICARONES_JOBS_DB", None)
        else:
            os.environ["PICARONES_JOBS_DB"] = old
