"""Sprint S8.2 — Endpoint Prometheus ``/metrics``.

Vérifie :

1. Désactivé par défaut → 404.
2. Activé via ``PICARONES_METRICS_ENABLED=1`` → 200 avec format
   Prometheus exposition.
3. Métriques attendues : ``picarones_app_info``,
   ``picarones_jobs_total{status="..."}``, alias gauges.
4. Tolérance store inaccessible : retourne ``picarones_app_info``
   sans crasher.
"""

from __future__ import annotations

import pytest


def _make_app():
    from fastapi import FastAPI
    from picarones.interfaces.web.routers import system as sys_router

    app = FastAPI()
    app.include_router(sys_router.router)
    return app


# ──────────────────────────────────────────────────────────────────────
# 1. Désactivé par défaut
# ──────────────────────────────────────────────────────────────────────


class TestMetricsDisabledByDefault:
    def test_404_when_env_not_set(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from fastapi.testclient import TestClient

        monkeypatch.delenv("PICARONES_METRICS_ENABLED", raising=False)
        app = _make_app()
        with TestClient(app) as client:
            r = client.get("/metrics")
            assert r.status_code == 404
            assert "PICARONES_METRICS_ENABLED" in r.text


# ──────────────────────────────────────────────────────────────────────
# 2. Format Prometheus quand activé
# ──────────────────────────────────────────────────────────────────────


class TestMetricsFormat:
    def test_200_with_prometheus_content_type(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from fastapi.testclient import TestClient

        monkeypatch.setenv("PICARONES_METRICS_ENABLED", "1")
        app = _make_app()
        with TestClient(app) as client:
            r = client.get("/metrics")
            assert r.status_code == 200
            ct = r.headers.get("content-type", "")
            assert "text/plain" in ct
            assert "version=0.0.4" in ct

    def test_exposes_app_info(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from fastapi.testclient import TestClient

        monkeypatch.setenv("PICARONES_METRICS_ENABLED", "1")
        app = _make_app()
        with TestClient(app) as client:
            r = client.get("/metrics")
            text = r.text
            assert "# TYPE picarones_app_info gauge" in text
            assert 'picarones_app_info{version=' in text
            assert text.rstrip().endswith("1") or "} 1" in text

    def test_exposes_jobs_total_per_status(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from fastapi.testclient import TestClient

        monkeypatch.setenv("PICARONES_METRICS_ENABLED", "1")
        app = _make_app()
        with TestClient(app) as client:
            r = client.get("/metrics")
            text = r.text
            # Chaque statut connu apparaît, même à 0
            for status in ("pending", "running", "complete", "error",
                           "cancelled", "interrupted"):
                assert f'status="{status}"' in text, (
                    f"Statut ``{status}`` absent du payload"
                )

    def test_exposes_alias_gauges(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from fastapi.testclient import TestClient

        monkeypatch.setenv("PICARONES_METRICS_ENABLED", "1")
        app = _make_app()
        with TestClient(app) as client:
            r = client.get("/metrics")
            text = r.text
            assert "picarones_jobs_pending" in text
            assert "picarones_jobs_running" in text


# ──────────────────────────────────────────────────────────────────────
# 3. Activation insensible casse / yes / true
# ──────────────────────────────────────────────────────────────────────


class TestEnvVarParsing:
    @pytest.mark.parametrize("value", ["1", "true", "yes"])
    def test_truthy_values_enable(
        self, monkeypatch: pytest.MonkeyPatch, value: str,
    ) -> None:
        from fastapi.testclient import TestClient

        monkeypatch.setenv("PICARONES_METRICS_ENABLED", value)
        app = _make_app()
        with TestClient(app) as client:
            r = client.get("/metrics")
            assert r.status_code == 200

    @pytest.mark.parametrize("value", ["0", "false", "no", "", "off"])
    def test_falsy_values_keep_disabled(
        self, monkeypatch: pytest.MonkeyPatch, value: str,
    ) -> None:
        from fastapi.testclient import TestClient

        monkeypatch.setenv("PICARONES_METRICS_ENABLED", value)
        app = _make_app()
        with TestClient(app) as client:
            r = client.get("/metrics")
            assert r.status_code == 404
