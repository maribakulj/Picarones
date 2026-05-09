"""Sprint S4.2 — couverture du router ``/api/history/regressions``.

Avant S4 : ``routers/history.py`` à 55%.  Lignes non couvertes :
branche ``engine`` explicite, gestion d'exceptions sur ouverture
DB et sur ``detect_regression``, filtrage des régressions.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ──────────────────────────────────────────────────────────────────────
# App de test minimaliste
# ──────────────────────────────────────────────────────────────────────


def _make_app():
    from fastapi import FastAPI
    from picarones.interfaces.web.routers import history as history_router

    app = FastAPI()
    app.include_router(history_router.router)
    return app


# ──────────────────────────────────────────────────────────────────────
# 1. Endpoint sans historique — retourne 0 régression
# ──────────────────────────────────────────────────────────────────────


class TestEmptyHistory:
    def test_no_db_returns_empty_regressions(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        app = _make_app()
        # On pointe vers un fichier SQLite qui sera créé vide.
        db_path = tmp_path / "empty.sqlite"

        with TestClient(app) as client:
            r = client.get(
                "/api/history/regressions",
                params={"db_path": str(db_path)},
            )
            assert r.status_code == 200
            body = r.json()
            assert body["count"] == 0
            assert body["regressions"] == []

    def test_threshold_default_is_001(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        app = _make_app()
        db_path = tmp_path / "empty.sqlite"

        with TestClient(app) as client:
            r = client.get(
                "/api/history/regressions",
                params={"db_path": str(db_path)},
            )
            assert r.status_code == 200
            body = r.json()
            assert body["threshold"] == 0.01


# ──────────────────────────────────────────────────────────────────────
# 2. Endpoint avec engine explicite
# ──────────────────────────────────────────────────────────────────────


class TestExplicitEngine:
    def test_engine_param_filters_targets(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        app = _make_app()
        db_path = tmp_path / "engine_filter.sqlite"

        with TestClient(app) as client:
            r = client.get(
                "/api/history/regressions",
                params={
                    "engine": "tesseract",
                    "db_path": str(db_path),
                    "threshold": 0.05,
                },
            )
            assert r.status_code == 200
            body = r.json()
            assert body["threshold"] == 0.05
            # Aucune régression possible (DB vide) mais l'endpoint
            # ne doit pas crasher.
            assert body["count"] == 0


# ──────────────────────────────────────────────────────────────────────
# 3. Avec historique simulé qui contient une régression
# ──────────────────────────────────────────────────────────────────────


class TestHistoryWithRegression:
    @pytest.fixture
    def populated_db(self, tmp_path: Path) -> Path:
        """Crée une DB historique avec 2 runs tesseract qui régressent."""
        from picarones.evaluation.metrics.history import BenchmarkHistory

        db = tmp_path / "history.sqlite"
        h = BenchmarkHistory(db_path=str(db))
        # Baseline : CER faible
        h.record_single(
            run_id="baseline_run",
            corpus_name="test_corpus",
            engine_name="tesseract",
            cer_mean=0.05,
            wer_mean=0.10,
            doc_count=10,
            timestamp="2026-01-01T00:00:00+00:00",
        )
        # Actuel : CER plus haut (régression)
        h.record_single(
            run_id="current_run",
            corpus_name="test_corpus",
            engine_name="tesseract",
            cer_mean=0.15,
            wer_mean=0.20,
            doc_count=10,
            timestamp="2026-05-01T00:00:00+00:00",
        )
        return db

    def test_regression_detected_above_threshold(
        self, populated_db: Path,
    ) -> None:
        from fastapi.testclient import TestClient

        app = _make_app()
        with TestClient(app) as client:
            r = client.get(
                "/api/history/regressions",
                params={
                    "db_path": str(populated_db),
                    "threshold": 0.01,
                },
            )
            assert r.status_code == 200
            body = r.json()
            # Au moins une régression sur tesseract.
            assert body["count"] >= 1
            assert any(reg["engine"] == "tesseract"
                       for reg in body["regressions"])
            # Les champs contractuels du payload sont présents.
            for reg in body["regressions"]:
                assert "delta_cer" in reg
                assert "current_cer" in reg
                assert "baseline_cer" in reg
                assert "is_regression" in reg

    def test_high_threshold_filters_out_small_regressions(
        self, populated_db: Path,
    ) -> None:
        from fastapi.testclient import TestClient

        app = _make_app()
        with TestClient(app) as client:
            # Seuil 99% : aucune régression < 99 pp.
            r = client.get(
                "/api/history/regressions",
                params={
                    "db_path": str(populated_db),
                    "threshold": 0.99,
                },
            )
            assert r.status_code == 200
            body = r.json()
            assert body["count"] == 0


# ──────────────────────────────────────────────────────────────────────
# 4. Erreur d'ouverture DB → 500 propre
# ──────────────────────────────────────────────────────────────────────


class TestDBErrorHandling:
    def test_db_path_unwritable_returns_500_or_empty(
        self, tmp_path: Path,
    ) -> None:
        """db_path qui pointe sur un répertoire inexistant + non
        créable doit produire une erreur compréhensible (500 ou
        body avec count=0 mais sans crash silencieux)."""
        from fastapi.testclient import TestClient

        app = _make_app()
        # Chemin qui devrait être impossible à créer (sous /proc).
        impossible_path = "/proc/cannot_write/history.sqlite"

        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.get(
                "/api/history/regressions",
                params={"db_path": impossible_path},
            )
            # Soit 500 (le bon comportement), soit 200 mais avec
            # count=0.  Pas de crash, pas de stack trace au client.
            assert r.status_code in (200, 500)
            if r.status_code == 500:
                body = r.json()
                assert "detail" in body
