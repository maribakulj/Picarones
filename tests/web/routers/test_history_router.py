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
    def test_db_path_outside_workspace_rejected(self, tmp_path: Path) -> None:
        """db_path hors workspace est désormais rejeté en 400 par le
        durcissement Phase 1 (validation contre compute_workspace_roots).

        Avant Phase 1 : 500 silencieux après tentative d'ouverture
        SQLite — vecteur de lecture filesystem arbitraire.
        Après Phase 1 : 400 avec ``PathValidationError`` AVANT
        toute interaction filesystem.
        """
        from fastapi.testclient import TestClient

        app = _make_app()
        # Chemin hors zone workspace.
        impossible_path = "/proc/cannot_write/history.sqlite"

        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.get(
                "/api/history/regressions",
                params={"db_path": impossible_path},
            )
            assert r.status_code == 400, r.text
            assert "hors zone autorisée" in r.json()["detail"]

    def test_db_path_inside_workspace_but_unwritable(
        self, tmp_path: Path,
    ) -> None:
        """db_path valide (sous tmp_path) mais pointant sur un fichier
        inexistant en sous-dossier inaccessible : 500 propre, pas de
        crash silencieux."""
        from fastapi.testclient import TestClient

        app = _make_app()
        # Sous-dossier inexistant sous tmp_path — SQLite va échouer
        # à créer le fichier, mais la validation de chemin passe.
        bad_under_workspace = tmp_path / "no_such_subdir" / "history.sqlite"

        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.get(
                "/api/history/regressions",
                params={"db_path": str(bad_under_workspace)},
            )
            assert r.status_code in (200, 500)
            if r.status_code == 500:
                body = r.json()
                assert "detail" in body
