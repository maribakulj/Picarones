"""Sprint S4.3 — couverture des endpoints HTR-United / HuggingFace.

Avant S4 : ``routers/importers.py`` à 0% direct (testé
transitivement par d'autres tests web mais sans ciblage).

Cible : 80%+ de couverture des 4 endpoints :
- ``GET /api/htr-united/catalogue``
- ``POST /api/htr-united/import``
- ``GET /api/huggingface/search``
- ``POST /api/huggingface/import``

Mocking : les appels réseau sont mockés ; aucun test n'a besoin
d'Internet.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_app():
    from fastapi import FastAPI
    from picarones.interfaces.web.routers import importers as imp_router

    app = FastAPI()
    app.include_router(imp_router.router)
    return app


# ──────────────────────────────────────────────────────────────────────
# 1. HTR-United catalogue (GET)
# ──────────────────────────────────────────────────────────────────────


class TestHTRUnitedCatalogue:
    def test_default_lists_demo_catalogue(self) -> None:
        from fastapi.testclient import TestClient

        app = _make_app()
        with TestClient(app) as client:
            r = client.get("/api/htr-united/catalogue")
            assert r.status_code == 200
            body = r.json()
            assert "source" in body
            assert "total" in body
            assert "entries" in body
            assert isinstance(body["entries"], list)
            # La démo embarque au moins 1 entrée.
            assert body["total"] >= 1
            # Champs filtres exposés.
            assert "available_languages" in body
            assert "available_scripts" in body

    def test_query_filter_reduces_results(self) -> None:
        from fastapi.testclient import TestClient

        app = _make_app()
        with TestClient(app) as client:
            r1 = client.get("/api/htr-united/catalogue").json()
            r2 = client.get(
                "/api/htr-united/catalogue",
                params={"query": "zzzznonexistent"},
            ).json()
            assert r2["total"] <= r1["total"]
            # Une recherche bidon → 0 résultat (typiquement).
            assert r2["total"] == 0

    def test_language_filter_applied(self) -> None:
        from fastapi.testclient import TestClient

        app = _make_app()
        with TestClient(app) as client:
            # Premier appel : récupérer une langue valide.
            full = client.get("/api/htr-united/catalogue").json()
            available = full.get("available_languages", [])
            if not available:
                pytest.skip("Catalogue démo sans langues — fixture vide")
            lang = available[0]
            r = client.get(
                "/api/htr-united/catalogue",
                params={"language": lang},
            )
            assert r.status_code == 200


# ──────────────────────────────────────────────────────────────────────
# 2. HTR-United import (POST)
# ──────────────────────────────────────────────────────────────────────


class TestHTRUnitedImport:
    def test_unknown_entry_id_returns_404(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        app = _make_app()
        with TestClient(app) as client:
            r = client.post(
                "/api/htr-united/import",
                json={
                    "entry_id": "non_existent_id",
                    "output_dir": str(tmp_path),
                    "max_samples": 5,
                },
            )
            assert r.status_code == 404
            assert "non trouvée" in r.json()["detail"]

    def test_known_entry_calls_importer(self, tmp_path: Path) -> None:
        """Avec un entry_id du catalogue démo, l'endpoint appelle
        ``import_htr_united_corpus``.  On mocke pour éviter le
        download réel."""
        from fastapi.testclient import TestClient

        app = _make_app()
        with patch(
            "picarones.adapters.corpus.htr_united.import_htr_united_corpus",
        ) as mock_import:
            mock_import.return_value = {"imported": 3, "output_dir": str(tmp_path)}

            # Récupère un entry_id du catalogue démo.
            with TestClient(app) as client:
                catalog = client.get("/api/htr-united/catalogue").json()
                if not catalog["entries"]:
                    pytest.skip("Catalogue démo vide")
                entry_id = catalog["entries"][0]["id"]

                r = client.post(
                    "/api/htr-united/import",
                    json={
                        "entry_id": entry_id,
                        "output_dir": str(tmp_path),
                        "max_samples": 3,
                    },
                )
                assert r.status_code == 200
                assert mock_import.called


# ──────────────────────────────────────────────────────────────────────
# 3. HuggingFace search (GET)
# ──────────────────────────────────────────────────────────────────────


class TestHuggingFaceSearch:
    def test_search_returns_list(self) -> None:
        from fastapi.testclient import TestClient

        app = _make_app()
        # Mock le HF Hub pour ne pas appeler le vrai réseau.
        with patch(
            "picarones.adapters.corpus.huggingface.HuggingFaceImporter.search",
        ) as mock_search:
            fake_dataset = MagicMock()
            fake_dataset.as_dict.return_value = {
                "id": "test/dataset", "tags": ["ocr"], "language": "fr",
            }
            mock_search.return_value = [fake_dataset]

            with TestClient(app) as client:
                r = client.get(
                    "/api/huggingface/search",
                    params={"query": "ocr"},
                )
                assert r.status_code == 200
                body = r.json()
                assert body["total"] == 1
                assert body["datasets"][0]["id"] == "test/dataset"

    def test_search_empty_returns_empty_list(self) -> None:
        from fastapi.testclient import TestClient

        app = _make_app()
        with patch(
            "picarones.adapters.corpus.huggingface.HuggingFaceImporter.search",
            return_value=[],
        ):
            with TestClient(app) as client:
                r = client.get("/api/huggingface/search", params={"query": "x"})
                assert r.status_code == 200
                assert r.json() == {"total": 0, "datasets": []}

    def test_search_limit_validation(self) -> None:
        """``limit`` est entre 1 et 50 — au-delà, validation FastAPI."""
        from fastapi.testclient import TestClient

        app = _make_app()
        with TestClient(app) as client:
            r = client.get("/api/huggingface/search", params={"limit": 100})
            assert r.status_code == 422  # validation pydantic

    def test_search_tags_parsed_as_list(self) -> None:
        from fastapi.testclient import TestClient

        app = _make_app()
        with patch(
            "picarones.adapters.corpus.huggingface.HuggingFaceImporter.search",
        ) as mock_search:
            mock_search.return_value = []
            with TestClient(app) as client:
                client.get(
                    "/api/huggingface/search",
                    params={"tags": "ocr,manuscript,medieval"},
                )
                # Vérifie que les tags ont été splitté correctement.
                _, kwargs = mock_search.call_args
                assert kwargs["tags"] == ["ocr", "manuscript", "medieval"]


# ──────────────────────────────────────────────────────────────────────
# 4. HuggingFace import (POST)
# ──────────────────────────────────────────────────────────────────────


class TestHuggingFaceImport:
    def test_import_calls_importer(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        app = _make_app()
        with patch(
            "picarones.adapters.corpus.huggingface.HuggingFaceImporter.import_dataset",
        ) as mock_import:
            mock_import.return_value = {
                "imported": 5,
                "output_dir": str(tmp_path),
            }

            with TestClient(app) as client:
                r = client.post(
                    "/api/huggingface/import",
                    json={
                        "dataset_id": "test/dataset",
                        "output_dir": str(tmp_path),
                        "split": "train",
                        "max_samples": 5,
                    },
                )
                assert r.status_code == 200
                assert mock_import.called
                _, kwargs = mock_import.call_args
                assert kwargs["dataset_id"] == "test/dataset"
                assert kwargs["max_samples"] == 5
