"""Sprint S8.6 — couverture des défenses path traversal du
corpus router.

Avant : 79% (les paths défensifs ``195-210`` et ``111-114``
n'étaient pas exécutés en test).

Cible : verrouille les défenses sécurité par tests d'attaque
(complète Sprint S1 sur l'autre entry point).
"""

from __future__ import annotations

import pytest


def _make_app():
    from fastapi import FastAPI
    from picarones.interfaces.web.routers import corpus as corpus_router

    app = FastAPI()
    app.include_router(corpus_router.router)
    return app


# ──────────────────────────────────────────────────────────────────────
# /api/uploads/{upload_id}/{filename} — path traversal
# ──────────────────────────────────────────────────────────────────────


class TestUploadImagePathTraversal:
    """L'endpoint qui sert les images uploadées doit refuser tout
    upload_id ou filename contenant ``/``, ``\\``, ``..``."""

    @pytest.mark.parametrize(
        "upload_id",
        ["../etc", "abc/def", "abc\\def", "../../passwd", ".."],
    )
    def test_upload_id_with_traversal_rejected(self, upload_id: str) -> None:
        from fastapi.testclient import TestClient

        app = _make_app()
        with TestClient(app) as client:
            r = client.get(f"/api/uploads/{upload_id}/foo.png")
            # 400 (rejet explicite) ou 404 (path résolu absent) — pas 200,
            # pas un fichier hors du dossier upload.
            assert r.status_code in (400, 404, 422), (
                f"upload_id={upload_id!r} accepté (status {r.status_code}) "
                f"— défense path traversal manquante."
            )

    @pytest.mark.parametrize(
        "filename",
        ["../etc/passwd", "abc/def.png", "abc\\def.png", "../config.json"],
    )
    def test_filename_with_traversal_rejected(self, filename: str) -> None:
        from fastapi.testclient import TestClient

        app = _make_app()
        with TestClient(app) as client:
            # On utilise un upload_id valide pour ne tester QUE le
            # filename — sinon le rejet de upload_id masque celui
            # de filename.
            r = client.get(f"/api/uploads/safe_upload_id/{filename}")
            assert r.status_code in (400, 404, 422), (
                f"filename={filename!r} accepté (status {r.status_code})."
            )


class TestUploadImageNotFound:
    def test_unknown_upload_id_returns_404(self) -> None:
        from fastapi.testclient import TestClient

        app = _make_app()
        with TestClient(app) as client:
            r = client.get("/api/uploads/nonexistent_upload/image.png")
            assert r.status_code == 404
