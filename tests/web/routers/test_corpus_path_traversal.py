"""Sprint S8.6 — couverture des défenses path traversal du
corpus router.

Avant rewrite (bricolage)
-------------------------
Le test original ciblait ``/api/uploads/{upload_id}/{filename}``
qui n'existe **pas** : la vraie route est
``/api/corpus/image/{upload_id}/{filename}`` (router corpus).
Conséquence : tous les cas testés retournaient 404 depuis
le handler 404 par défaut de FastAPI — la défense n'était
**jamais exécutée**.  De plus, l'assertion ``status_code in
(400, 404, 422)`` acceptait n'importe quoi sans vérifier que
le contenu d'un fichier poison hors-corpus n'était pas
exfiltré, ce qui est précisément le risque qu'un test
path-traversal doit verrouiller.

Rewrite (S8.6.6)
----------------
1. Cible la **vraie** route : ``/api/corpus/image/...``.
2. Monkeypatch ``UPLOADS_DIR`` vers un ``tmp_path`` contrôlé pour :
   - poser un corpus légitime + une image accessible
     (contrôle positif : la requête saine retourne 200 +
     les bytes attendus),
   - poser un fichier **poison** hors du dossier d'uploads et
     vérifier que les attaques ne renvoient jamais son contenu.
3. Choisit des vecteurs d'attaque qui **atteignent réellement
   le handler** (sans normalisation côté httpx ou Starlette) :
   - upload_id contenant ``..`` littéral (pas de ``/``) →
     ``...``, ``..foo`` ;
   - upload_id avec ``\\`` URL-encodé (``%5C``) ;
   - upload_id avec ``..`` URL-encodé (``%2E%2E``) ;
   - filename avec mêmes vecteurs.
4. Asserte 400 (défense déclenchée) ET assert que le corps
   ne contient **jamais** le contenu poison — la garantie
   recherchée.
"""

from __future__ import annotations

import base64

import pytest

POISON_CONTENT = "PWNED-secret-do-not-leak-aaaa"

# 1×1 PNG transparent valide (en bytes minimal pour FileResponse).
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkAAIAAAoAAv/lpgAA"
    "AABJRU5ErkJggg==",
)


@pytest.fixture
def app_with_uploads(tmp_path, monkeypatch):
    """Construit une app FastAPI dont ``UPLOADS_DIR`` pointe vers
    un ``tmp_path`` contenant un corpus légitime, et pose un
    fichier poison **hors** de ``UPLOADS_DIR`` pour vérifier
    qu'aucune fuite n'est possible.
    """
    from fastapi import FastAPI

    from picarones.interfaces.web.routers import corpus as corpus_router

    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    safe_corpus = uploads_dir / "safe_corpus"
    safe_corpus.mkdir()
    (safe_corpus / "image.png").write_bytes(_TINY_PNG)

    # Fichier poison hors du dossier d'uploads — un attaquant
    # voudrait y accéder via ``../<chemin>``.
    poison = tmp_path / "secret.txt"
    poison.write_text(POISON_CONTENT, encoding="utf-8")

    monkeypatch.setattr(corpus_router, "UPLOADS_DIR", uploads_dir)

    app = FastAPI()
    app.include_router(corpus_router.router)
    return app, uploads_dir, poison


class TestLegitimateAccess:
    """Contrôle positif : sans tentative de traversée, la route
    sert bien le fichier attendu."""

    def test_legitimate_image_served(self, app_with_uploads) -> None:
        from fastapi.testclient import TestClient

        app, uploads_dir, _ = app_with_uploads
        with TestClient(app) as client:
            r = client.get("/api/corpus/image/safe_corpus/image.png")
            assert r.status_code == 200, r.text
            assert r.content == _TINY_PNG
            assert r.headers["content-type"] == "image/png"

    def test_unknown_corpus_returns_404(self, app_with_uploads) -> None:
        from fastapi.testclient import TestClient

        app, _, _ = app_with_uploads
        with TestClient(app) as client:
            # ``does_not_exist`` n'est pas un nom interdit (pas de ``..``,
            # pas de ``/``) → la défense ne se déclenche pas (400) ;
            # le 404 vient de l'absence du fichier (sémantique propre).
            r = client.get("/api/corpus/image/does_not_exist/image.png")
            assert r.status_code == 404


class TestUploadIdTraversalRejected:
    """L'``upload_id`` doit être rejeté avec 400 dès qu'il contient
    ``..``, ``/`` ou ``\\`` — et le corps ne doit jamais contenir
    le contenu du fichier poison."""

    @pytest.mark.parametrize(
        "upload_id",
        [
            "...",            # triple-dot, contient ``..``
            "..foo",          # littéral ``..`` en préfixe
            "foo..bar",       # littéral ``..`` en infixe
            "%2E%2E",         # ``..`` URL-encodé (httpx ne normalise pas)
            "..%5Cetc",       # ``..\etc`` (backslash URL-encodé)
            "%5Cetc",         # ``\etc`` seul → contient ``\``
        ],
    )
    def test_upload_id_with_traversal_returns_400(
        self, app_with_uploads, upload_id: str,
    ) -> None:
        from fastapi.testclient import TestClient

        app, _, _ = app_with_uploads
        with TestClient(app) as client:
            r = client.get(f"/api/corpus/image/{upload_id}/image.png")
            assert r.status_code == 400, (
                f"upload_id={upload_id!r} accepté avec status "
                f"{r.status_code} — défense path traversal manquante."
            )
            assert "upload_id" in r.text.lower(), (
                f"message d'erreur attendu mentionnant 'upload_id' : {r.text!r}"
            )
            # Garantie de non-fuite : le corps ne contient pas le
            # contenu du fichier poison hors-dossier.
            assert POISON_CONTENT not in r.text


class TestFilenameTraversalRejected:
    """Le ``filename`` doit être rejeté avec 400 dès qu'il contient
    ``..``, ``/`` ou ``\\``, même si l'``upload_id`` est valide."""

    @pytest.mark.parametrize(
        "filename",
        [
            "..passwd",       # littéral ``..`` en préfixe
            "image..png",     # littéral ``..`` en infixe
            "%2E%2Esecret",   # ``..secret`` URL-encodé
            "foo%5Cbar",      # ``foo\bar`` (backslash URL-encodé)
            "%5C..",          # ``\..``
        ],
    )
    def test_filename_with_traversal_returns_400(
        self, app_with_uploads, filename: str,
    ) -> None:
        from fastapi.testclient import TestClient

        app, _, _ = app_with_uploads
        with TestClient(app) as client:
            # On utilise ``safe_corpus`` (corpus légitime monkeypatché)
            # pour ne tester QUE le filename — sinon le rejet de
            # upload_id masquerait celui de filename.
            r = client.get(f"/api/corpus/image/safe_corpus/{filename}")
            assert r.status_code == 400, (
                f"filename={filename!r} accepté avec status "
                f"{r.status_code} — défense path traversal manquante."
            )
            assert "filename" in r.text.lower(), (
                f"message d'erreur attendu mentionnant 'filename' : {r.text!r}"
            )
            # Garantie de non-fuite.
            assert POISON_CONTENT not in r.text


class TestNoExfiltrationOfPoisonFile:
    """Verrouille le contrat de sécurité : peu importe l'attaque,
    le contenu du fichier poison hors-dossier ne doit jamais
    apparaître dans la réponse.  Catch-all des vecteurs vus en
    pentest interne BnF."""

    @pytest.mark.parametrize(
        "url",
        [
            "/api/corpus/image/safe_corpus/..%2F..%2Fsecret.txt",
            "/api/corpus/image/..%2F..%2F/secret.txt",
            "/api/corpus/image/.../secret.txt",
            "/api/corpus/image/safe_corpus/..secret.txt",
            "/api/corpus/image/%2E%2E/secret.txt",
        ],
    )
    def test_attack_does_not_leak_poison(
        self, app_with_uploads, url: str,
    ) -> None:
        from fastapi.testclient import TestClient

        app, _, poison = app_with_uploads
        # Sanity check : le fichier poison existe vraiment et
        # contient bien le marqueur — sinon le test est vide.
        assert poison.read_text(encoding="utf-8") == POISON_CONTENT

        with TestClient(app) as client:
            r = client.get(url)
            # Le statut peut être 400 (défense), 404 (route ne match
            # pas après normalisation), ou même 200 sur un fichier
            # légitime, MAIS le corps ne doit jamais contenir le
            # marqueur secret.
            assert POISON_CONTENT not in r.text, (
                f"fuite détectée pour {url!r} (status {r.status_code}) — "
                f"contenu poison exfiltré.  Body: {r.text[:200]!r}"
            )
