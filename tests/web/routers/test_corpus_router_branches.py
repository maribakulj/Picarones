"""Sprint S8.7 — couverture des branches d'erreur du corpus router.

Cible (avant) : 88% — lignes 36-37, 50, 71-72, 111-114, 130-132,
169, 174, 183-184 non couvertes.  Toutes représentent des
contrats fonctionnels réels (403 sur path interdit, 415 sur
image rejetée, robustness sur uploads dir absent…).
"""

from __future__ import annotations

from pathlib import Path


def _make_app(tmp_path, monkeypatch):
    from fastapi import FastAPI

    from picarones.interfaces.web.routers import corpus as corpus_router

    uploads_dir = tmp_path / "uploads"
    monkeypatch.setattr(corpus_router, "UPLOADS_DIR", uploads_dir)
    # ``_BROWSE_ROOTS`` est calculé au module-load depuis l'``UPLOADS_DIR``
    # original.  Pour le browse 403 on remplace par un set explicite
    # contenant uniquement le dossier autorisé du test.
    monkeypatch.setattr(
        corpus_router, "_BROWSE_ROOTS", [tmp_path.resolve()],
    )

    app = FastAPI()
    app.include_router(corpus_router.router)
    return app, uploads_dir


# ──────────────────────────────────────────────────────────────────────
# /api/corpus/browse — défense 403 + 404
# ──────────────────────────────────────────────────────────────────────


class TestBrowseDefenses:
    def test_browse_outside_allowed_roots_returns_403(
        self, tmp_path, monkeypatch,
    ) -> None:
        """Tente de browser un dossier réel mais hors des
        ``_BROWSE_ROOTS`` autorisés → 403."""
        from fastapi.testclient import TestClient

        # Crée un dossier réel hors du tmp_path autorisé.
        outside_dir = tmp_path.parent / f"outside_{tmp_path.name}"
        outside_dir.mkdir()
        try:
            app, _ = _make_app(tmp_path, monkeypatch)
            with TestClient(app) as client:
                r = client.get(
                    "/api/corpus/browse",
                    params={"path": str(outside_dir)},
                )
                assert r.status_code == 403, r.text
                assert "Accès refusé" in r.text or "refusé" in r.text
        finally:
            outside_dir.rmdir()

    def test_browse_nonexistent_path_returns_404(
        self, tmp_path, monkeypatch,
    ) -> None:
        from fastapi.testclient import TestClient

        app, _ = _make_app(tmp_path, monkeypatch)
        with TestClient(app) as client:
            r = client.get(
                "/api/corpus/browse",
                params={"path": str(tmp_path / "nope")},
            )
            assert r.status_code == 404

    def test_browse_legitimate_path_returns_listing(
        self, tmp_path, monkeypatch,
    ) -> None:
        """Contrôle positif : path autorisé → 200 + listing avec
        détection ``has_corpus`` sur les sous-dossiers contenant
        des ``.gt.txt``."""
        from fastapi.testclient import TestClient

        # Sous-dossier avec un fichier ``.gt.txt`` → has_corpus=True.
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "doc1.gt.txt").write_text("ground truth", encoding="utf-8")

        app, _ = _make_app(tmp_path, monkeypatch)
        with TestClient(app) as client:
            r = client.get(
                "/api/corpus/browse", params={"path": str(tmp_path)},
            )
            assert r.status_code == 200
            data = r.json()
            sub_item = next(
                it for it in data["items"] if it["name"] == "sub"
            )
            assert sub_item["is_dir"] is True
            assert sub_item["gt_count"] == 1
            assert sub_item["has_corpus"] is True


# ──────────────────────────────────────────────────────────────────────
# /api/corpus/uploads — listing avec dossiers absents/non-dir
# ──────────────────────────────────────────────────────────────────────


class TestUploadsListing:
    def test_uploads_dir_missing_returns_empty_list(
        self, tmp_path, monkeypatch,
    ) -> None:
        """Pas d'``UPLOADS_DIR`` → liste vide (pas une erreur)."""
        from fastapi.testclient import TestClient

        app, uploads_dir = _make_app(tmp_path, monkeypatch)
        assert not uploads_dir.exists()  # pre-condition
        with TestClient(app) as client:
            r = client.get("/api/corpus/uploads")
            assert r.status_code == 200
            assert r.json() == {"uploads": []}

    def test_uploads_skips_non_directory_entries(
        self, tmp_path, monkeypatch,
    ) -> None:
        """Un fichier accidentel à la racine d'``UPLOADS_DIR`` ne doit
        pas planter le listing — on saute, on continue."""
        from fastapi.testclient import TestClient

        app, uploads_dir = _make_app(tmp_path, monkeypatch)
        uploads_dir.mkdir()
        (uploads_dir / "stray.txt").write_text("not a corpus")

        # Vrai corpus dans un sous-dossier — détecté normalement.
        real = uploads_dir / "real_corpus"
        real.mkdir()
        (real / "img.png").write_bytes(b"")
        (real / "img.gt.txt").write_text("gt", encoding="utf-8")

        with TestClient(app) as client:
            r = client.get("/api/corpus/uploads")
            assert r.status_code == 200
            uploads = r.json()["uploads"]
            ids = [u["corpus_id"] for u in uploads]
            assert "real_corpus" in ids
            assert "stray.txt" not in ids, (
                "le fichier non-dir aurait dû être sauté"
            )

    def test_uploads_handles_broken_corpus_with_warning(
        self, tmp_path, monkeypatch, caplog,
    ) -> None:
        """``analyze_corpus_dir`` qui plante sur un dossier doit être
        loggé en warning, pas masquer la liste des autres."""
        from fastapi.testclient import TestClient

        from picarones.interfaces.web.routers import corpus as corpus_router

        app, uploads_dir = _make_app(tmp_path, monkeypatch)
        uploads_dir.mkdir()
        (uploads_dir / "good_corpus").mkdir()
        (uploads_dir / "broken_corpus").mkdir()

        # Force ``analyze_corpus_dir`` à lever pour ``broken_corpus``
        # uniquement, pour vérifier que le listing continue après
        # l'exception.
        original_analyze = corpus_router.analyze_corpus_dir

        def fake_analyze(d: Path) -> dict:
            if d.name == "broken_corpus":
                raise RuntimeError("disque corrompu simulé")
            return original_analyze(d)

        monkeypatch.setattr(
            corpus_router, "analyze_corpus_dir", fake_analyze,
        )

        with caplog.at_level("WARNING"):
            with TestClient(app) as client:
                r = client.get("/api/corpus/uploads")
        assert r.status_code == 200
        # ``good_corpus`` est listé, ``broken_corpus`` ignoré + warning.
        ids = [u["corpus_id"] for u in r.json()["uploads"]]
        assert "good_corpus" in ids
        assert "broken_corpus" not in ids
        assert any(
            "broken_corpus" in rec.message for rec in caplog.records
        ), "warning sur le corpus cassé attendu"


# ──────────────────────────────────────────────────────────────────────
# /api/corpus/upload — dépassement de quota → 413 (streaming)
# ──────────────────────────────────────────────────────────────────────


class TestUploadImageRejection:
    def test_oversized_file_returns_413_at_stream_time(
        self, tmp_path, monkeypatch,
    ) -> None:
        """Fichier > plafond unitaire → rejet **au streaming** (413),
        avant toute matérialisation RAM complète. Plus correct que
        l'ancien 415 (qui exigeait de bufferiser le fichier entier
        avant de le refuser)."""
        from fastapi.testclient import TestClient

        app, uploads_dir = _make_app(tmp_path, monkeypatch)
        uploads_dir.mkdir()
        monkeypatch.setenv("PICARONES_MAX_UPLOAD_MB", "1")

        big_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * (2 * 1024 * 1024)

        with TestClient(app) as client:
            r = client.post(
                "/api/corpus/upload",
                files={"files": ("big.png", big_data, "image/png")},
            )
            assert r.status_code == 413, r.text
            assert "PICARONES_MAX_UPLOAD_MB" in r.text
        # Quota dépassé ⇒ corpus_dir purgé (pas de résidu disque).
        assert list(uploads_dir.iterdir()) == [], (
            "corpus_dir aurait dû être supprimé après dépassement quota"
        )

    def test_total_upload_cap_returns_413(
        self, tmp_path, monkeypatch,
    ) -> None:
        """Plusieurs fichiers chacun sous la limite unitaire mais dont
        le cumul dépasse ``PICARONES_MAX_TOTAL_UPLOAD_MB`` → 413."""
        from fastapi.testclient import TestClient

        app, uploads_dir = _make_app(tmp_path, monkeypatch)
        uploads_dir.mkdir()
        monkeypatch.setenv("PICARONES_MAX_UPLOAD_MB", "10")
        monkeypatch.setenv("PICARONES_MAX_TOTAL_UPLOAD_MB", "1")

        blob = b"\x00" * (700 * 1024)  # 0.7 Mo, sous la limite unitaire

        with TestClient(app) as client:
            r = client.post(
                "/api/corpus/upload",
                files=[
                    ("files", ("a.txt", blob, "text/plain")),
                    ("files", ("b.txt", blob, "text/plain")),
                ],
            )
            assert r.status_code == 413, r.text
            assert "PICARONES_MAX_TOTAL_UPLOAD_MB" in r.text
        assert list(uploads_dir.iterdir()) == []


# ──────────────────────────────────────────────────────────────────────
# _is_path_allowed — branche d'exception (ValueError/TypeError)
# ──────────────────────────────────────────────────────────────────────


class TestBrowsePermissionError:
    """``iterdir()`` lève ``PermissionError`` sur un dossier sans
    droits → 403 (avec le message d'erreur OS).  Couvre la branche
    ``except PermissionError`` du browse handler."""

    def test_iterdir_permission_error_returns_403(
        self, tmp_path, monkeypatch,
    ) -> None:
        from fastapi.testclient import TestClient

        # Crée un dossier valide qui passe les checks d'existence et
        # ``_is_path_allowed`` (qui dépend de ``_BROWSE_ROOTS``).
        target = tmp_path / "blocked"
        target.mkdir()

        app, _ = _make_app(tmp_path, monkeypatch)

        # Mock ``Path.iterdir`` pour lever PermissionError sur le
        # target spécifique.
        original_iterdir = Path.iterdir

        def raising_iterdir(self):
            if self == target:
                raise PermissionError("EACCES: permission denied")
            return original_iterdir(self)

        monkeypatch.setattr(Path, "iterdir", raising_iterdir)

        with TestClient(app) as client:
            r = client.get(
                "/api/corpus/browse", params={"path": str(target)},
            )
            assert r.status_code == 403, r.text
            assert "permission" in r.text.lower() or "EACCES" in r.text


class TestUploadGenericException:
    """Branche catch-all dans ``api_corpus_upload`` : si le
    ``_finalize_uploaded_dir`` lève autre chose qu'une
    ``ValueError`` (par ex. ``OSError`` disque plein), on doit
    nettoyer ``corpus_dir`` ET retourner un 500 propre."""

    def test_unexpected_exception_returns_500_and_cleans_corpus_dir(
        self, tmp_path, monkeypatch,
    ) -> None:
        from fastapi.testclient import TestClient

        from picarones.interfaces.web.routers import corpus as corpus_router

        app, uploads_dir = _make_app(tmp_path, monkeypatch)
        uploads_dir.mkdir()

        # Mock _finalize_uploaded_dir pour lever une exception
        # non-ValueError (donc non interceptée par le 415 path).
        def raising_finalize(corpus_dir, staged):
            raise RuntimeError("disk full simulé")

        monkeypatch.setattr(
            corpus_router, "_finalize_uploaded_dir", raising_finalize,
        )

        with TestClient(app) as client:
            r = client.post(
                "/api/corpus/upload",
                files={"files": ("test.png", b"fake-png-bytes", "image/png")},
            )
            assert r.status_code == 500, r.text
            assert "disk full" in r.text or "RuntimeError" in r.text

        # ``corpus_dir`` doit être nettoyé (sinon fuite disque sur
        # tous les uploads ratés).
        assert list(uploads_dir.iterdir()) == [], (
            "corpus_dir aurait dû être supprimé après l'erreur"
        )


class TestIsPathAllowedException:
    def test_value_error_on_compare_continues_to_next_root(
        self, monkeypatch,
    ) -> None:
        """``Path.is_relative_to`` lève ``ValueError`` quand on
        compare des paths de drives différents (Windows) ou autres
        cas pathologiques.  Le helper doit continuer à itérer
        plutôt que de planter."""
        from picarones.interfaces.web.routers import corpus as corpus_router

        class RaisingPath:
            """Fake Path qui lève sur ``__eq__``/``is_relative_to``."""

            def __eq__(self, other):
                raise ValueError("simulated path comparison error")

            def is_relative_to(self, other):
                raise ValueError("simulated")

        # Premier root lève → continue ; deuxième root match.
        from pathlib import Path as RealPath

        target = RealPath("/tmp")
        monkeypatch.setattr(
            corpus_router,
            "_BROWSE_ROOTS",
            [RaisingPath(), target],
        )
        assert corpus_router._is_path_allowed(target) is True

    def test_no_match_returns_false(self, monkeypatch) -> None:
        from pathlib import Path as RealPath

        from picarones.interfaces.web.routers import corpus as corpus_router

        # ``_BROWSE_ROOTS`` ne contient que des paths qui ne
        # contiennent pas ``/totally/unrelated``.
        monkeypatch.setattr(
            corpus_router,
            "_BROWSE_ROOTS",
            [RealPath("/var/picarones-uploads-test-only")],
        )
        assert corpus_router._is_path_allowed(
            RealPath("/totally/unrelated"),
        ) is False


class TestMultipartDedupe:
    """Audit P0.5 — deux fichiers multipart de même basename ne
    doivent pas s'écraser silencieusement (perte de données /
    mauvaise association image-GT)."""

    def test_unique_name_unchanged(self) -> None:
        from picarones.interfaces.web.routers.corpus import _dedupe_name

        assert _dedupe_name("photo.png", set()) == "photo.png"

    def test_duplicate_gets_suffix_not_overwrite(self) -> None:
        from picarones.interfaces.web.routers.corpus import _dedupe_name

        seen: set[str] = set()
        n1 = _dedupe_name("photo.png", seen)
        seen.add(n1)
        n2 = _dedupe_name("photo.png", seen)
        seen.add(n2)
        n3 = _dedupe_name("photo.png", seen)
        assert n1 == "photo.png"
        assert n2 == "photo_1.png"
        assert n3 == "photo_2.png"
        assert len({n1, n2, n3}) == 3  # aucun écrasement

    def test_compound_extension_preserved(self) -> None:
        from picarones.interfaces.web.routers.corpus import _dedupe_name

        # ``.gt.txt`` : partition('.') garde la 1re extension + reste.
        assert _dedupe_name("doc.gt.txt", {"doc.gt.txt"}) == "doc_1.gt.txt"
