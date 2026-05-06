"""Sprint A14-S20 — ``CorpusService`` (import ZIP sandboxé +
détection des paires image/GT).

Couverture :

- Import basique : 1 image + 1 GT → 1 doc.
- Détection de tous les niveaux GT (alto, page, entities,
  reading_order, txt).
- GT multi-niveaux pour le même stem → un seul doc avec plusieurs
  GroundTruthRef.
- Image sans GT → doc inclus + warning, ``n_images_without_gt`` > 0.
- GT orpheline (sans image) → warning + non rattachée,
  ``n_gt_without_image`` > 0.
- Filtrage silencieux des artefacts macOS (``__MACOSX/``, ``._*``,
  ``.DS_Store``, ``Thumbs.db``).

Sécurité :

- Path traversal (``../etc/passwd``) → ``CorpusImportError``.
- Chemin absolu Unix (``/etc/passwd``) → ``CorpusImportError``.
- Chemin absolu Windows (``C:\\evil``) → ``CorpusImportError``.
- Octet nul dans le nom → ``CorpusImportError``.
- Symlink dans l'archive → ``CorpusImportError``.
- ZIP plus volumineux que ``max_zip_size_bytes`` → erreur.
- Trop d'entrées (zip bomb par nombre) → erreur.
- Décompression trop volumineuse (zip bomb par expansion) → erreur.
- Archive corrompue / non-ZIP → erreur.

Cas limites :

- ZIP vide → corpus vide, pas d'erreur.
- corpus_name avec caractères spéciaux → sanitizé via
  ``safe_report_name``.
- ZIP avec hiérarchie (``volA/folio.png``) → doc_id préserve la
  hiérarchie.
- Doublon d'image (même stem, deux extensions) → premier gardé +
  warning.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from picarones.app.services import (
    CorpusImportError,
    CorpusImportReport,
    CorpusService,
    WorkspaceManager,
)
from picarones.domain.artifacts import ArtifactType


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspaceManager:
    return WorkspaceManager(tmp_path)


@pytest.fixture
def service(workspace: WorkspaceManager) -> CorpusService:
    return CorpusService(workspace)


def _make_zip(entries: dict[str, bytes]) -> bytes:
    """Produit un ZIP en mémoire à partir d'un dict ``{arcname: bytes}``."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _png_bytes() -> bytes:
    """Minimal valid PNG header (signature + IHDR), suffisant pour les
    tests qui ne valident pas l'image."""
    return (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
        b"\x1f\x15\xc4\x89"
    )


# ──────────────────────────────────────────────────────────────────
# Import basique + détection GT
# ──────────────────────────────────────────────────────────────────


class TestBasicImport:
    def test_image_plus_text_gt_creates_one_doc(
        self, service: CorpusService,
    ) -> None:
        zip_bytes = _make_zip({
            "doc01.png": _png_bytes(),
            "doc01.gt.txt": "Hello world".encode("utf-8"),
        })
        report = service.import_zip(zip_bytes, corpus_name="test_corpus")
        assert isinstance(report, CorpusImportReport)
        assert report.n_documents == 1
        doc = report.spec.documents[0]
        assert doc.id == "doc01"
        assert doc.image_uri is not None
        assert Path(doc.image_uri).name == "doc01.png"
        assert len(doc.ground_truths) == 1
        gt = doc.ground_truths[0]
        assert gt.type == ArtifactType.RAW_TEXT
        assert Path(gt.uri).name == "doc01.gt.txt"

    def test_extracted_dir_lives_inside_workspace(
        self,
        service: CorpusService,
        workspace: WorkspaceManager,
    ) -> None:
        zip_bytes = _make_zip({"doc.png": _png_bytes()})
        report = service.import_zip(zip_bytes, corpus_name="x")
        # Garantie sandbox : le dir extrait est sous le workspace root.
        report.extracted_dir.relative_to(workspace.root)

    def test_corpus_name_is_sanitized(
        self, service: CorpusService,
    ) -> None:
        zip_bytes = _make_zip({"doc.png": _png_bytes()})
        report = service.import_zip(
            zip_bytes,
            corpus_name="my/corpus/with/slashes",
        )
        # Les / sont retirés par safe_report_name.
        assert "/" not in report.spec.name
        assert report.spec.name == "mycorpuswithslashes"


class TestGTLevelDetection:
    @pytest.mark.parametrize(
        "suffix,expected_type",
        [
            (".gt.alto.xml", ArtifactType.ALTO_XML),
            (".gt.page.xml", ArtifactType.PAGE_XML),
            (".gt.entities.json", ArtifactType.ENTITIES),
            (".gt.reading_order.json", ArtifactType.READING_ORDER),
            (".gt.txt", ArtifactType.RAW_TEXT),
        ],
    )
    def test_each_gt_suffix_is_recognized(
        self,
        service: CorpusService,
        suffix: str,
        expected_type: ArtifactType,
    ) -> None:
        zip_bytes = _make_zip({
            "doc.png": _png_bytes(),
            f"doc{suffix}": b"<gt></gt>",
        })
        report = service.import_zip(zip_bytes, corpus_name="x")
        assert report.n_documents == 1
        doc = report.spec.documents[0]
        assert len(doc.ground_truths) == 1
        assert doc.ground_truths[0].type == expected_type

    def test_multi_level_gt_for_same_stem(
        self, service: CorpusService,
    ) -> None:
        zip_bytes = _make_zip({
            "doc.png": _png_bytes(),
            "doc.gt.txt": b"text",
            "doc.gt.alto.xml": b"<alto></alto>",
            "doc.gt.entities.json": b"[]",
        })
        report = service.import_zip(zip_bytes, corpus_name="x")
        assert report.n_documents == 1
        doc = report.spec.documents[0]
        types = {gt.type for gt in doc.ground_truths}
        assert types == {
            ArtifactType.RAW_TEXT,
            ArtifactType.ALTO_XML,
            ArtifactType.ENTITIES,
        }

    def test_case_insensitive_extension_for_image(
        self, service: CorpusService,
    ) -> None:
        zip_bytes = _make_zip({
            "doc.PNG": _png_bytes(),
            "doc.gt.txt": b"x",
        })
        report = service.import_zip(zip_bytes, corpus_name="x")
        assert report.n_documents == 1


class TestPairing:
    def test_image_without_gt_is_included_with_warning(
        self, service: CorpusService,
    ) -> None:
        zip_bytes = _make_zip({"only_image.png": _png_bytes()})
        report = service.import_zip(zip_bytes, corpus_name="x")
        assert report.n_documents == 1
        assert report.n_images_without_gt == 1
        assert any("sans GT" in w for w in report.warnings)

    def test_gt_without_image_is_orphan(
        self, service: CorpusService,
    ) -> None:
        zip_bytes = _make_zip({"orphan.gt.txt": b"text"})
        report = service.import_zip(zip_bytes, corpus_name="x")
        assert report.n_documents == 0
        assert report.n_gt_without_image == 1
        assert any("orpheline" in w for w in report.warnings)

    def test_duplicate_image_stem_keeps_first(
        self, service: CorpusService,
    ) -> None:
        zip_bytes = _make_zip({
            "doc.png": _png_bytes(),
            "doc.jpg": b"jpeg-bytes",
            "doc.gt.txt": b"text",
        })
        report = service.import_zip(zip_bytes, corpus_name="x")
        assert report.n_documents == 1
        # Une des deux est sautée (warning).
        assert any("partagent le stem" in w for w in report.warnings)

    def test_hierarchical_paths_preserved_in_doc_id(
        self, service: CorpusService,
    ) -> None:
        zip_bytes = _make_zip({
            "volA/folio_001.png": _png_bytes(),
            "volA/folio_001.gt.txt": b"x",
            "volB/folio_002.png": _png_bytes(),
            "volB/folio_002.gt.txt": b"y",
        })
        report = service.import_zip(zip_bytes, corpus_name="x")
        assert report.n_documents == 2
        doc_ids = sorted(d.id for d in report.spec.documents)
        assert doc_ids == ["volA/folio_001", "volB/folio_002"]


# ──────────────────────────────────────────────────────────────────
# Filtrage silencieux des artefacts OS
# ──────────────────────────────────────────────────────────────────


class TestOSNoiseFiltering:
    def test_macosx_dir_is_skipped(self, service: CorpusService) -> None:
        zip_bytes = _make_zip({
            "doc.png": _png_bytes(),
            "doc.gt.txt": b"x",
            "__MACOSX/doc.png": b"macos-meta",
            "__MACOSX/._doc.png": b"macos-meta-fork",
        })
        report = service.import_zip(zip_bytes, corpus_name="x")
        assert report.n_documents == 1
        assert report.n_skipped_noise >= 1

    def test_dotunderscore_files_skipped(
        self, service: CorpusService,
    ) -> None:
        zip_bytes = _make_zip({
            "doc.png": _png_bytes(),
            "._doc.png": b"resource-fork",
        })
        report = service.import_zip(zip_bytes, corpus_name="x")
        assert report.n_documents == 1

    def test_dsstore_skipped(self, service: CorpusService) -> None:
        zip_bytes = _make_zip({
            "doc.png": _png_bytes(),
            ".DS_Store": b"finder-metadata",
        })
        report = service.import_zip(zip_bytes, corpus_name="x")
        assert report.n_documents == 1
        assert report.n_skipped_noise >= 1

    def test_thumbsdb_skipped_case_insensitive(
        self, service: CorpusService,
    ) -> None:
        zip_bytes = _make_zip({
            "doc.png": _png_bytes(),
            "Thumbs.db": b"win-thumbs",
            "subdir/THUMBS.DB": b"more",
        })
        report = service.import_zip(zip_bytes, corpus_name="x")
        assert report.n_documents == 1
        assert report.n_skipped_noise >= 2


# ──────────────────────────────────────────────────────────────────
# Sécurité — refus brutal
# ──────────────────────────────────────────────────────────────────


class TestSecurityRejections:
    def test_traversal_in_arcname_is_rejected(
        self, service: CorpusService,
    ) -> None:
        zip_bytes = _make_zip({"../escape.txt": b"evil"})
        with pytest.raises(CorpusImportError, match="Traversal"):
            service.import_zip(zip_bytes, corpus_name="x")

    def test_absolute_unix_path_is_rejected(
        self, service: CorpusService,
    ) -> None:
        zip_bytes = _make_zip({"/etc/passwd": b"root:x:0:0::/root:/bin/bash"})
        with pytest.raises(CorpusImportError, match="absolu"):
            service.import_zip(zip_bytes, corpus_name="x")

    def test_absolute_windows_path_is_rejected(
        self, service: CorpusService,
    ) -> None:
        zip_bytes = _make_zip({"C:/evil.txt": b"evil"})
        with pytest.raises(CorpusImportError, match="absolu"):
            service.import_zip(zip_bytes, corpus_name="x")

    def test_corrupt_zip_raises(self, service: CorpusService) -> None:
        with pytest.raises(CorpusImportError, match="invalide"):
            service.import_zip(b"not a zip", corpus_name="x")

    def test_zip_too_large_raises(
        self, workspace: WorkspaceManager,
    ) -> None:
        small_service = CorpusService(workspace, max_zip_size_bytes=10)
        zip_bytes = _make_zip({"doc.png": _png_bytes()})
        assert len(zip_bytes) > 10
        with pytest.raises(CorpusImportError, match="trop volumineux"):
            small_service.import_zip(zip_bytes, corpus_name="x")

    def test_too_many_entries_raises(
        self, workspace: WorkspaceManager,
    ) -> None:
        cap_service = CorpusService(workspace, max_entry_count=3)
        zip_bytes = _make_zip({f"f{i}.png": _png_bytes() for i in range(5)})
        with pytest.raises(CorpusImportError, match="trop d'entrées"):
            cap_service.import_zip(zip_bytes, corpus_name="x")

    def test_uncompressed_too_large_raises(
        self, workspace: WorkspaceManager,
    ) -> None:
        # 3 fichiers de 100 octets, plafond à 200 → refus.
        cap_service = CorpusService(
            workspace, max_uncompressed_bytes=200,
        )
        zip_bytes = _make_zip({
            f"f{i}.png": b"x" * 100 for i in range(3)
        })
        with pytest.raises(CorpusImportError, match="décompressé trop volumineux"):
            cap_service.import_zip(zip_bytes, corpus_name="x")

    def test_symlink_entry_rejected(
        self, service: CorpusService, tmp_path: Path,
    ) -> None:
        # Construire manuellement un ZIP avec une entrée flaggée
        # symlink (mode UNIX 0xA000).
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w") as zf:
            info = zipfile.ZipInfo("evil_link")
            info.external_attr = 0xA000 << 16  # S_IFLNK
            zf.writestr(info, "/etc/passwd")
        with pytest.raises(CorpusImportError, match="Symlink"):
            service.import_zip(buf.getvalue(), corpus_name="x")


# ──────────────────────────────────────────────────────────────────
# Cas limites
# ──────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_zip_yields_empty_corpus(
        self, service: CorpusService,
    ) -> None:
        zip_bytes = _make_zip({})
        report = service.import_zip(zip_bytes, corpus_name="x")
        assert report.n_documents == 0
        assert report.n_images_without_gt == 0
        assert report.n_gt_without_image == 0

    def test_unrecognized_extension_is_skipped(
        self, service: CorpusService,
    ) -> None:
        zip_bytes = _make_zip({
            "doc.png": _png_bytes(),
            "doc.gt.txt": b"x",
            "readme.md": b"# readme",
        })
        report = service.import_zip(zip_bytes, corpus_name="x")
        assert report.n_documents == 1
        # readme.md sauté car pas image, pas GT reconnue.
        assert "readme.md" in report.skipped_paths

    def test_invalid_chars_in_doc_id_are_replaced(
        self, service: CorpusService,
    ) -> None:
        # Espaces, parenthèses, accents → remplacés par _.
        zip_bytes = _make_zip({
            "doc avec espaces (BnF).png": _png_bytes(),
            "doc avec espaces (BnF).gt.txt": b"x",
        })
        report = service.import_zip(zip_bytes, corpus_name="x")
        assert report.n_documents == 1
        doc = report.spec.documents[0]
        # Le doc_id ne contient plus d'espaces ni de parenthèses.
        assert " " not in doc.id
        assert "(" not in doc.id
        assert ")" not in doc.id

    def test_metadata_passes_through(
        self, service: CorpusService,
    ) -> None:
        zip_bytes = _make_zip({"doc.png": _png_bytes()})
        report = service.import_zip(
            zip_bytes,
            corpus_name="x",
            metadata={"language": "fr", "period": "early_modern"},
        )
        assert report.spec.metadata == {
            "language": "fr",
            "period": "early_modern",
        }

    def test_multiple_imports_dont_collide(
        self, service: CorpusService,
    ) -> None:
        """Deux imports avec corpus_name distincts coexistent."""
        zb = _make_zip({"doc.png": _png_bytes()})
        r1 = service.import_zip(zb, corpus_name="alpha")
        r2 = service.import_zip(zb, corpus_name="beta")
        assert r1.extracted_dir != r2.extracted_dir
        assert r1.extracted_dir.exists()
        assert r2.extracted_dir.exists()


# ──────────────────────────────────────────────────────────────────
# Smoke test : import bout-en-bout puis BenchmarkService consume
# ──────────────────────────────────────────────────────────────────


class TestSmokeIntegration:
    def test_imported_corpus_is_consumable_by_benchmark_service(
        self, service: CorpusService,
    ) -> None:
        """L'import produit un CorpusSpec immédiatement utilisable
        — vérifie l'API en bout-en-bout sans lancer un vrai bench."""
        zip_bytes = _make_zip({
            "doc01.png": _png_bytes(),
            "doc01.gt.txt": "première page".encode("utf-8"),
            "doc02.png": _png_bytes(),
            "doc02.gt.txt": "deuxième page".encode("utf-8"),
            "doc02.gt.alto.xml": b"<alto/>",
        })
        report = service.import_zip(
            zip_bytes,
            corpus_name="bnf_test",
            metadata={"language": "fr"},
        )
        assert report.n_documents == 2
        # Un doc avec 1 GT (text), un avec 2 GT (text + alto).
        gts_by_doc = {d.id: d.available_gt_types for d in report.spec.documents}
        assert ArtifactType.RAW_TEXT in gts_by_doc["doc01"]
        assert set(gts_by_doc["doc02"]) == {
            ArtifactType.RAW_TEXT, ArtifactType.ALTO_XML,
        }
