"""Sprint S4.9 — couverture directe de ``CorpusService``.

Avant S4 : 0% direct (testé transitivement via les tests web et
les tests S1.5 ZIP slip).

Cible : 85%+ — vérifie le flux import normal, plus quelques cas
limites non couverts par S1.5 (qui se concentrait sur les attaques).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from picarones.app.services.corpus_service import (
    CorpusImportError,
    CorpusImportReport,
    CorpusService,
)
from picarones.app.services.path_security import WorkspaceManager


# ──────────────────────────────────────────────────────────────────────
# Helpers — ZIP minimal valide
# ──────────────────────────────────────────────────────────────────────


_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15\xc4\x89"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
    b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _build_zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


@pytest.fixture
def service(tmp_path: Path) -> CorpusService:
    ws = WorkspaceManager(base_dir=tmp_path)
    return CorpusService(workspace=ws)


# ──────────────────────────────────────────────────────────────────────
# 1. Import normal : 1 image + 1 GT
# ──────────────────────────────────────────────────────────────────────


class TestNormalImport:
    def test_simple_corpus_imports(self, service: CorpusService) -> None:
        zip_bytes = _build_zip({
            "doc01.png": _PNG,
            "doc01.gt.txt": "Bonjour le monde".encode("utf-8"),
        })
        report = service.import_zip(zip_bytes, corpus_name="t")
        assert isinstance(report, CorpusImportReport)
        assert report.n_documents == 1
        assert report.spec.name == "t"
        assert (report.extracted_dir / "doc01.png").exists()
        assert (report.extracted_dir / "doc01.gt.txt").exists()

    def test_two_documents_imported(self, service: CorpusService) -> None:
        zip_bytes = _build_zip({
            "a.png": _PNG,
            "a.gt.txt": b"texte a",
            "b.png": _PNG,
            "b.gt.txt": b"texte b",
        })
        report = service.import_zip(zip_bytes, corpus_name="t")
        assert report.n_documents == 2

    def test_metadata_passed_through(self, service: CorpusService) -> None:
        zip_bytes = _build_zip({"d.png": _PNG, "d.gt.txt": b"x"})
        report = service.import_zip(
            zip_bytes,
            corpus_name="meta_test",
            metadata={"language": "fr", "script": "latin"},
        )
        assert report.spec.metadata.get("language") == "fr"
        assert report.spec.metadata.get("script") == "latin"


# ──────────────────────────────────────────────────────────────────────
# 2. Cas dégradés
# ──────────────────────────────────────────────────────────────────────


class TestDegradedCases:
    def test_image_without_gt_counted_separately(
        self, service: CorpusService,
    ) -> None:
        zip_bytes = _build_zip({
            "with_gt.png": _PNG,
            "with_gt.gt.txt": b"x",
            "no_gt.png": _PNG,  # pas de GT associé
        })
        report = service.import_zip(zip_bytes, corpus_name="t")
        # Le service compte les images orphelines à part.
        assert report.n_images_without_gt >= 1

    def test_gt_without_image_counted_separately(
        self, service: CorpusService,
    ) -> None:
        zip_bytes = _build_zip({
            "doc.png": _PNG,
            "doc.gt.txt": b"x",
            "orphan.gt.txt": b"orphan",
        })
        report = service.import_zip(zip_bytes, corpus_name="t")
        assert report.n_gt_without_image >= 1

    def test_invalid_zip_bytes_raises(self, service: CorpusService) -> None:
        with pytest.raises(CorpusImportError):
            service.import_zip(b"not a zip", corpus_name="t")

    def test_empty_zip_imports_zero_docs(
        self, service: CorpusService,
    ) -> None:
        """Un ZIP vide est accepté (pas d'erreur), mais le report
        annonce 0 documents."""
        zip_bytes = _build_zip({})
        report = service.import_zip(zip_bytes, corpus_name="t")
        assert report.n_documents == 0

    def test_corpus_name_with_traversal_is_handled(
        self, service: CorpusService, tmp_path: Path,
    ) -> None:
        """Un corpus_name avec ``../`` ne doit pas écrire hors du
        workspace.  Soit refusé, soit le path est sanitisé."""
        zip_bytes = _build_zip({"d.png": _PNG, "d.gt.txt": b"x"})
        try:
            report = service.import_zip(zip_bytes, corpus_name="../escape")
        except (CorpusImportError, ValueError):
            return  # Comportement souhaité
        # Si pas de raise, le path doit rester confiné.
        assert tmp_path in report.extracted_dir.resolve().parents


# ──────────────────────────────────────────────────────────────────────
# 3. Limites configurables
# ──────────────────────────────────────────────────────────────────────


class TestLimits:
    def test_too_many_entries_rejected(self, tmp_path: Path) -> None:
        ws = WorkspaceManager(base_dir=tmp_path)
        # Limite à 3 entrées max.
        svc = CorpusService(workspace=ws, max_entry_count=3)

        # ZIP avec 5 entrées → refus.
        entries = {
            f"doc{i:02d}.png": _PNG for i in range(5)
        }
        zip_bytes = _build_zip(entries)
        with pytest.raises(CorpusImportError, match="entrées"):
            svc.import_zip(zip_bytes, corpus_name="t")

    def test_zip_blob_size_limit(self, tmp_path: Path) -> None:
        ws = WorkspaceManager(base_dir=tmp_path)
        # Limite ZIP à 100 octets (artificiellement bas).
        svc = CorpusService(workspace=ws, max_zip_size_bytes=100)

        # Notre ZIP minimal fait > 100 octets.
        zip_bytes = _build_zip({"d.png": _PNG, "d.gt.txt": b"x"})
        with pytest.raises(CorpusImportError):
            svc.import_zip(zip_bytes, corpus_name="t")
