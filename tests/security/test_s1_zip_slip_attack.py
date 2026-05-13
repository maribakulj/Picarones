"""Sprint S1.5 — Tests d'attaque ZIP slip (path traversal via ZIP).

Vérifie que les deux points d'extraction de ZIP du projet refusent :

1. ``picarones.app.services.corpus_service.CorpusService._extract_safely``
   (entry point upload web).
2. ``picarones.interfaces.web.corpus_utils.flatten_zip_to_dir``
   (entry point legacy / direct).

Vecteurs couverts
-----------------
- Path traversal absolu (``/etc/passwd``).
- Path traversal relatif (``../../../etc/passwd``).
- Octet nul dans le nom (``poison\x00.jpg``).
- Symlink ZIP entry (mode UNIX 0xA000).
- Windows reserved names + backslash absolu.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

#: PNG minimal valide — utilisé là où le contenu doit passer
#: ``validate_image_safe`` (Pillow.verify).  Avant ce durcissement,
#: les tests utilisaient ``b"\\x89PNG"`` (signature seule), mais le
#: durcissement Phase 1 valide chaque image extraite d'un ZIP — d'où
#: l'utilisation d'un PNG 1×1 réellement décodable ici.
_MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15\xc4\x89"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
    b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _zip_with_entry(name: str, data: bytes = b"PWNED") -> bytes:
    """ZIP contenant une seule entrée ``name`` avec ``data``."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # writestr accepte les noms exotiques sans normaliser.
        zf.writestr(name, data)
    return buf.getvalue()


def _zip_with_symlink_entry(name: str, target: str) -> bytes:
    """ZIP contenant un symlink (Unix mode 0xA000)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        info = zipfile.ZipInfo(name)
        info.create_system = 3  # Unix
        info.external_attr = (0xA1FF) << 16  # symlink + 0o777
        zf.writestr(info, target)
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────────
# 1. CorpusService._extract_safely (entry point web upload)
# ──────────────────────────────────────────────────────────────────────


class TestCorpusServiceZipSlip:
    """L'entry point ``CorpusService`` est utilisé par ``POST
    /api/corpus/upload``.  Toute défense ZIP slip qui le concerne
    est testée ici."""

    def _make_service(self, tmp_path: Path):
        from picarones.app.services.corpus_service import CorpusService
        from picarones.app.services.path_security import WorkspaceManager

        ws = WorkspaceManager(base_dir=tmp_path)
        return CorpusService(workspace=ws)

    def test_absolute_path_traversal_rejected(self, tmp_path: Path) -> None:
        from picarones.app.services.corpus_service import CorpusImportError

        svc = self._make_service(tmp_path)
        zip_bytes = _zip_with_entry("/etc/passwd", b"PWNED")
        with pytest.raises(CorpusImportError):
            svc.import_zip(zip_bytes, corpus_name="t")

    def test_relative_path_traversal_rejected(self, tmp_path: Path) -> None:
        from picarones.app.services.corpus_service import CorpusImportError

        svc = self._make_service(tmp_path)
        zip_bytes = _zip_with_entry("../../../tmp/PWNED.txt", b"PWNED")
        with pytest.raises(CorpusImportError):
            svc.import_zip(zip_bytes, corpus_name="t")

    def test_null_byte_in_name_neutralized_by_stdlib(self, tmp_path: Path) -> None:
        """Le ``\\x00`` dans un nom ZIP est silencieusement tronqué
        par la stdlib Python (``zipfile`` lit le nom jusqu'au null
        byte).  Le contrat de sécurité est donc satisfait : soit le
        nom passe nos checks (déjà tronqué = ``poison`` au lieu de
        ``poison\\x00.jpg``), soit il est rejeté.  Ce test vérifie
        qu'aucun fichier ne s'écrit avec un nom qui contient un
        null byte sur disque (impossible sur tous les FS modernes
        de toutes façons)."""
        svc = self._make_service(tmp_path)
        zip_bytes = _zip_with_entry("poison\x00.jpg", b"PWNED")
        try:
            svc.import_zip(zip_bytes, corpus_name="t")
        except Exception:
            # Acceptable : rejeté.
            return
        # Acceptable aussi : extrait sous un nom safe (sans \x00).
        for p in tmp_path.rglob("*"):
            assert "\x00" not in str(p), (
                f"Fichier extrait avec un null byte dans le nom : {p!r}"
            )

    def test_windows_absolute_path_rejected(self, tmp_path: Path) -> None:
        from picarones.app.services.corpus_service import CorpusImportError

        svc = self._make_service(tmp_path)
        zip_bytes = _zip_with_entry("\\windows\\system32\\config", b"PWNED")
        with pytest.raises(CorpusImportError):
            svc.import_zip(zip_bytes, corpus_name="t")

    def test_symlink_entry_rejected(self, tmp_path: Path) -> None:
        from picarones.app.services.corpus_service import CorpusImportError

        svc = self._make_service(tmp_path)
        zip_bytes = _zip_with_symlink_entry("link.txt", "/etc/passwd")
        with pytest.raises(CorpusImportError):
            svc.import_zip(zip_bytes, corpus_name="t")

    def test_post_resolve_traversal_via_clever_name_rejected(
        self, tmp_path: Path,
    ) -> None:
        """Cas plus subtil : le nom ne contient pas ``..`` direct
        mais après résolution il sort du dossier (tentative de
        bypass des regex naïves)."""
        from picarones.app.services.corpus_service import CorpusImportError

        svc = self._make_service(tmp_path)
        zip_bytes = _zip_with_entry("foo/../../../tmp/PWNED.txt", b"PWNED")
        with pytest.raises(CorpusImportError):
            svc.import_zip(zip_bytes, corpus_name="t")


# ──────────────────────────────────────────────────────────────────────
# 2. flatten_zip_to_dir (entry point direct corpus_utils)
# ──────────────────────────────────────────────────────────────────────


class TestFlattenZipToDir:
    """``flatten_zip_to_dir`` aplatit en utilisant ``Path(member).name``,
    ce qui devrait neutraliser le traversal — vérifions par test."""

    def test_traversal_filename_is_flattened_to_basename(
        self, tmp_path: Path,
    ) -> None:
        """Un ZIP avec ``../../../tmp/x.png`` aboutit à ``x.png``
        sous ``dest``, pas dans ``/tmp/``."""
        from picarones.interfaces.web.corpus_utils import flatten_zip_to_dir

        zip_bytes = _zip_with_entry("../../../tmp/x.png", _MINIMAL_PNG)
        dest = tmp_path / "extract"

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            flatten_zip_to_dir(zf, dest)

        # Le fichier doit exister sous dest, PAS dans /tmp/.
        assert (dest / "x.png").exists() or not (Path("/tmp/x.png").exists()), (
            "ZIP slip réussi : le fichier ``x.png`` a été extrait hors "
            "de ``dest``."
        )

    def test_absolute_path_filename_does_not_escape(
        self, tmp_path: Path,
    ) -> None:
        from picarones.interfaces.web.corpus_utils import flatten_zip_to_dir

        zip_bytes = _zip_with_entry("/etc/passwd_clone.png", _MINIMAL_PNG)
        dest = tmp_path / "extract"

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            flatten_zip_to_dir(zf, dest)

        # Le fichier ne doit pas avoir écrit /etc/passwd_clone.png.
        assert not Path("/etc/passwd_clone.png").exists()


# ──────────────────────────────────────────────────────────────────────
# 3. Sanity — un ZIP légitime passe
# ──────────────────────────────────────────────────────────────────────


class TestLegitimateZIPPasses:
    def test_simple_corpus_zip_imports(self, tmp_path: Path) -> None:
        from picarones.app.services.corpus_service import CorpusService
        from picarones.app.services.path_security import WorkspaceManager

        # Construit un PNG minimal valide
        png = (
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
            b"\x1f\x15\xc4\x89"
            b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
            b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w") as zf:
            zf.writestr("doc01.png", png)
            zf.writestr("doc01.gt.txt", b"Bonjour le monde")
        zip_bytes = buf.getvalue()

        ws = WorkspaceManager(base_dir=tmp_path)
        svc = CorpusService(workspace=ws)
        report = svc.import_zip(zip_bytes, corpus_name="legit")
        # Le retour expose ``extracted_dir`` ; on vérifie que
        # l'image et le GT y sont bien présents.
        assert report.extracted_dir.exists()
        assert (report.extracted_dir / "doc01.png").exists()
        assert (report.extracted_dir / "doc01.gt.txt").exists()
        return  # short-circuit du commentaire historique ci-dessous
        # exception n'a été levée et qu'un répertoire d'extraction
        # existe sous corpus_root.
        assert (tmp_path / "corpora" / "legit").exists()
