"""Sprint A14-S19 — ``WorkspaceManager`` + foyer définitif des helpers.

Vérifie que :

- les 4 helpers (``validated_path``, ``safe_report_name``,
  ``validated_prompt_filename``, ``PathValidationError``) sont
  accessibles depuis ``picarones.app.services.path_security`` et
  re-exportés par ``picarones.app.services``.
- ``picarones.web.security`` continue de les exposer (non-régression
  pour le legacy web).
- ``WorkspaceManager`` :
  - crée un dossier isolé par session (UUID auto ou ``session_id``
    explicite) ;
  - rejette ``base_dir`` inexistant ;
  - ``subpath`` empêche la traversée ``..`` et les chemins absolus
    hors du root ;
  - ``safe_output_path`` sanitize avant join ;
  - ``cleanup`` supprime le workspace, idempotent ;
  - ``__enter__/__exit__`` cleanup automatique en context manager ;
  - deux managers ne se collisionnent pas.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from picarones.app.services import (
    PathValidationError,
    WorkspaceManager,
    safe_report_name,
    validated_path,
    validated_prompt_filename,
)
from picarones.app.services import path_security as _ps_module


# ──────────────────────────────────────────────────────────────────
# Foyer définitif : symboles accessibles
# ──────────────────────────────────────────────────────────────────


class TestDefinitiveHomeExports:
    def test_path_security_exports_helpers(self) -> None:
        assert callable(validated_path)
        assert callable(safe_report_name)
        assert callable(validated_prompt_filename)
        # PathValidationError est une exception.
        assert issubclass(PathValidationError, ValueError)

    def test_path_security_module_has_all_symbols(self) -> None:
        for sym in (
            "PathValidationError",
            "WorkspaceManager",
            "safe_report_name",
            "validated_path",
            "validated_prompt_filename",
        ):
            assert sym in _ps_module.__all__, f"{sym} manquant de __all__"

    def test_legacy_web_security_reexports_helpers(self) -> None:
        """Le legacy ``web.security`` continue d'exposer les 4 symboles
        (re-import, pas de duplication)."""
        from picarones.interfaces.web import security as legacy
        assert legacy.validated_path is validated_path
        assert legacy.safe_report_name is safe_report_name
        assert legacy.validated_prompt_filename is validated_prompt_filename
        assert legacy.PathValidationError is PathValidationError


# ──────────────────────────────────────────────────────────────────
# WorkspaceManager — création
# ──────────────────────────────────────────────────────────────────


class TestWorkspaceCreation:
    def test_creates_root_with_auto_session_id(self, tmp_path: Path) -> None:
        ws = WorkspaceManager(tmp_path)
        assert ws.root.exists()
        assert ws.root.is_dir()
        assert ws.root.parent == tmp_path.resolve()
        # session_id auto : UUID4 hex (32 chars hexa).
        assert len(ws.session_id) == 32
        assert all(c in "0123456789abcdef" for c in ws.session_id)

    def test_creates_root_with_explicit_session_id(
        self, tmp_path: Path,
    ) -> None:
        ws = WorkspaceManager(tmp_path, session_id="bnf_session_001")
        assert ws.session_id == "bnf_session_001"
        assert ws.root.name == "bnf_session_001"

    def test_two_managers_have_distinct_workspaces(
        self, tmp_path: Path,
    ) -> None:
        ws1 = WorkspaceManager(tmp_path)
        ws2 = WorkspaceManager(tmp_path)
        assert ws1.root != ws2.root
        assert ws1.session_id != ws2.session_id

    def test_rejects_nonexistent_base_dir(self, tmp_path: Path) -> None:
        with pytest.raises(PathValidationError, match="inexistant"):
            WorkspaceManager(tmp_path / "does_not_exist")

    def test_rejects_base_dir_that_is_a_file(self, tmp_path: Path) -> None:
        f = tmp_path / "not_a_dir.txt"
        f.write_text("x")
        with pytest.raises(PathValidationError, match="n'est pas un répertoire"):
            WorkspaceManager(f)

    def test_rejects_session_id_with_path_separator(
        self, tmp_path: Path,
    ) -> None:
        with pytest.raises(PathValidationError):
            WorkspaceManager(tmp_path, session_id="../escape")
        with pytest.raises(PathValidationError):
            WorkspaceManager(tmp_path, session_id="evil/sub")

    def test_idempotent_when_session_dir_exists(self, tmp_path: Path) -> None:
        """Si on recrée un manager avec le même session_id, on accepte
        (utile pour reprendre une session interrompue)."""
        ws1 = WorkspaceManager(tmp_path, session_id="resume_me")
        marker = ws1.root / "marker.txt"
        marker.write_text("already_here")
        # Second manager avec même id.
        ws2 = WorkspaceManager(tmp_path, session_id="resume_me")
        assert ws2.root == ws1.root
        # Le marker précédent est conservé.
        assert marker.read_text() == "already_here"


# ──────────────────────────────────────────────────────────────────
# WorkspaceManager.subpath — sandboxing
# ──────────────────────────────────────────────────────────────────


class TestSubpathSandboxing:
    def test_relative_subpath_is_resolved_under_root(
        self, tmp_path: Path,
    ) -> None:
        ws = WorkspaceManager(tmp_path)
        target = ws.subpath("uploads/image.png")
        assert target == ws.root / "uploads" / "image.png"

    def test_absolute_path_inside_root_is_accepted(
        self, tmp_path: Path,
    ) -> None:
        ws = WorkspaceManager(tmp_path)
        # Chemin absolu déjà sous le root.
        absolute = ws.root / "report.html"
        target = ws.subpath(str(absolute))
        assert target == absolute

    def test_relative_traversal_is_rejected(self, tmp_path: Path) -> None:
        ws = WorkspaceManager(tmp_path)
        # ``..`` qui sort du workspace.
        with pytest.raises(PathValidationError, match="hors zone autorisée"):
            ws.subpath("../escape.txt")

    def test_absolute_path_outside_root_is_rejected(
        self, tmp_path: Path,
    ) -> None:
        ws = WorkspaceManager(tmp_path)
        with pytest.raises(PathValidationError, match="hors zone autorisée"):
            ws.subpath("/etc/passwd")

    def test_must_exist_check(self, tmp_path: Path) -> None:
        ws = WorkspaceManager(tmp_path)
        with pytest.raises(PathValidationError, match="inexistant"):
            ws.subpath("missing.txt", must_exist=True)
        # Création puis check OK.
        (ws.root / "ok.txt").write_text("hi")
        target = ws.subpath("ok.txt", must_exist=True)
        assert target.exists()

    def test_must_be_dir_check(self, tmp_path: Path) -> None:
        ws = WorkspaceManager(tmp_path)
        (ws.root / "f.txt").write_text("x")
        with pytest.raises(PathValidationError, match="n'est pas un répertoire"):
            ws.subpath("f.txt", must_be_dir=True)
        (ws.root / "subdir").mkdir()
        target = ws.subpath("subdir", must_be_dir=True)
        assert target.is_dir()

    def test_subpath_with_null_byte_is_rejected(self, tmp_path: Path) -> None:
        ws = WorkspaceManager(tmp_path)
        with pytest.raises(PathValidationError, match="octet nul"):
            ws.subpath("file\x00.txt")

    def test_subpath_empty_is_rejected(self, tmp_path: Path) -> None:
        ws = WorkspaceManager(tmp_path)
        with pytest.raises(PathValidationError, match="vide"):
            ws.subpath("")


# ──────────────────────────────────────────────────────────────────
# WorkspaceManager.safe_output_path
# ──────────────────────────────────────────────────────────────────


class TestSafeOutputPath:
    def test_sanitizes_then_joins(self, tmp_path: Path) -> None:
        ws = WorkspaceManager(tmp_path)
        target = ws.safe_output_path("rapport_2026.html")
        assert target == ws.root / "rapport_2026.html"

    def test_rejects_separators(self, tmp_path: Path) -> None:
        ws = WorkspaceManager(tmp_path)
        # safe_report_name strip /, mais le résultat reste sous root —
        # pas d'erreur, juste un nom nettoyé.
        target = ws.safe_output_path("path/with/slashes.html")
        # Tous les / sont retirés → "pathwithslashes.html".
        assert target == ws.root / "pathwithslashes.html"

    def test_rejects_empty_after_cleaning(self, tmp_path: Path) -> None:
        ws = WorkspaceManager(tmp_path)
        with pytest.raises(PathValidationError, match="invalide après nettoyage"):
            ws.safe_output_path("///")


# ──────────────────────────────────────────────────────────────────
# WorkspaceManager.cleanup
# ──────────────────────────────────────────────────────────────────


class TestCleanup:
    def test_cleanup_removes_workspace(self, tmp_path: Path) -> None:
        ws = WorkspaceManager(tmp_path)
        (ws.root / "file.txt").write_text("data")
        (ws.root / "sub").mkdir()
        (ws.root / "sub" / "nested.txt").write_text("nested")
        assert ws.root.exists()
        ws.cleanup()
        assert not ws.root.exists()

    def test_cleanup_is_idempotent(self, tmp_path: Path) -> None:
        ws = WorkspaceManager(tmp_path)
        ws.cleanup()
        # Deuxième cleanup ne lève pas.
        ws.cleanup()

    def test_cleanup_does_not_touch_base_dir(self, tmp_path: Path) -> None:
        # Une autre session dans le même base_dir doit survivre.
        ws1 = WorkspaceManager(tmp_path, session_id="alpha")
        ws2 = WorkspaceManager(tmp_path, session_id="beta")
        (ws1.root / "a.txt").write_text("a")
        (ws2.root / "b.txt").write_text("b")
        ws1.cleanup()
        assert not ws1.root.exists()
        # ws2 et le base_dir restent intacts.
        assert ws2.root.exists()
        assert (ws2.root / "b.txt").read_text() == "b"
        assert tmp_path.exists()


# ──────────────────────────────────────────────────────────────────
# Context manager
# ──────────────────────────────────────────────────────────────────


class TestContextManager:
    def test_enter_exit_cleans_up(self, tmp_path: Path) -> None:
        with WorkspaceManager(tmp_path) as ws:
            (ws.root / "scratch.txt").write_text("ephemeral")
            saved_root = ws.root
            assert saved_root.exists()
        assert not saved_root.exists()

    def test_enter_returns_self(self, tmp_path: Path) -> None:
        ws = WorkspaceManager(tmp_path)
        with ws as same_ws:
            assert same_ws is ws

    def test_cleanup_runs_on_exception(self, tmp_path: Path) -> None:
        saved_root = None
        try:
            with WorkspaceManager(tmp_path) as ws:
                saved_root = ws.root
                (ws.root / "f.txt").write_text("x")
                raise RuntimeError("simulated failure")
        except RuntimeError:
            pass
        assert saved_root is not None
        assert not saved_root.exists()


# ──────────────────────────────────────────────────────────────────
# Régression : helpers via le foyer définitif et le legacy alias
# ──────────────────────────────────────────────────────────────────


class TestHelperFunctionalRegression:
    """Ces tests reproduisent un sous-ensemble des assertions du test
    historique S1 pour vérifier que la migration n'a rien cassé."""

    def test_validated_path_rejects_traversal(self, tmp_path: Path) -> None:
        with pytest.raises(PathValidationError, match="hors zone autorisée"):
            validated_path("../escape", allowed_roots=[tmp_path])

    def test_validated_path_rejects_null_byte(self, tmp_path: Path) -> None:
        with pytest.raises(PathValidationError, match="octet nul"):
            validated_path("foo\x00", allowed_roots=[tmp_path])

    def test_validated_path_accepts_in_root(self, tmp_path: Path) -> None:
        target = tmp_path / "ok.txt"
        target.write_text("x")
        result = validated_path(str(target), allowed_roots=[tmp_path])
        assert result == target.resolve()

    def test_safe_report_name_sanitizes(self) -> None:
        assert safe_report_name("rapport.html") == "rapport.html"
        assert safe_report_name("x/y/z.html") == "xyz.html"
        with pytest.raises(PathValidationError):
            safe_report_name("\x00")

    def test_safe_report_name_truncates(self) -> None:
        assert len(safe_report_name("a" * 500, max_length=64)) == 64

    def test_validated_prompt_filename_rejects_separator(self) -> None:
        with pytest.raises(PathValidationError, match="séparateur"):
            validated_prompt_filename("../etc/passwd")

    def test_validated_prompt_filename_rejects_dot_prefix(self) -> None:
        with pytest.raises(PathValidationError, match="suspect"):
            validated_prompt_filename(".env")

    def test_validated_prompt_filename_accepts_simple_name(self) -> None:
        assert validated_prompt_filename("ocr_correction.txt") == \
            "ocr_correction.txt"
