"""Sprint A14-S1 — A.I.0 P0 : validation des chemins utilisateur.

Tests sur ``picarones.interfaces.web.security.validated_path``,
``validated_prompt_filename`` et ``safe_report_name`` : les helpers
introduits pour bloquer les chemins arbitraires reçus des endpoints
benchmark/run et benchmark/start.

Avant le sprint S1 du rewrite ciblé, l'API web acceptait :

- n'importe quel ``corpus_path`` validé uniquement par ``Path.exists()`` ;
- n'importe quel ``output_dir`` créé par ``Path(req.output_dir).mkdir()`` ;
- n'importe quel ``report_name`` concaténé directement (escape via ``../``) ;
- n'importe quel ``prompt_file`` absolu (vecteur d'exfiltration via LLM).

Les tests ci-dessous font office de filet de sécurité.  Toute évolution
ultérieure de la couche security.py qui ferait régresser ces invariants
est bloquée par cette suite.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from picarones.interfaces.web.security import (
    PathValidationError,
    safe_report_name,
    validated_path,
    validated_prompt_filename,
)


# ──────────────────────────────────────────────────────────────────────
# validated_path
# ──────────────────────────────────────────────────────────────────────


class TestValidatedPath:
    def test_accepts_path_within_allowed_root(self, tmp_path: Path) -> None:
        sub = tmp_path / "corpus_a"
        sub.mkdir()
        result = validated_path(str(sub), allowed_roots=[tmp_path], must_be_dir=True)
        assert result == sub.resolve()

    def test_rejects_path_outside_allowed_roots(self, tmp_path: Path) -> None:
        # /etc/passwd existe sur tout Linux et est clairement hors workspace.
        with pytest.raises(PathValidationError, match="hors zone autorisée"):
            validated_path("/etc/passwd", allowed_roots=[tmp_path])

    def test_rejects_traversal_via_dot_dot(self, tmp_path: Path) -> None:
        sub = tmp_path / "inside"
        sub.mkdir()
        # tmp_path/inside/../../../etc → résolu = /etc → hors zone
        evasion = str(sub / ".." / ".." / ".." / "etc")
        with pytest.raises(PathValidationError, match="hors zone autorisée"):
            validated_path(evasion, allowed_roots=[tmp_path])

    def test_rejects_empty_path(self, tmp_path: Path) -> None:
        with pytest.raises(PathValidationError, match="vide"):
            validated_path("", allowed_roots=[tmp_path])

    def test_rejects_null_byte(self, tmp_path: Path) -> None:
        with pytest.raises(PathValidationError, match="octet nul"):
            validated_path("foo\x00bar", allowed_roots=[tmp_path])

    def test_rejects_when_no_allowed_roots(self, tmp_path: Path) -> None:
        with pytest.raises(PathValidationError, match="Aucune racine autorisée"):
            validated_path(str(tmp_path), allowed_roots=[])

    def test_must_exist_raises_on_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist"
        with pytest.raises(PathValidationError, match="inexistant"):
            validated_path(str(missing), allowed_roots=[tmp_path], must_exist=True)

    def test_must_be_dir_raises_on_file(self, tmp_path: Path) -> None:
        f = tmp_path / "a_file.txt"
        f.write_text("hello")
        with pytest.raises(PathValidationError, match="n'est pas un répertoire"):
            validated_path(str(f), allowed_roots=[tmp_path], must_be_dir=True)

    def test_resolves_symlinks(self, tmp_path: Path) -> None:
        # Si on crée un symlink dans tmp_path qui pointe vers /tmp/ailleurs,
        # ``resolve()`` doit suivre le symlink.  Si la cible est hors zone,
        # on rejette.
        outside = Path(tempfile.mkdtemp(prefix="picarones_outside_"))
        try:
            link = tmp_path / "tricky_link"
            link.symlink_to(outside)
            with pytest.raises(PathValidationError, match="hors zone autorisée"):
                validated_path(str(link), allowed_roots=[tmp_path])
        finally:
            # cleanup
            outside.rmdir()


# ──────────────────────────────────────────────────────────────────────
# safe_report_name
# ──────────────────────────────────────────────────────────────────────


class TestSafeReportName:
    def test_accepts_simple_name(self) -> None:
        assert safe_report_name("rapport_2026") == "rapport_2026"

    def test_strips_path_separators(self) -> None:
        # Les séparateurs sont supprimés silencieusement.
        # ``../etc/passwd`` → ``..etcpasswd``, et ``..`` initial est strippé →
        # ``etcpasswd`` (caractères neutres, pas de chemin).
        result = safe_report_name("../etc/passwd")
        assert "/" not in result
        assert "\\" not in result

    def test_rejects_empty(self) -> None:
        with pytest.raises(PathValidationError, match="vide"):
            safe_report_name("")

    def test_rejects_null_byte(self) -> None:
        with pytest.raises(PathValidationError, match="octet nul"):
            safe_report_name("rapport\x00.html")

    def test_rejects_pure_separators(self) -> None:
        with pytest.raises(PathValidationError, match="invalide"):
            safe_report_name("///")

    def test_rejects_dot_only(self) -> None:
        with pytest.raises(PathValidationError):
            safe_report_name(".")

    def test_truncates_to_max_length(self) -> None:
        long_name = "a" * 500
        assert len(safe_report_name(long_name, max_length=128)) == 128


# ──────────────────────────────────────────────────────────────────────
# validated_prompt_filename
# ──────────────────────────────────────────────────────────────────────


class TestValidatedPromptFilename:
    def test_accepts_builtin_name(self) -> None:
        assert (
            validated_prompt_filename("correction_medieval_french.txt")
            == "correction_medieval_french.txt"
        )

    def test_rejects_absolute_path(self) -> None:
        with pytest.raises(PathValidationError, match="séparateur de chemin"):
            validated_prompt_filename("/etc/passwd")

    def test_rejects_relative_traversal(self) -> None:
        with pytest.raises(PathValidationError):
            validated_prompt_filename("../prompts/secret.txt")

    def test_rejects_dot_dot_inline(self) -> None:
        with pytest.raises(PathValidationError, match="suspect"):
            validated_prompt_filename("foo..bar.txt")

    def test_rejects_windows_separator(self) -> None:
        with pytest.raises(PathValidationError, match="séparateur de chemin"):
            validated_prompt_filename(r"C:\Users\victim\file.txt")

    def test_rejects_dot_prefix(self) -> None:
        with pytest.raises(PathValidationError, match="suspect"):
            validated_prompt_filename(".env")

    def test_rejects_null_byte(self) -> None:
        with pytest.raises(PathValidationError, match="octet nul"):
            validated_prompt_filename("file\x00.txt")

    def test_rejects_control_characters(self) -> None:
        with pytest.raises(PathValidationError, match="caractère de contrôle"):
            validated_prompt_filename("file\x01.txt")

    def test_rejects_empty(self) -> None:
        with pytest.raises(PathValidationError, match="vide"):
            validated_prompt_filename("")
