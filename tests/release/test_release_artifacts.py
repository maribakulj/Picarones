"""Tests Sprint A9 — pipeline de release et artefacts.

Items M-5, M-6, m-15, m-16 de l'audit institutional-readiness-2026-05.

Ces tests valident le **contrat de release** sans déclencher de
build réel (qui nécessiterait Docker buildx + accès PyPI). Ils
vérifient que les fichiers de configuration sont cohérents et que
les workflows existent.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# M-5 — setuptools_scm + version dynamique
# ---------------------------------------------------------------------------


def test_pyproject_uses_dynamic_version() -> None:
    """``pyproject.toml`` doit déclarer ``version`` en dynamique
    (résolu par setuptools_scm) plutôt qu'en dur."""
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    # ``version = "1.0.0"`` ne doit plus apparaître au scope ``[project]``.
    project_block = re.search(
        r"\[project\](.*?)(?=\n\[)",
        pyproject,
        re.DOTALL,
    )
    assert project_block is not None
    block = project_block.group(1)
    assert 'dynamic = ["version"]' in block, (
        "[project] doit avoir dynamic = [\"version\"] (Sprint A9 M-5)"
    )
    # Pas de ligne ``version = "..."`` codée en dur dans [project].
    assert not re.search(r'^version\s*=\s*"', block, re.MULTILINE), (
        '[project] ne doit pas avoir version = "..." en dur — '
        'incompatible avec setuptools_scm.'
    )


def test_setuptools_scm_configured() -> None:
    """``[tool.setuptools_scm]`` doit exister avec ``write_to`` pointant
    vers ``picarones/_version.py``."""
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[tool.setuptools_scm]" in pyproject
    assert 'write_to = "picarones/_version.py"' in pyproject
    # Politique : pas de ``+local`` dans la version (problématique pour
    # PyPI qui rejette les locals).
    assert 'local_scheme = "no-local-version"' in pyproject


def test_build_system_includes_setuptools_scm() -> None:
    """``[build-system].requires`` doit inclure setuptools_scm."""
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    build_block = re.search(
        r"\[build-system\](.*?)(?=\n\[)",
        pyproject,
        re.DOTALL,
    )
    assert build_block is not None
    block = build_block.group(1)
    assert "setuptools_scm" in block


def test_picarones_version_resolves() -> None:
    """``picarones.__version__`` doit être lisible et au format PEP 440."""
    import picarones

    v = picarones.__version__
    assert isinstance(v, str)
    assert len(v) > 0
    # PEP 440 : commence par X.Y.Z
    assert re.match(r"^\d+\.\d+", v), f"Version mal formée : {v!r}"


# ---------------------------------------------------------------------------
# M-5 + M-6 — Workflow release.yml
# ---------------------------------------------------------------------------


def test_release_workflow_exists() -> None:
    """``.github/workflows/release.yml`` doit exister."""
    f = REPO_ROOT / ".github" / "workflows" / "release.yml"
    assert f.exists(), "release.yml manquant — pipeline release non automatisé"


def test_release_workflow_triggers_on_tag() -> None:
    """Le workflow doit se déclencher sur push d'un tag ``v*.*.*``."""
    f = REPO_ROOT / ".github" / "workflows" / "release.yml"
    data = yaml.safe_load(f.read_text(encoding="utf-8"))
    # YAML parse ``on`` en bool True — utiliser ``True`` ou la clé string.
    triggers = data.get(True) or data.get("on") or {}
    assert "push" in triggers
    push = triggers["push"]
    assert "tags" in push
    tags = push["tags"]
    # Au moins un pattern qui matche v*.*.*
    assert any("v*" in str(t) for t in tags)


def test_release_workflow_uses_oidc() -> None:
    """Le workflow doit utiliser OIDC trust pour PyPI (pas de token long-lived)."""
    f = REPO_ROOT / ".github" / "workflows" / "release.yml"
    text = f.read_text(encoding="utf-8")
    # Vérifie que ``id-token: write`` est présent pour les jobs publish
    assert "id-token: write" in text
    # Vérifie que pypi-publish est utilisé (gh-action-pypi-publish)
    assert "pypa/gh-action-pypi-publish" in text


def test_release_workflow_publishes_to_ghcr() -> None:
    """Le workflow doit construire et pousser une image multi-arch
    sur ghcr.io."""
    f = REPO_ROOT / ".github" / "workflows" / "release.yml"
    text = f.read_text(encoding="utf-8")
    assert "ghcr.io/" in text
    assert "linux/amd64,linux/arm64" in text
    assert "docker/build-push-action" in text


def test_release_workflow_creates_github_release() -> None:
    """Le workflow doit créer une GitHub Release avec les artefacts."""
    f = REPO_ROOT / ".github" / "workflows" / "release.yml"
    text = f.read_text(encoding="utf-8")
    assert "softprops/action-gh-release" in text or "create-release" in text


# ---------------------------------------------------------------------------
# m-15 — picarones.spec (PyInstaller) sans hiddenimports manuels
# ---------------------------------------------------------------------------


def test_pyinstaller_spec_uses_collect_submodules() -> None:
    """``picarones.spec`` doit utiliser ``collect_submodules`` au lieu
    d'une liste manuelle d'imports cachés."""
    f = REPO_ROOT / "picarones.spec"
    if not f.exists():
        pytest.skip("picarones.spec absent — release PyInstaller non utilisée")
    text = f.read_text(encoding="utf-8")
    assert "collect_submodules" in text, (
        "picarones.spec doit utiliser PyInstaller.utils.hooks.collect_submodules "
        "pour auto-détecter les imports — la liste manuelle dérivait silencieusement."
    )
    assert 'collect_submodules("picarones")' in text


def test_pyinstaller_spec_no_obsolete_paths() -> None:
    """``picarones.spec`` ne doit plus référencer les anciens chemins
    qui n'existent plus depuis le refactor Cercle 1/2/3 (Sprint 33)."""
    f = REPO_ROOT / "picarones.spec"
    if not f.exists():
        pytest.skip("picarones.spec absent")
    text = f.read_text(encoding="utf-8")
    obsolete = [
        "picarones.core.runner",  # → measurements.runner
        "picarones.core.statistics",  # → measurements.statistics
        "picarones.core.confusion",  # → measurements.confusion
        "picarones.importers.iiif",  # → extras.importers.iiif
    ]
    for path in obsolete:
        assert path not in text, (
            f"Chemin obsolète référencé : {path}. "
            "Refactor Sprint 33 a déplacé les modules — collect_submodules "
            "résout automatiquement."
        )


# ---------------------------------------------------------------------------
# m-16 — Extras placeholders retirés
# ---------------------------------------------------------------------------


def test_no_empty_extras_placeholders() -> None:
    """Les extras ``[historical]`` et ``[importers]`` qui valaient
    ``[]`` ont été retirés (Sprint A9 m-16)."""
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    # On cherche ``historical = []`` ou ``importers = []`` au scope
    # ``[project.optional-dependencies]``.
    assert not re.search(
        r"^historical\s*=\s*\[\s*\]",
        pyproject,
        re.MULTILINE,
    ), "Placeholder vide ``historical = []`` doit être retiré."
    assert not re.search(
        r"^importers\s*=\s*\[\s*\]",
        pyproject,
        re.MULTILINE,
    ), "Placeholder vide ``importers = []`` doit être retiré."


def test_all_extra_does_not_reference_removed_extras() -> None:
    """L'extra ``all`` ne doit plus référencer ``historical`` /
    ``importers``."""
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    # Cherche la définition de ``all = [...]``
    m = re.search(r"^all\s*=\s*\[(.*?)\]", pyproject, re.MULTILINE | re.DOTALL)
    assert m is not None
    all_block = m.group(1)
    assert "historical" not in all_block
    assert "importers" not in all_block


# ---------------------------------------------------------------------------
# Doc release-process.md
# ---------------------------------------------------------------------------


def test_release_process_doc_exists() -> None:
    """``docs/operations/release-process.md`` doit exister et couvrir
    les sections clés de la procédure."""
    f = REPO_ROOT / "docs" / "operations" / "release-process.md"
    assert f.exists()
    text = f.read_text(encoding="utf-8")
    for section in [
        "Procédure release standard",
        "Versionnement",
        "rollback",
        "OIDC",
    ]:
        assert section.lower() in text.lower(), (
            f"Section manquante dans release-process.md : {section!r}"
        )
