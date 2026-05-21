"""Sprint S3.6 — ``mypy --strict`` doit passer sur ``picarones.domain``.

Avant S3.6 :
- ``[tool.mypy.overrides] module = "picarones.domain.*" strict = true``
  était configuré.
- MAIS le plugin Pydantic n'était pas chargé → ``BaseModel`` traité
  comme ``Any`` → 20 erreurs ``Class cannot subclass "BaseModel"``.
- La CI ne faisait pas tomber sur ces erreurs (faux sentiment de
  sécurité).

Après S3.6 :
- ``[tool.mypy] plugins = ["pydantic.mypy"]`` activé.
- ``pydantic`` ajouté en dépendance core.
- Ce test échoue si mypy strict sur ``domain/`` réintroduit des
  erreurs.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_mypy_strict_on_domain_passes() -> None:
    """``mypy picarones/domain/`` doit retourner 0 erreur.

    Utilise l'API programmatique ``mypy.api.run`` plutôt qu'un
    ``subprocess.run`` : (a) plus rapide (pas de fork), (b) pas de
    parsing de stdout, (c) si mypy est absent, l'erreur est explicite
    (``ImportError``) au lieu d'un échec silencieux du subprocess.
    """
    try:
        from mypy import api as mypy_api
    except ImportError as e:
        pytest.fail(
            f"mypy n'est pas installé — ce test ne peut pas être skippé "
            f"en silence car il verrouille un invariant strict.\n"
            f"Installer via ``pip install -e .[dev]``.  ImportError: {e}"
        )

    # Travailler depuis REPO_ROOT pour que pyproject.toml soit
    # découvert correctement par mypy.
    prev_cwd = Path.cwd()
    try:
        os.chdir(REPO_ROOT)
        stdout, stderr, exit_code = mypy_api.run(["picarones/domain/"])
    finally:
        os.chdir(prev_cwd)

    if exit_code != 0:
        pytest.fail(
            f"mypy strict sur ``picarones/domain`` échoue.\n"
            f"return code: {exit_code}\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr[-500:]}"
        )


def test_mypy_pydantic_plugin_loaded() -> None:
    """Le plugin ``pydantic.mypy`` doit être déclaré dans pyproject.toml.

    Sans ce plugin, les ``class X(BaseModel)`` sont traitées comme
    ``class X(Any)`` et toute la couche ``domain`` est faussement
    typée.  C'est exactement le bug que l'audit a détecté et que ce
    test verrouille."""
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "pydantic.mypy" in pyproject, (
        "Le plugin ``pydantic.mypy`` n'est plus déclaré dans "
        "``[tool.mypy] plugins = [...]``.  Sans ce plugin, "
        "``mypy --strict`` sur ``domain/`` reporte de faux "
        "positifs ``Class cannot subclass BaseModel``."
    )
