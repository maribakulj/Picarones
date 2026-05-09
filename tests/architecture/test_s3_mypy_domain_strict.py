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

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_mypy_strict_on_domain_passes() -> None:
    """``mypy picarones/domain/`` doit retourner 0 erreur."""
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "picarones/domain/"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )
    if result.returncode != 0:
        pytest.fail(
            f"mypy strict sur ``picarones/domain`` échoue.\n"
            f"return code: {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr[-500:]}"
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
