"""Garde-fou : tout entry point déclaré doit pointer vers un module
réellement importable.

Audit de mai 2026 — ``app.py`` (entry point HuggingFace Spaces)
référençait ``picarones.web.app:app``, paquet supprimé au sprint H.4
mais jamais rebranché.  ``python app.py`` produisait un
``ModuleNotFoundError`` ; le Dockerfile masquait le bug en lançant
``picarones serve`` à la place.

Ce test vérifie qu'au fil du temps, **chaque entry point déclaré**
(``app.py`` racine, ``[project.scripts]`` dans ``pyproject.toml``,
``huggingface``, ``uvicorn``...) pointe vers un module qui
``importlib.import_module()`` accepte.
"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _extract_uvicorn_targets(path: Path) -> list[str]:
    """Extrait les chaînes ``module:attr`` passées à ``uvicorn.run(...)``.

    Plutôt que de regex à la louche, on parse l'AST pour ne capturer
    que les littéraux passés au premier argument positionnel d'un
    appel ``uvicorn.run(...)``.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    targets: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_uvicorn_run = (
            isinstance(func, ast.Attribute)
            and func.attr == "run"
            and isinstance(func.value, ast.Name)
            and func.value.id == "uvicorn"
        )
        if not is_uvicorn_run or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            targets.append(first.value)
    return targets


def _check_target_importable(target: str) -> None:
    """Vérifie que ``module:attr`` est résolvable."""
    if ":" in target:
        module_name, attr = target.split(":", 1)
    else:
        module_name, attr = target, None
    module = importlib.import_module(module_name)
    if attr is not None:
        assert hasattr(module, attr), (
            f"Entry point '{target}' : le module '{module_name}' "
            f"existe mais n'expose pas l'attribut '{attr}'."
        )


def test_hf_spaces_app_py_resolves() -> None:
    """``app.py`` à la racine doit pointer vers un module importable.

    Régression : ``picarones.web.app:app`` (legacy supprimé v2.0)
    → ``picarones.interfaces.web.app:app`` (Phase 0.1 audit code-quality).
    """
    app_py = REPO_ROOT / "app.py"
    if not app_py.exists():
        pytest.skip("app.py absent (déploiement non-HF)")

    targets = _extract_uvicorn_targets(app_py)
    assert targets, (
        f"{app_py} ne contient aucun appel ``uvicorn.run('module:attr', ...)`` "
        f"— le test ne peut pas vérifier la cible."
    )

    for target in targets:
        _check_target_importable(target)


def test_pyproject_console_scripts_resolve() -> None:
    """Chaque ``[project.scripts]`` doit pointer vers un module importable.

    Format pyproject : ``picarones = "picarones.interfaces.cli:cli"``.
    """
    pyproject = REPO_ROOT / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")

    # Sous-section [project.scripts] — capture tout jusqu'à la
    # prochaine section [...] ou fin de fichier.
    match = re.search(
        r"\[project\.scripts\]\s*\n(.*?)(?=^\[|\Z)",
        content,
        re.DOTALL | re.MULTILINE,
    )
    if not match:
        pytest.skip("Aucune section [project.scripts] dans pyproject.toml")

    # Chaque ligne ``key = "module:attr"``.
    script_re = re.compile(r'^\s*[\w_-]+\s*=\s*"([^"]+)"\s*$', re.MULTILINE)
    targets = script_re.findall(match.group(1))
    assert targets, "Section [project.scripts] vide ou mal formatée"

    for target in targets:
        _check_target_importable(target)


def test_no_legacy_picarones_web_references() -> None:
    """``picarones.web`` a été supprimé en v2.0 ; aucune chaîne
    ``picarones.web.app`` ne doit subsister dans les entry points
    ou fichiers de déploiement.
    """
    suspicious_paths = [
        REPO_ROOT / "app.py",
        REPO_ROOT / "Dockerfile",
        REPO_ROOT / "docker-compose.yml",
        REPO_ROOT / "picarones.spec",
    ]
    pattern = re.compile(r"\bpicarones\.web(?:\.|\b)")
    for path in suspicious_paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        match = pattern.search(text)
        assert match is None, (
            f"{path.name} référence encore le paquet legacy 'picarones.web' "
            f"(supprimé v2.0) à la position {match.start()}. "
            f"Remplacer par 'picarones.interfaces.web'."
        )
