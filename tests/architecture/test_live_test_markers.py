"""Phase 3.5 audit code-quality — les tests dans
``tests/integration/live/`` doivent porter le marker
``@pytest.mark.live`` sur **chacune** de leurs fonctions de test.

Contexte : ``pyproject.toml`` déclare le marker ``live`` comme
« tests d'intégration contre vraie API/binaire (Tesseract,
Anthropic, OpenAI, Mistral) ; exclus par défaut, opt-in via
``pytest -m live`` ».  Le filtre ``addopts = '-m "not live and not
network"'`` les déselectionne au runner par défaut.

Si une fonction dans ``tests/integration/live/`` oublie le marker,
elle s'exécute lors du ``pytest tests/`` standard et :

- échoue sur les runners sans la dep cloud → faux échec CI ;
- consomme du quota API (clé en CI = facture surprise) ;
- introduit une dépendance réseau non documentée.

L'agent d'audit avait flaggé ``test_tesseract_live.py`` comme
« skip top-level inconditionnel ».  Vérification : le skip est en
fait **conditionnel** (``if shutil.which("tesseract") is None``),
ce qui est légitime — un test live qui peut s'exécuter seulement
si le binaire est présent.  Mais le garde-fou ci-dessous évite
qu'une nouvelle fonction de test oublie le marker.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

LIVE_DIR = Path(__file__).resolve().parents[1] / "integration" / "live"


def _test_functions(path: Path) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Liste les fonctions ``test_*`` au top-level d'un fichier."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            out.append((node.name, node))
    return out


def _has_live_marker(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for deco in fn.decorator_list:
        # ``@pytest.mark.live`` ou ``@pytest.mark.live(reason=...)``
        if isinstance(deco, ast.Attribute) and deco.attr == "live":
            return True
        if isinstance(deco, ast.Call) and isinstance(deco.func, ast.Attribute) and deco.func.attr == "live":
            return True
    return False


def _live_test_files() -> list[Path]:
    if not LIVE_DIR.exists():
        return []
    return [
        p for p in sorted(LIVE_DIR.glob("test_*.py"))
        if p.name != "__init__.py" and p.name != "conftest.py"
    ]


@pytest.mark.parametrize("path", _live_test_files(), ids=lambda p: p.name)
def test_every_function_in_live_dir_has_live_marker(path: Path) -> None:
    """Chaque ``test_*`` dans ``tests/integration/live/`` porte ``@pytest.mark.live``.

    Sinon le test peut s'exécuter en CI standard et casser sur
    l'absence de clé API / binaire externe.
    """
    missing: list[str] = []
    for name, fn in _test_functions(path):
        if not _has_live_marker(fn):
            missing.append(f"  {path.name}:{fn.lineno} :: {name}")

    assert not missing, (
        f"Fonctions dans {LIVE_DIR.name}/ sans ``@pytest.mark.live`` :\n"
        + "\n".join(missing)
        + "\n\nAjouter ``@pytest.mark.live`` au-dessus de chaque test "
        "qui hit une API/un binaire externe — sinon le test "
        "s'exécute sans opt-in et peut faire échouer le CI standard."
    )
