"""Phase 4.4 audit code-quality — interdit les ``pytest.skip("dep
non installée")`` sur des dépendances déclarées **obligatoires**
dans ``pyproject.toml``.

Pattern zombie typique :

.. code-block:: python

    try:
        import click
    except ImportError:
        pytest.skip("click non installé")

Si ``click`` est dans ``[project.dependencies]`` (pas dans
``[project.optional-dependencies]``), cet ``ImportError`` ne peut
jamais se déclencher → le skip est vacuement vrai et le test
n'est jamais exécuté.  L'audit code-quality (2026-05) en a trouvé
**7 occurrences** dans ``tests/integration/test_chantier{4,5}.py``,
toutes sur ``click``.

Ce test scanne ``tests/`` à la recherche de skips qui mentionnent
une dep obligatoire et échoue avec un message clair indiquant
quel test transformer en exécution franche.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = REPO_ROOT / "tests"

#: Liste de noms de packages déclarés en dep obligatoire
#: ``[project.dependencies]``.  Source de vérité :
#: ``pyproject.toml``.  À synchroniser si la liste évolue (rare
#: — les deps obligatoires sont stables par construction).
MANDATORY_DEPS: frozenset[str] = frozenset({
    "click",
    "pydantic",
    "fastapi",
    "uvicorn",
    "lxml",
    "defusedxml",
    "rapidfuzz",
    "jiwer",
    "numpy",
    "pyyaml",
    "annotated_types",
    "typing_extensions",
})

#: ``pytest.skip("<package> non installé")`` ou variantes.  Capture
#: le nom du package à l'intérieur de la chaîne pour le rapporter.
_SKIP_RE = re.compile(
    r"pytest\.skip\s*\(\s*[fr]?[\"']([^\"']*?)\b"
    r"(?P<pkg>[a-zA-Z_][\w\-]*)\b[^\"']*?non installé",
    re.IGNORECASE,
)


def _scan_zombie_skips() -> list[tuple[Path, int, str]]:
    """Scan AST plutôt que regex pour ignorer commentaires et docstrings."""
    import ast

    findings: list[tuple[Path, int, str]] = []
    for path in sorted(TESTS_DIR.rglob("test_*.py")):
        # On ignore ce test lui-même (sinon il se signale).
        if path == Path(__file__):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            # Cherche les appels ``pytest.skip("...")``.
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_pytest_skip = (
                isinstance(func, ast.Attribute)
                and func.attr == "skip"
                and isinstance(func.value, ast.Name)
                and func.value.id == "pytest"
            )
            if not is_pytest_skip or not node.args:
                continue
            first = node.args[0]
            if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
                continue
            msg = first.value
            m = _SKIP_RE.search(f'pytest.skip("{msg}")')
            if not m:
                continue
            pkg = m.group("pkg").lower()
            if pkg in MANDATORY_DEPS:
                findings.append((path, node.lineno, pkg))
    return findings


def test_no_skip_on_mandatory_dependency() -> None:
    """Aucun ``pytest.skip("<dep> non installé")`` ne doit cibler
    une dep obligatoire.

    Si une dep apparaît dans le scan, deux options :

    1. **Recommandée** — la dep est vraiment obligatoire : retirer
       le ``try/except ImportError`` et faire un ``import`` direct.
       Le test plantera franchement si l'environnement est cassé,
       ce qui est le comportement correct (signal opérationnel).
    2. **Exceptionnelle** — la dep est en fait optionnelle (a déménagé
       vers ``[project.optional-dependencies]``) : retirer le nom
       de :data:`MANDATORY_DEPS` ci-dessus.
    """
    zombies = _scan_zombie_skips()
    if zombies:
        lines = "\n".join(
            f"  {p.relative_to(REPO_ROOT)}:{ln} → skip '{pkg} non installé'"
            for p, ln, pkg in zombies
        )
        raise AssertionError(
            "Skips zombies détectés (dep obligatoire = ImportError "
            "impossible) :\n" + lines
            + "\n\nRemplacer le ``try/except ImportError → pytest.skip`` "
            "par un import direct, ou retirer la dep de MANDATORY_DEPS "
            "si elle est devenue optionnelle."
        )


def test_scanner_catches_obvious_zombie_pattern(tmp_path: Path) -> None:
    """Méta-test : le scanner détecte effectivement le pattern.

    Garde-fou contre un regex trop laxiste qui passerait à côté.
    """
    sample = tmp_path / "test_sample.py"
    sample.write_text(
        "import pytest\n"
        "\n"
        "def test_x():\n"
        "    try:\n"
        "        import click\n"
        "    except ImportError:\n"
        "        pytest.skip('click non installé')\n",
        encoding="utf-8",
    )
    matches = list(_SKIP_RE.finditer(sample.read_text(encoding="utf-8")))
    assert len(matches) == 1
    assert matches[0].group("pkg").lower() == "click"
