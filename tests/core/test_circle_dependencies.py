"""Garde-fou architectural — direction des dépendances entre cercles.

Sprint A3 du plan de remédiation institutionnelle (renforce B-1, B-2,
B-3 contre toute régression future).

L'architecture en 3 cercles documentée dans
:doc:`docs/architecture.md` impose que les imports aillent **uniquement**
de l'extérieur vers l'intérieur :

::

    Cercle 3 (extras, report, cli, web)
       │
       ▼
    Cercle 2 (measurements, engines, llm, pipelines, modules)
       │
       ▼
    Cercle 1 (core)

Ce module parse l'AST de tous les fichiers ``.py`` dans Cercles 1 et 2
et **échoue** dès qu'un import remontant vers un cercle plus extérieur
est détecté. Le test couvre :

- Imports top-level (``from picarones.report import …``).
- Imports paresseux à l'intérieur des fonctions (le piège classique
  qui a permis la naissance de B-1 et B-2).
- ``import picarones.report.X`` au format module (en plus de
  ``from picarones.report.X import ...``).

Mécanismes d'exception : aucun. Toute violation doit être corrigée en
remontant le code à un cercle approprié, **pas** ajoutée à une
liste d'exceptions.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PICARONES_ROOT = REPO_ROOT / "picarones"


# ---------------------------------------------------------------------------
# Cartographie des cercles
# ---------------------------------------------------------------------------

#: Modules de Cercle 1 (abstractions pures).
CIRCLE_1_PREFIXES: frozenset[str] = frozenset({"picarones.core"})

#: Modules de Cercle 2 (logique métier).
CIRCLE_2_PREFIXES: frozenset[str] = frozenset(
    {
        "picarones.measurements",
        "picarones.engines",
        "picarones.llm",
        "picarones.pipelines",
        "picarones.modules",
    }
)

#: Modules de Cercle 3 (entrées, plugins, rendu).
CIRCLE_3_PREFIXES: frozenset[str] = frozenset(
    {
        "picarones.report",
        "picarones.cli",
        "picarones.web",
        "picarones.extras",
    }
)


def _circle_of(module_dotted: str) -> int:
    """Retourne 1, 2, 3 ou 0 (hors-package) pour un nom de module."""
    if not module_dotted.startswith("picarones"):
        return 0
    if any(module_dotted == p or module_dotted.startswith(p + ".") for p in CIRCLE_1_PREFIXES):
        return 1
    if any(module_dotted == p or module_dotted.startswith(p + ".") for p in CIRCLE_2_PREFIXES):
        return 2
    if any(module_dotted == p or module_dotted.startswith(p + ".") for p in CIRCLE_3_PREFIXES):
        return 3
    return 0


def _file_to_module(path: Path) -> str:
    """Convertit ``picarones/measurements/runner.py`` en
    ``picarones.measurements.runner``."""
    rel = path.relative_to(REPO_ROOT)
    parts = rel.with_suffix("").parts
    # Supprime ``__init__`` final
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


# ---------------------------------------------------------------------------
# Extraction des imports via AST
# ---------------------------------------------------------------------------


def _walk_imports(source: str) -> Iterator[tuple[str, int]]:
    """Yield ``(module_dotted, lineno)`` pour chaque import du fichier,
    qu'il soit top-level ou paresseux dans une fonction.

    Capture :

    - ``import picarones.report.X``        → ``picarones.report.X``
    - ``from picarones.report.X import Y`` → ``picarones.report.X``
    - ``from picarones.report import X``   → ``picarones.report.X`` (Y ignoré
      pour la classification de cercle, mais le préfixe importe).
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0:
                # Imports relatifs ne franchissent jamais de cercle.
                continue
            if node.module is None:
                continue
            yield node.module, node.lineno


# ---------------------------------------------------------------------------
# Collecte des fichiers à auditer
# ---------------------------------------------------------------------------


def _python_files_in(*subpaths: str) -> list[Path]:
    out: list[Path] = []
    for sub in subpaths:
        d = PICARONES_ROOT / sub
        if not d.exists():
            continue
        out.extend(p for p in d.rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(out)


CIRCLE_1_FILES = _python_files_in("core")
CIRCLE_2_FILES = _python_files_in(
    "measurements", "engines", "llm", "pipelines", "modules"
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", CIRCLE_1_FILES, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_circle_1_no_outer_import(path: Path) -> None:
    """Aucun fichier de Cercle 1 ne doit importer Cercle 2 ou 3."""
    source = path.read_text(encoding="utf-8")
    own_module = _file_to_module(path)
    violations: list[tuple[str, int]] = []
    for imported, lineno in _walk_imports(source):
        # Ignorer les imports vers le module lui-même
        if imported == own_module:
            continue
        circle = _circle_of(imported)
        if circle in (2, 3):
            violations.append((imported, lineno))
    assert not violations, (
        f"{path.relative_to(REPO_ROOT)} (Cercle 1) importe vers un cercle "
        f"plus extérieur — violation de la règle d'architecture :\n"
        + "\n".join(f"  ligne {ln}: import {mod}" for mod, ln in violations)
    )


@pytest.mark.parametrize("path", CIRCLE_2_FILES, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_circle_2_no_outer_import(path: Path) -> None:
    """Aucun fichier de Cercle 2 ne doit importer Cercle 3.

    Cercle 2 → Cercle 1 reste autorisé (et même attendu pour les
    abstractions partagées). Cercle 2 → Cercle 2 (entre sous-packages
    measurements/engines/llm/…) est aussi autorisé."""
    source = path.read_text(encoding="utf-8")
    own_module = _file_to_module(path)
    violations: list[tuple[str, int]] = []
    for imported, lineno in _walk_imports(source):
        if imported == own_module:
            continue
        circle = _circle_of(imported)
        if circle == 3:
            violations.append((imported, lineno))
    assert not violations, (
        f"{path.relative_to(REPO_ROOT)} (Cercle 2) importe vers Cercle 3 — "
        f"violation de la règle d'architecture :\n"
        + "\n".join(f"  ligne {ln}: import {mod}" for mod, ln in violations)
        + "\n\nFix: déplacer la logique réutilisable dans Cercle 1, "
        "ou refactorer pour que la dépendance s'inverse."
    )


def test_no_circle_1_file_imports_circle_3() -> None:
    """Méta-test : énumère explicitement les violations Cercle 1 → 3.

    Permet d'avoir un seul échec global lisible si la regex de
    parametrize masque le compte total."""
    total_violations: list[str] = []
    for path in CIRCLE_1_FILES:
        source = path.read_text(encoding="utf-8")
        for imported, lineno in _walk_imports(source):
            if _circle_of(imported) in (2, 3):
                total_violations.append(
                    f"{path.relative_to(REPO_ROOT)}:{lineno} → {imported}"
                )
    assert not total_violations, (
        f"{len(total_violations)} violation(s) totales Cercle 1 → extérieur :\n"
        + "\n".join(total_violations)
    )


def test_no_circle_2_file_imports_circle_3() -> None:
    """Méta-test : énumère explicitement les violations Cercle 2 → 3."""
    total_violations: list[str] = []
    for path in CIRCLE_2_FILES:
        source = path.read_text(encoding="utf-8")
        for imported, lineno in _walk_imports(source):
            if _circle_of(imported) == 3:
                total_violations.append(
                    f"{path.relative_to(REPO_ROOT)}:{lineno} → {imported}"
                )
    assert not total_violations, (
        f"{len(total_violations)} violation(s) totales Cercle 2 → 3 :\n"
        + "\n".join(total_violations)
    )


# ---------------------------------------------------------------------------
# Sanité
# ---------------------------------------------------------------------------


def test_circles_are_not_empty() -> None:
    """Pré-requis : les listes de fichiers ne doivent pas être vides
    (sinon les paramétrisations ne couvrent rien)."""
    assert CIRCLE_1_FILES, "Cercle 1 vide — chemin core/ introuvable."
    assert CIRCLE_2_FILES, "Cercle 2 vide — au moins un sous-package attendu."


def test_circle_classification_examples() -> None:
    """Tests d'auto-validation de ``_circle_of``."""
    assert _circle_of("picarones.core.corpus") == 1
    assert _circle_of("picarones.core.diff_utils") == 1
    assert _circle_of("picarones.measurements.runner") == 2
    assert _circle_of("picarones.engines.tesseract") == 2
    assert _circle_of("picarones.report.generator") == 3
    assert _circle_of("picarones.cli") == 3
    assert _circle_of("picarones.web.app") == 3
    assert _circle_of("picarones.extras.importers.huggingface") == 3
    assert _circle_of("numpy") == 0
    assert _circle_of("picarones") == 0  # le package racine lui-même
