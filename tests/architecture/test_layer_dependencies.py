"""Sprint A14-S3 — règles de dépendance des nouvelles couches.

Le rewrite ciblé (cf. ``docs/roadmap/rewrite-2026.md``) restructure
``picarones/`` en 8 couches.  Ce module **interdit** dès aujourd'hui
qu'un module d'une couche importe une couche plus extérieure ou
une lib externe non autorisée pour sa couche.

::

    domain          (cercle 1, le plus central)
       ▲
    evaluation
       ▲
    pipeline
       ▲
    formats        ┐
    adapters       ├ cercle 3 — implémentations concrètes
    app/services   │
       ▲           │
    interfaces     ┘ cercle 5 — transport (CLI, web)
    reports_v2

Règles encodées (les "couches plus internes" sont autorisées) :

- ``domain``       : stdlib, pydantic, typing_extensions UNIQUEMENT.
- ``evaluation``   : domain + stdlib + numpy + scipy.
- ``pipeline``     : domain + evaluation + stdlib.
- ``formats``      : domain + stdlib + lxml + defusedxml.
- ``adapters``     : domain + pipeline + formats + libs externes.
- ``app``          : domain + evaluation + pipeline + formats + adapters.
- ``interfaces``   : app + libs transport (fastapi, click, ...).
- ``reports_v2``   : domain + evaluation + stdlib + jinja2.

Compatibilité ascendante : ce test ne touche **pas** aux anciens
packages (``picarones.core``, ``picarones.measurements``, etc.) qui
restent gouvernés par ``tests/core/test_circle_dependencies.py``.
Les deux jeux de règles cohabitent pendant le rewrite — le test
historique disparaîtra à la fin du Sprint S22 quand l'ancien code
aura été migré ou supprimé.

Mécanismes d'exception : aucun.  Toute violation se corrige en
remontant le code dans la couche appropriée, **pas** en allongeant
une whitelist.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PICARONES_ROOT = REPO_ROOT / "picarones"


# ---------------------------------------------------------------------------
# Cartographie des couches
# ---------------------------------------------------------------------------

#: Ordre des couches du plus interne au plus externe.  Un module
#: d'une couche peut importer toutes les couches **strictement
#: avant** la sienne (i.e. plus internes), mais jamais l'inverse.
LAYER_ORDER: tuple[str, ...] = (
    "domain",
    "evaluation",
    "pipeline",
    "formats",
    "adapters",
    "app",
    "reports_v2",
    "interfaces",
)


def _layer_index(name: str) -> int:
    return LAYER_ORDER.index(name)


#: Libs externes additionnellement autorisées par couche (au-delà
#: des couches plus internes).  Liste blanche stricte ; tout import
#: hors stdlib qui n'est pas dans cette liste fait échouer le test.
EXTERNAL_ALLOWED: dict[str, frozenset[str]] = {
    "domain": frozenset({"pydantic", "typing_extensions", "annotated_types"}),
    "evaluation": frozenset({
        "pydantic", "typing_extensions", "annotated_types",
        "numpy", "scipy", "jiwer", "rapidfuzz",
    }),
    "pipeline": frozenset({
        "pydantic", "typing_extensions", "annotated_types",
        "numpy", "scipy",
    }),
    "formats": frozenset({
        "pydantic", "typing_extensions", "annotated_types",
        "lxml", "defusedxml", "yaml",
    }),
    # Adapters: tout est permis (libs OCR/LLM/cloud spécifiques).
    "adapters": None,  # type: ignore[dict-item]  # marqueur "*"
    "app": frozenset({
        "pydantic", "typing_extensions", "annotated_types",
        "numpy", "scipy", "jiwer", "yaml", "lxml", "defusedxml",
    }),
    "interfaces": frozenset({
        "pydantic", "typing_extensions", "annotated_types",
        "fastapi", "starlette", "click", "uvicorn",
        "jinja2", "markupsafe",
        "httpx", "anyio", "h11", "httpcore",
        "multipart",
    }),
    "reports_v2": frozenset({
        "pydantic", "typing_extensions", "annotated_types",
        "jinja2", "markupsafe", "yaml",
    }),
}


def _layer_of(file_path: Path) -> str | None:
    """Retourne la couche d'un fichier ``picarones/*.py``, ou None
    s'il appartient à un ancien package non encore migré."""
    rel = file_path.relative_to(PICARONES_ROOT)
    if not rel.parts:
        return None
    top = rel.parts[0]
    if top in LAYER_ORDER:
        return top
    return None


# ---------------------------------------------------------------------------
# Parsing des imports
# ---------------------------------------------------------------------------


def _imports_in_file(path: Path) -> Iterator[tuple[str, int]]:
    """Yields ``(module_dotted, line_no)`` pour chaque ``import`` du fichier.

    Couvre ``import a.b``, ``from a.b import c``, et les imports
    paresseux à l'intérieur de fonctions (``ast.walk`` parcourt
    tout l'AST, pas seulement les statements top-level).
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        pytest.fail(f"{path} : SyntaxError {exc}")
        return  # pragma: no cover
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            # Imports relatifs (``from .. import x``) sont résolus
            # par le runtime — on n'a pas besoin de les vérifier ici
            # tant qu'ils restent dans le même package (et donc la
            # même couche).
            if node.module is None:
                continue
            if node.level > 0:
                # Import relatif : on ignore.
                continue
            yield node.module, node.lineno


def _python_files(root: Path) -> Iterator[Path]:
    for p in root.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        yield p


# ---------------------------------------------------------------------------
# Vérifications
# ---------------------------------------------------------------------------


def _internal_layer(module_dotted: str) -> str | None:
    """Si ``module_dotted`` est un module ``picarones.<layer>...``,
    retourne ``<layer>`` si ``<layer>`` est dans LAYER_ORDER ; sinon
    None (vieux package, hors-couche)."""
    if not module_dotted.startswith("picarones."):
        return None
    parts = module_dotted.split(".")
    if len(parts) < 2:
        return None
    candidate = parts[1]
    return candidate if candidate in LAYER_ORDER else None


def _external_top(module_dotted: str) -> str:
    """Top-level d'un module externe (``numpy.linalg`` → ``numpy``)."""
    return module_dotted.split(".")[0]


def _is_stdlib(top: str) -> bool:
    import sys
    return top in getattr(sys, "stdlib_module_names", set()) or top in {
        "tomllib", "pyexpat",
    }


@pytest.mark.parametrize(
    "layer",
    LAYER_ORDER,
    ids=lambda x: f"layer-{x}",
)
def test_layer_imports_are_legal(layer: str) -> None:
    """Pour chaque module de la couche ``layer``, vérifier que tous
    ses imports remontent vers une couche plus interne (ou égale)
    et que les libs externes utilisées sont dans la whitelist.

    Test trivialement vert tant que la couche est vide ; échoue dès
    qu'on ajoute du code qui viole les règles.
    """
    layer_dir = PICARONES_ROOT / layer
    if not layer_dir.exists():
        pytest.skip(f"Couche {layer} pas encore créée — skip.")

    layer_idx = _layer_index(layer)
    allowed_externals = EXTERNAL_ALLOWED.get(layer)
    violations: list[str] = []

    for path in _python_files(layer_dir):
        for module, lineno in _imports_in_file(path):
            internal = _internal_layer(module)
            if internal is not None:
                # Import vers une couche du nouveau découpage.
                target_idx = _layer_index(internal)
                # Une couche peut importer elle-même ou plus interne.
                if target_idx > layer_idx:
                    violations.append(
                        f"{path.relative_to(REPO_ROOT)}:{lineno} "
                        f"importe '{module}' (couche '{internal}', "
                        f"plus externe que '{layer}')."
                    )
                continue

            if module.startswith("picarones."):
                # Import vers un ancien package (core/measurements/
                # engines/llm/pipelines/modules/report/cli/web/extras).
                # Pendant le rewrite, c'est interdit dans les
                # nouvelles couches : si tu as besoin d'un truc de
                # l'ancien code, déplace-le d'abord (Sprints S9-S11).
                violations.append(
                    f"{path.relative_to(REPO_ROOT)}:{lineno} "
                    f"importe '{module}' (ancien package non migré). "
                    "Une nouvelle couche ne doit pas dépendre de "
                    "l'ancien code — déplacer d'abord."
                )
                continue

            # Import externe.
            top = _external_top(module)
            if _is_stdlib(top):
                continue
            if allowed_externals is None:
                # ``adapters`` accepte tout externe.
                continue
            if top not in allowed_externals:
                violations.append(
                    f"{path.relative_to(REPO_ROOT)}:{lineno} "
                    f"importe '{module}' (lib externe '{top}' non "
                    f"autorisée pour la couche '{layer}'). "
                    f"Whitelist : {sorted(allowed_externals)}."
                )

    assert not violations, (
        f"\nViolations de couche dans '{layer}' "
        f"(plan rewrite-2026 §architecture cible) :\n"
        + "\n".join(f"  - {v}" for v in violations)
        + "\n\nDeux choix :\n"
        "  1. Remonter le code dans la couche correcte.\n"
        "  2. Si la lib externe est légitime, l'ajouter à "
        "EXTERNAL_ALLOWED dans ce fichier (avec justification "
        "explicite dans le commit message)."
    )


def test_layer_order_well_formed() -> None:
    """Méta-test : LAYER_ORDER doit lister chaque couche une fois."""
    assert len(LAYER_ORDER) == len(set(LAYER_ORDER))
    for layer in LAYER_ORDER:
        assert layer in EXTERNAL_ALLOWED, (
            f"Couche '{layer}' déclarée dans LAYER_ORDER mais absente "
            "de EXTERNAL_ALLOWED."
        )


def test_all_new_layer_dirs_exist() -> None:
    """Méta-test : toutes les couches déclarées dans LAYER_ORDER ont
    un répertoire correspondant.  Sinon le test ``test_layer_imports_are_legal``
    skip silencieusement et la règle n'est pas appliquée."""
    missing = [
        layer for layer in LAYER_ORDER
        if not (PICARONES_ROOT / layer).is_dir()
    ]
    assert not missing, (
        f"Couches déclarées sans répertoire correspondant : {missing}.  "
        "Soit créer le répertoire avec son ``__init__.py``, soit "
        "retirer l'entrée de LAYER_ORDER."
    )
