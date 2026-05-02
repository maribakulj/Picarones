"""Garde-fou contre les modules sans consommateur en production.

Chaque module dans ``picarones/measurements/`` doit être importé par
au moins un fichier de production (hors lui-même, hors ``tests/``).
Sinon le module est *test-only* — sa couverture de test est haute mais
il n'est branché à rien dans le pipeline réel.

Snapshot v1.0.0 (2026-05-02, recalibré post-audit du 2026-05-02) :
**13 modules** dans ``measurements/`` n'ont aucun consommateur
direct hors tests. La baseline initiale (12 modules) reposait sur
une regex texte qui (a) ne capturait pas la syntaxe
``from picarones.measurements import X`` utilisée dans
``__init__.py`` (3 faux positifs : alto_metrics, builtin_metrics,
reading_order), et (b) capturait à tort les imports DANS DES
DOCSTRINGS (4 faux négatifs : error_absorption, longitudinal,
module_policy, reliability).

Le check est désormais basé sur le module ``ast`` standard de
Python qui ignore correctement le contenu des chaînes/docstrings
et reconnaît toutes les formes d'import valides.

Trois actions possibles, par module :

1. **Câbler** dans le runner ou un renderer (le module devient un
   produit, pas une expérience).
2. **Déplacer** vers ``picarones/extras/`` si c'est expérimental
   et non livré dans le pipeline standard.
3. **Retirer** si c'est mort (le travail reste dans l'historique git).

Test ratchet :

- Tout module ``measurements/X.py`` qui devient test-only sans entrer
  dans la baseline → échec (régression).
- Tout module de la baseline qui gagne un consommateur → échec
  jusqu'à ce que la baseline soit mise à jour pour verrouiller le gain.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PICARONES_DIR = REPO_ROOT / "picarones"
MEASUREMENTS_DIR = PICARONES_DIR / "measurements"

#: Snapshot v1.0.0 (post-audit AST). Modules de
#: ``picarones/measurements/`` sans consommateur en production.
#: À résorber par paliers.
TEST_ONLY_BASELINE: frozenset[str] = frozenset({
    "baseline_comparison",
    "cost_projection",
    "equivalence_profile",
    "error_absorption",
    "layout",
    "longitudinal",
    "marginal_cost",
    "module_policy",
    "ner_backends",
    "rare_tokens",
    "reliability",
    "taxonomy_cooccurrence",
    "taxonomy_intra_doc",
})


def _measurements_modules() -> list[str]:
    return sorted(
        p.stem
        for p in MEASUREMENTS_DIR.glob("*.py")
        if p.stem != "__init__"
    )


def _imports_target_module(node: ast.AST, module_name: str) -> bool:
    """True si ce nœud AST importe ``picarones.measurements.<module_name>``.

    Couvre les 5 syntaxes valides Python :

    - ``import picarones.measurements.X``
    - ``import picarones.measurements.X.sub`` (sous-module)
    - ``from picarones.measurements.X import Y``
    - ``from picarones.measurements import X``
    - ``from picarones.measurements import (X, Y)`` (forme parenthésée)
    """
    target_dotted = f"picarones.measurements.{module_name}"
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name == target_dotted or alias.name.startswith(
                target_dotted + ".",
            ):
                return True
        return False
    if isinstance(node, ast.ImportFrom):
        # ``from picarones.measurements.X import …``
        if node.module == target_dotted:
            return True
        # ``from picarones.measurements import X``
        if node.module == "picarones.measurements":
            for alias in node.names:
                if alias.name == module_name:
                    return True
    return False


def _imports_target_relative(
    node: ast.AST, module_name: str, source_dir: Path,
) -> bool:
    """True si ce nœud AST importe ``module_name`` via un import relatif
    valide depuis le package ``measurements``.

    Couvre :

    - ``from . import X`` (depuis ``measurements/__init__.py`` ou
      tout autre module dans le package).
    - ``from .X import Y``.
    """
    if not isinstance(node, ast.ImportFrom):
        return False
    if node.level < 1:
        return False
    if source_dir != MEASUREMENTS_DIR:
        return False
    # ``from .X import …``
    if node.module == module_name:
        return True
    # ``from . import X``
    if node.module is None:
        for alias in node.names:
            if alias.name == module_name:
                return True
    return False


def _has_production_consumer(module_name: str) -> bool:
    """True si ``module_name`` est importé par un fichier de production.

    "Production" = sous ``picarones/``, hors le module lui-même.

    Le check parse l'AST de chaque fichier (au lieu de grep) pour deux
    raisons :

    1. **Toutes les syntaxes d'import sont reconnues** sans bricolage
       de regex (``from picarones.measurements import X`` était la
       grosse cible manquée par la regex initiale).
    2. **Les chaînes/docstrings ne déclenchent pas de faux positif**
       (un exemple de code dans une docstring ne compte pas comme
       import réel).
    """
    own_file = MEASUREMENTS_DIR / f"{module_name}.py"
    for path in PICARONES_DIR.rglob("*.py"):
        if path == own_file:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if _imports_target_module(node, module_name):
                return True
            if _imports_target_relative(node, module_name, path.parent):
                return True
    return False


def _test_only_modules() -> frozenset[str]:
    return frozenset(
        m for m in _measurements_modules()
        if not _has_production_consumer(m)
    )


def test_no_new_test_only_modules() -> None:
    """Aucun module ne doit devenir test-only sans entrer dans la baseline."""
    current = _test_only_modules()
    new = current - TEST_ONLY_BASELINE
    assert not new, (
        f"\n{len(new)} module(s) de measurements/ sans consommateur en "
        f"production : {sorted(new)}.\n\n"
        "Choisis l'une des trois options :\n"
        "  1. Câble le module dans le runner ou un renderer.\n"
        "  2. Déplace-le sous picarones/extras/ s'il est expérimental.\n"
        "  3. Retire-le si c'est mort.\n\n"
        "En dernier recours, ajoute son nom à TEST_ONLY_BASELINE dans "
        "tests/architecture/test_module_coverage.py — c'est admettre "
        "consciemment qu'il vit hors du pipeline standard."
    )


def test_baseline_modules_still_orphaned() -> None:
    """Si un module de la baseline a gagné un consommateur, lock le gain.

    Force à mettre à jour la baseline pour verrouiller chaque câblage,
    sinon une régression future re-deviendrait test-only sans alerte.
    """
    current = _test_only_modules()
    fixed = TEST_ONLY_BASELINE - current
    assert not fixed, (
        f"\nExcellent : {len(fixed)} module(s) ont gagné un consommateur en "
        f"production : {sorted(fixed)}.\n\n"
        "Retire ces noms de TEST_ONLY_BASELINE dans "
        "tests/architecture/test_module_coverage.py pour verrouiller le gain."
    )
