"""Garde-fou contre les modules sans consommateur en production.

Chaque module dans ``picarones/measurements/`` doit être importé par
au moins un fichier de production (hors lui-même, hors ``tests/``).
Sinon le module est *test-only* — sa couverture de test est haute mais
il n'est branché à rien dans le pipeline réel.

Snapshot v1.0.0 (2026-05-02, recalibré post-audit du 2026-05-02) :
**0 module test-only** après le sprint « câblage des 13 modules
test-only ». L'historique :

- 12 modules (initial v1.0.0) : regex texte buggy.
- 13 modules (audit AST) : 3 faux positifs sortis (alto_metrics,
  builtin_metrics, reading_order — déjà importés en
  ``__init__.py``) + 4 faux négatifs ajoutés (error_absorption,
  longitudinal, module_policy, reliability — détectés à tort
  comme consommés via des imports DANS DES DOCSTRINGS).
- **0 module** (sprint « câblage des modules test-only »,
  mai 2026) : 4 modules réellement câblés dans le rapport HTML
  (``rare_tokens``, ``taxonomy_cooccurrence``, ``taxonomy_intra_doc``,
  ``marginal_cost`` via ``picarones/report/report_data/extra_metrics.py``)
  + 9 modules ajoutés explicitement aux imports de
  ``picarones/measurements/__init__.py`` (avec ``# noqa: F401`` et
  justification individuelle de leur scope hors-runner).

Le check est basé sur le module ``ast`` standard de Python qui
ignore correctement le contenu des chaînes/docstrings et reconnaît
toutes les formes d'import valides.

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
import functools
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PICARONES_DIR = REPO_ROOT / "picarones"
MEASUREMENTS_DIR = PICARONES_DIR / "measurements"

#: Snapshot post-sprint « câblage des 13 modules test-only ».
#: **Zéro module** test-only : tous sont consommés en production,
#: soit via un appel automatique dans le rapport HTML
#: (``picarones/report/report_data/extra_metrics.py``), soit via
#: l'API publique du package (imports explicites avec directive
#: de fin de ligne ``noqa F401`` dans
#: ``picarones/measurements/__init__.py``).
TEST_ONLY_BASELINE: frozenset[str] = frozenset({
    # Lot D — la majorité des entrées historiques (shims
    # ``measurements/X.py`` vers ``evaluation/metrics/X``) ont été
    # supprimées au Lot D, donc retirées de cette baseline.  Ne
    # restent que les modules qui n'ont pas de canonique migré et
    # dont le seul consommateur production est désormais hors
    # ``picarones/`` (renderer canonique qui consomme le canonique
    # directement, mais module legacy gardé pour les tests).
    "numerical_sequences_hooks",
    # Sprint D.6.b du plan v2.0 — le sous-package
    # ``measurements.runner`` a été supprimé.  ``builtin_hooks``
    # était son consommateur direct (registre des hooks de
    # métriques) ; sans le runner, il n'a plus de consommateur
    # production.  Suppression / migration prévue en Sprint E
    # (migration des hooks vers ``evaluation/metric_hooks/``).
    "builtin_hooks",
    # Sprint E.2 du plan v2.0 — module ``measurements.searchability``
    # est devenu un shim après son déplacement vers
    # ``evaluation/metrics/searchability``.  Le shim garde son entrée
    # ici pour que le scanner ne crie pas tant qu'il existe.
    "searchability",
    # Sprint E.3 du plan v2.0 — module ``measurements.metrics`` est
    # devenu un shim après son déplacement vers
    # ``evaluation/metrics/text_metrics``.  Le shim n'a plus de
    # consommateur production (les 3 callers sont migrés).
    "metrics",
    # Sprint E.4 du plan v2.0 — modules ``hooks`` migrés vers
    # ``evaluation/metrics/`` ; les shims n'ont plus de consommateur
    # production.
    "philological_hooks",
    "readability_hooks",
    "searchability_hooks",
    # Sprint E.5 du plan v2.0 — derniers shims (history,
    # robustness) sans consommateur production direct.
    "history",
    "robustness",
})


def _measurements_modules() -> list[str]:
    """Modules et sous-packages exposés par ``picarones/measurements/``.

    Inclut :

    - Les fichiers ``*.py`` au top-level (hors ``__init__.py``).
    - Les sous-packages, c'est-à-dire les sous-dossiers contenant un
      ``__init__.py`` (ex: ``narrative/``, ``statistics/`` après le
      sprint de découpage de ``statistics.py``).

    L'ancienne version ne listait que les ``*.py`` ; les sous-packages
    devenaient invisibles au test → couverture perdue dès qu'un module
    était éclaté en sous-package.
    """
    modules: set[str] = {
        p.stem
        for p in MEASUREMENTS_DIR.glob("*.py")
        if p.stem != "__init__"
    }
    for sub in MEASUREMENTS_DIR.iterdir():
        if sub.is_dir() and (sub / "__init__.py").exists():
            modules.add(sub.name)
    return sorted(modules)


def _imports_target_module(node: ast.AST, module_name: str) -> bool:
    """True si ce nœud AST importe ``picarones.measurements.<module_name>``.

    Couvre les 6 syntaxes valides Python (essentielles quand X peut
    être un module ``X.py`` OU un sous-package ``X/``) :

    - ``import picarones.measurements.X``
    - ``import picarones.measurements.X.sub`` (sous-module)
    - ``from picarones.measurements.X import Y``
    - ``from picarones.measurements.X.sub import Y`` (sous-sous-module
      d'un sous-package — ex: ``from .statistics.wilcoxon import …``)
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
        # ``from picarones.measurements.X import …`` ou
        # ``from picarones.measurements.X.sub import …`` (sous-package).
        if node.module == target_dotted or (
            node.module is not None
            and node.module.startswith(target_dotted + ".")
        ):
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
    qui pointe vers ``picarones/measurements/<module_name>``.

    Couvre les imports relatifs depuis n'importe quel sous-dossier du
    package ``measurements`` (y compris ``measurements/narrative/`` et
    ``measurements/narrative/detectors/``) :

    - ``from . import X`` (level=1) depuis ``measurements/foo.py``.
    - ``from .X import Y`` (level=1, module=X) depuis le même.
    - ``from .. import X`` (level=2) depuis ``measurements/sub/foo.py``.
    - ``from ..X import Y`` (level=2, module=X) depuis le même.
    - Idem pour level=3 et au-delà depuis sous-sous-packages.

    L'ancien check ``source_dir == MEASUREMENTS_DIR`` ratait tous les
    imports relatifs depuis les sous-packages — bombe à retardement
    qui devient critique dès qu'un sous-package importe un voisin.
    """
    if not isinstance(node, ast.ImportFrom):
        return False
    if node.level < 1:
        return False
    # Remonter ``node.level - 1`` niveaux pour résoudre le package cible.
    # Pour ``from . import X`` (level=1) on reste dans ``source_dir`` ;
    # pour ``from ..X import Y`` (level=2) on remonte d'un niveau ; etc.
    target_dir = source_dir
    for _ in range(node.level - 1):
        target_dir = target_dir.parent
    if target_dir != MEASUREMENTS_DIR:
        return False
    # ``from .X import …`` ou ``from ..X import …``
    if node.module == module_name:
        return True
    # ``from . import X`` ou ``from .. import X``
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


@functools.cache
def _test_only_modules() -> frozenset[str]:
    """Retourne les modules de ``measurements/`` sans consommateur prod.

    Mémoïsée par ``functools.cache`` : les deux tests de ce fichier
    appellent cette fonction (≈ 12 s par appel sur ~200 fichiers
    Python), donc sans cache on parsait l'AST de tout le projet
    deux fois pour rien.
    """
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
