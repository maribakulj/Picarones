"""Garde-fou contre les modules sans consommateur en production.

Chaque module dans ``picarones/measurements/`` doit être importé par
au moins un fichier de production (hors lui-même, hors ``tests/``).
Sinon le module est *test-only* — sa couverture de test est haute mais
il n'est branché à rien dans le pipeline réel.

Snapshot v1.0.0 (2026-05-02) : **12 modules** dans ``measurements/``
n'ont aucun consommateur direct hors tests :

- ``alto_metrics``, ``baseline_comparison``, ``builtin_metrics``,
  ``cost_projection``, ``equivalence_profile``, ``layout``,
  ``marginal_cost``, ``ner_backends``, ``rare_tokens``,
  ``reading_order``, ``taxonomy_cooccurrence``,
  ``taxonomy_intra_doc``.

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

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PICARONES_DIR = REPO_ROOT / "picarones"
MEASUREMENTS_DIR = PICARONES_DIR / "measurements"

#: Snapshot v1.0.0. Modules de ``picarones/measurements/`` sans
#: consommateur en production. À résorber par paliers.
TEST_ONLY_BASELINE: frozenset[str] = frozenset({
    "alto_metrics",
    "baseline_comparison",
    "builtin_metrics",
    "cost_projection",
    "equivalence_profile",
    "layout",
    "marginal_cost",
    "ner_backends",
    "rare_tokens",
    "reading_order",
    "taxonomy_cooccurrence",
    "taxonomy_intra_doc",
})


def _measurements_modules() -> list[str]:
    return sorted(
        p.stem
        for p in MEASUREMENTS_DIR.glob("*.py")
        if p.stem != "__init__"
    )


def _has_production_consumer(module_name: str) -> bool:
    """True si ``module_name`` est importé par un fichier de production.

    "Production" = sous ``picarones/``, hors le module lui-même.
    On accepte les imports absolus (``from picarones.measurements.X``
    et ``import picarones.measurements.X``) ainsi que les imports
    relatifs depuis le package ``measurements`` (``from .X``).
    """
    own_file = MEASUREMENTS_DIR / f"{module_name}.py"
    absolute_pattern = re.compile(
        rf"\bfrom\s+picarones\.measurements\.{re.escape(module_name)}\b"
        rf"|\bimport\s+picarones\.measurements\.{re.escape(module_name)}\b"
    )
    relative_pattern = re.compile(
        rf"\bfrom\s+\.\s*{re.escape(module_name)}\b"
        rf"|\bfrom\s+\.measurements\.{re.escape(module_name)}\b"
    )
    for path in PICARONES_DIR.rglob("*.py"):
        if path == own_file:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if absolute_pattern.search(text):
            return True
        # Imports relatifs : ne sont valides que depuis l'arbre measurements.
        try:
            path.relative_to(MEASUREMENTS_DIR)
        except ValueError:
            continue
        if relative_pattern.search(text):
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
