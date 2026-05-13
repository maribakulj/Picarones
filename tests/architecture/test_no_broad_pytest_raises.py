"""Phase 8 audit code-quality (2026-05) — ``pytest.raises(Exception)``
trop large interdit dans ``tests/``.

Catch-all sur ``Exception`` masque les régressions : un test qui
attend ``FrozenInstanceError`` (champ frozen modifié) mais reçoit
``KeyError`` (mauvais setup de la fixture) passera quand même au
vert.  L'audit avait identifié ~15 sites avec ce pattern, tous
remplacés par la classe d'exception précise :

- ``FrozenInstanceError`` (dataclass frozen)
- ``ValidationError`` (Pydantic frozen ou validation)
- ``PicaronesError`` / sous-classes (erreurs métier)
- etc.

Ce test scanne ``tests/`` (hors fixtures, conftest et ce fichier
lui-même) et refuse toute nouvelle occurrence.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = REPO_ROOT / "tests"


def _scan_broad_raises() -> list[tuple[Path, int]]:
    """Trouve les ``pytest.raises(Exception)`` ou ``pytest.raises(Exception, ...)``.

    Le scan ignore aussi ``pytest.raises(BaseException)`` qui est
    encore plus large (couvre ``KeyboardInterrupt`` etc.).
    """
    findings: list[tuple[Path, int]] = []
    for path in sorted(TESTS_DIR.rglob("test_*.py")):
        if path == Path(__file__):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            # ``pytest.raises(...)``
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "raises"
                and isinstance(func.value, ast.Name)
                and func.value.id == "pytest"
            ):
                continue
            if not node.args:
                continue
            first = node.args[0]
            # ``pytest.raises(Exception)`` ou ``pytest.raises(BaseException)``
            if isinstance(first, ast.Name) and first.id in {"Exception", "BaseException"}:
                findings.append((path, node.lineno))
    return findings


#: Baseline du nombre de ``pytest.raises(Exception)`` dans la suite.
#: La Phase 8 audit code-quality (2026-05) en a éliminé ~15 sites sur
#: les patterns évidents (FrozenInstanceError, ValidationError).
#: Le reste (~20 sites résiduels sur des cas plus subtils — chaînage
#: de validateurs Pydantic, exceptions custom au choix multiple) est
#: laissé pour une PR de polissage dédiée.
#:
#: Test ratchet : le compteur ne peut que **diminuer**.  Pour le
#: faire baisser :
#:
#: 1. Remplacer ``pytest.raises(Exception)`` par la classe précise.
#: 2. Baisser :data:`BROAD_RAISES_BASELINE` du même montant.
BROAD_RAISES_BASELINE = 24


def test_broad_pytest_raises_below_baseline() -> None:
    """Le compteur ``pytest.raises(Exception)`` ne peut que baisser."""
    findings = _scan_broad_raises()
    count = len(findings)
    if count > BROAD_RAISES_BASELINE:
        lines = "\n".join(
            f"  {p.relative_to(REPO_ROOT)}:{ln}"
            for p, ln in findings
        )
        raise AssertionError(
            f"Sites ``pytest.raises(Exception)`` : {count} > "
            f"baseline {BROAD_RAISES_BASELINE}.\n\n"
            + lines
            + "\n\nRemplacer par la classe d'exception précise "
            "attendue (``FrozenInstanceError``, ``ValidationError``, "
            "``PicaronesError``, etc.).  Un catch-all masque les "
            "régressions où une exception différente serait levée."
        )


def test_baseline_must_be_tightened_when_progress_made() -> None:
    """Si le compteur est sous la baseline, abaisser :data:`BROAD_RAISES_BASELINE`."""
    count = len(_scan_broad_raises())
    assert count >= BROAD_RAISES_BASELINE, (
        f"Sites ``pytest.raises(Exception)`` : {count} < baseline "
        f"{BROAD_RAISES_BASELINE}.\n\nMets à jour BROAD_RAISES_BASELINE "
        f"= {count} dans ce fichier (le gain est verrouillé)."
    )
