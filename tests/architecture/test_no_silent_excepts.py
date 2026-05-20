"""Sprint S7.4 — invariant CLAUDE.md : pas de ``except Exception: pass``.

CLAUDE.md déclare comme règle stricte :

    **Ne jamais mettre `except Exception: pass`** : remplacer par
    `logger.warning("[module] fonctionnalité dégradée : %s", e)`.

Ce test scanne la base et échoue si un nouveau silent-except est
introduit.  Trois patterns silencieux interdits :

1. ``except Exception: pass`` (le plus visible).
2. ``except Exception: ...`` (Ellipsis comme no-op).
3. ``except Exception as e: pass`` (typo qui ressemble à du log).

Best-effort accepté avec ``logger.debug(...)`` minimum (signal
opérationnel pour ops).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = REPO_ROOT / "picarones"

# Patterns interdits
_SILENT_EXCEPT_RE = re.compile(
    r"except\s+(?:\([^)]*Exception[^)]*\)|Exception)\s*"
    r"(?:as\s+\w+)?\s*:\s*(?:#[^\n]*)?\s*\n"
    r"(\s+)(pass|\.\.\.)\s*$",
    re.MULTILINE,
)


def _scan_silent_excepts() -> list[tuple[str, int, str]]:
    issues: list[tuple[str, int, str]] = []
    for f in PRODUCTION.rglob("*.py"):
        if "__pycache__" in str(f):
            continue
        text = f.read_text(encoding="utf-8")
        for m in _SILENT_EXCEPT_RE.finditer(text):
            line_no = text[: m.start()].count("\n") + 1
            relpath = str(f.relative_to(REPO_ROOT))
            issues.append((relpath, line_no, m.group(2)))
    return issues


def test_no_silent_except_exception_in_production() -> None:
    """Aucun ``except Exception: pass`` dans le code production.

    Le test échoue si un mainteneur introduit un silent except.
    Pour fixer : remplacer ``pass`` par ``logger.warning(...)``
    ou ``logger.debug(...)`` selon la criticité.
    """
    issues = _scan_silent_excepts()
    if issues:
        formatted = "\n".join(
            f"  - {f}:{line} ({kind})" for f, line, kind in issues
        )
        raise AssertionError(
            f"Violations CLAUDE.md règle « pas de "
            f"``except Exception: pass`` » détectées dans "
            f"{len(issues)} emplacement(s) :\n{formatted}\n\n"
            f"Remplacer ``pass`` par "
            f"``logger.warning('[module] feature dégradée : %s', e)`` "
            f"ou ``logger.debug(...)`` pour les best-effort."
        )
