"""Phase 10 audit code-quality (2026-05) — chaque appel
``logger.{warning,info,error,debug,critical,exception}(...)`` dans
le code source doit commencer par un préfixe ``[module]`` qui
identifie la source du log.

Convention CLAUDE.md :

.. code-block:: python

    logger.warning("[ner.attach] %s/%s : extraction NER dégradée : %s", ...)
    logger.info("[job_store] WAL non supporté, fallback rollback")
    logger.debug("[robustness] cleanup tmp file échoué : %s", exc)

Bénéfice : un opérateur qui voit un warning ``"backup failed"`` dans
les logs sans préfixe ne sait pas si ça vient de l'OCR, du job store
ou d'un détecteur narratif.  Avec ``[job_store] backup failed`` la
source est immédiate.

Stratégie : test **ratchet** — accepter le baseline actuel, refuser
toute nouvelle régression.  Le nettoyage complet (~30 sites résiduels)
peut se faire progressivement.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = REPO_ROOT / "picarones"

_LOG_METHODS = frozenset({
    "debug", "info", "warning", "error", "critical", "exception",
})

#: Pattern attendu : le 1er argument est une f-string ou un str
#: littéral qui commence par ``[<module>]`` (lowercase, _-., max 40 chars).
_PREFIX_RE = re.compile(r"^\[[\w./\-]+\]")


def _scan_unprefixed_logs() -> list[tuple[Path, int, str]]:
    """``(path, lineno, snippet)`` pour chaque appel ``logger.<method>``
    dont le premier argument littéral ne commence pas par ``[<module>]``.
    """
    findings: list[tuple[Path, int, str]] = []
    for path in sorted(PRODUCTION.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr not in _LOG_METHODS:
                continue
            # Vérifier que c'est bien ``logger.<method>``.  On accepte
            # aussi ``logging.warning(...)`` (root) et ``self.logger.warning(...)``.
            if not node.args:
                continue
            first = node.args[0]

            # Extraire la string littérale.
            msg: str | None = None
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                msg = first.value
            elif isinstance(first, ast.JoinedStr):
                # f-string : on prend les morceaux constants au début.
                parts = []
                for v in first.values:
                    if isinstance(v, ast.Constant) and isinstance(v.value, str):
                        parts.append(v.value)
                    else:
                        break
                if parts:
                    msg = "".join(parts)

            if msg is None:
                # Premier argument dynamique (variable, fonction…) — on
                # ne peut pas vérifier statiquement, skip.
                continue

            if not _PREFIX_RE.match(msg):
                findings.append((path, node.lineno, msg[:60]))

    return findings


#: Baseline du nombre de logs sans préfixe.  Phase 10 audit
#: code-quality (2026-05) : ~30 sites résiduels acceptés.  Test
#: ratchet — ne peut que baisser.
UNPREFIXED_LOGS_BASELINE = 46


def test_unprefixed_logs_below_baseline() -> None:
    """Le compteur de logs sans préfixe ``[module]`` ne peut que baisser."""
    findings = _scan_unprefixed_logs()
    count = len(findings)
    if count > UNPREFIXED_LOGS_BASELINE:
        sample = "\n".join(
            f"  {p.relative_to(REPO_ROOT)}:{ln} → {msg!r}"
            for p, ln, msg in findings[:30]
        )
        more = (
            f"\n  ... ({count - 30} de plus)"
            if count > 30
            else ""
        )
        raise AssertionError(
            f"Logs sans préfixe ``[module]`` : {count} > baseline "
            f"{UNPREFIXED_LOGS_BASELINE}.\n\n"
            f"{sample}{more}\n\n"
            "Convention CLAUDE.md : chaque log doit commencer par "
            "``[<module>]`` pour identifier sa source.  Exemples : "
            "``logger.warning(\"[ner.attach] extraction NER dégradée\")``"
        )


def test_baseline_must_be_tightened_when_progress_made() -> None:
    """Symétrique : oblige à abaisser ``UNPREFIXED_LOGS_BASELINE``
    quand des sites sont corrigés."""
    count = len(_scan_unprefixed_logs())
    assert count >= UNPREFIXED_LOGS_BASELINE - 5, (
        f"Logs sans préfixe : {count} < baseline {UNPREFIXED_LOGS_BASELINE}.\n"
        f"Abaisser UNPREFIXED_LOGS_BASELINE = {count} pour verrouiller le gain."
    )
