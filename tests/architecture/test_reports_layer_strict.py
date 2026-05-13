"""Phase 5.1 audit code-quality — ``picarones.reports/`` (couche 7)
n'importe que depuis les couches **strictement intérieures** : domain,
formats, evaluation, pipeline.

Avant la Phase 5.1, 4 imports illégaux ``reports/ → app/`` (couche
7 → 6) violaient l'orientation des couches :

- ``reports/csv/render.py:53`` → ``from picarones.app.results import RunResult``
- ``reports/json/render.py:51`` → idem
- ``reports/html/render.py:54`` → ``from picarones.app.results import RunDocumentResult, RunResult``

La résolution a déplacé ``RunResult`` / ``RunDocumentResult`` /
``ReportRenderer`` de ``picarones.app.results`` vers
``picarones.pipeline.run_result`` (couche 4) — accessible depuis
``reports/`` selon l'ordre canonique du manifeste CLAUDE.md.

Ce test verrouille la règle pour empêcher la régression : tout PR
qui réintroduit un ``from picarones.app...`` dans ``reports/``
échoue ici.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "picarones" / "reports"

#: Couches strictement intérieures à ``reports/`` (couche 7).  Les
#: imports cross-package du module ``reports`` ne doivent pointer
#: que vers ces noms.
ALLOWED_INNER_LAYERS: frozenset[str] = frozenset({
    "domain",      # 1
    "formats",     # 2
    "evaluation",  # 3
    "pipeline",    # 4
})

#: Couches **extérieures** à ``reports/`` — interdites en import.
FORBIDDEN_OUTER_LAYERS: frozenset[str] = frozenset({
    "adapters",    # 5
    "app",         # 6
    # 7 = reports lui-même (auto-import OK)
    "interfaces",  # 8
})


def _imported_top_packages(path: Path) -> set[str]:
    """Retourne l'ensemble des sous-paquets de ``picarones`` importés
    par un fichier ``reports/...``.  Exclut les sous-modules de
    ``reports`` lui-même."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            parts = mod.split(".")
            if len(parts) >= 2 and parts[0] == "picarones":
                out.add(parts[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if len(parts) >= 2 and parts[0] == "picarones":
                    out.add(parts[1])
    return out


def test_reports_does_not_import_from_outer_layers() -> None:
    """``picarones/reports/**.py`` ne contient aucun import depuis
    une couche plus externe (``adapters``, ``app``, ``interfaces``).
    """
    offenders: list[tuple[str, set[str]]] = []
    for path in sorted(REPORTS_DIR.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        imported = _imported_top_packages(path)
        bad = imported & FORBIDDEN_OUTER_LAYERS
        if bad:
            offenders.append((str(path.relative_to(REPO_ROOT)), bad))

    if offenders:
        lines = "\n".join(
            f"  {p} importe depuis : {sorted(b)}"
            for p, b in offenders
        )
        raise AssertionError(
            "Imports illégaux ``reports/ → couches externes`` :\n"
            + lines
            + "\n\nLa règle CLAUDE.md interdit à ``reports/`` (couche 7) "
            "d'importer depuis ``adapters/`` (5), ``app/`` (6) ou "
            "``interfaces/`` (8).  Si le type recherché vit dans ces "
            "couches, soit le déplacer vers ``pipeline/`` (couche 4) "
            "ou plus interne, soit créer un protocole côté ``domain/``."
        )


def test_run_result_moved_to_pipeline_layer() -> None:
    """``RunResult`` / ``RunDocumentResult`` / ``ReportRenderer``
    doivent être importables depuis ``picarones.pipeline.run_result``.

    Régression : si un PR repasse les définitions canoniques dans
    ``app/results.py``, le shim de re-export plante.
    """
    from picarones.pipeline.run_result import (
        ReportRenderer,
        RunDocumentResult,
        RunResult,
    )

    assert RunResult is not None
    assert RunDocumentResult is not None
    assert ReportRenderer is not None


def test_app_results_is_compat_shim() -> None:
    """``picarones.app.results`` reste accessible (compat interne)
    et expose les mêmes noms que ``picarones.pipeline.run_result``."""
    from picarones.app import results as app_results
    from picarones.pipeline import run_result as canonical

    assert app_results.RunResult is canonical.RunResult
    assert app_results.RunDocumentResult is canonical.RunDocumentResult
    assert app_results.ReportRenderer is canonical.ReportRenderer
