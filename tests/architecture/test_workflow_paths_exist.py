"""Garde-fou : tout path Python cité dans un workflow GitHub Actions
doit pointer vers un fichier ou glob réel du repo.

Pourquoi ce test existe
-----------------------

Le workflow ``perf_regression.yml`` a surveillé pendant ~6 mois un
fichier ``picarones/app/services/benchmark_runner.py`` supprimé lors
du rewrite.  Conséquence : aucune PR touchant l'orchestrateur ne
déclenchait le check de régression CER — faux positif silencieux.

Ce test prévient structurellement la classe d'erreur : si un
workflow référence un path inexistant (typo, fichier renommé,
module supprimé), la CI échoue.

Périmètre
---------

On scanne les clés ``paths`` et ``paths-ignore`` sous chaque trigger
``on.<event>`` de chaque workflow.  Les paths peuvent contenir des
globs (``**``, ``*``) ; on les résout via ``Path.glob``.

Sont tolérées (allowlist) :

- Les paths qui matchent zéro fichier MAIS dont le pattern reste
  syntaxiquement valide pour de futures additions (ex.
  ``examples/**/*.py`` si ``examples/`` est vide mais réservé).
  Liste explicite — toute addition doit être justifiée en revue.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

#: Paths tolérés bien que ne matchant aucun fichier (pattern réservé
#: pour additions futures).  Ajouter ici demande une justification.
ALLOWED_EMPTY_PATTERNS: frozenset[str] = frozenset()


def _iter_workflow_paths(doc: dict) -> list[tuple[str, str]]:
    """Retourne la liste ``[(event, path_pattern), ...]`` pour un
    workflow YAML.

    PyYAML parse la clé ``on:`` en tant que ``True`` (le booléen YAML
    1.1) — on cherche les deux variantes.
    """
    triggers = doc.get("on") or doc.get(True) or {}
    if not isinstance(triggers, dict):
        return []

    pairs: list[tuple[str, str]] = []
    for event, cfg in triggers.items():
        if not isinstance(cfg, dict):
            continue
        for key in ("paths", "paths-ignore"):
            for pattern in cfg.get(key) or []:
                pairs.append((str(event), str(pattern)))
    return pairs


def _pattern_matches_any_file(pattern: str) -> bool:
    """Vrai si ``pattern`` (glob style GitHub Actions) matche au moins
    un fichier sous ``REPO_ROOT``.

    GitHub Actions accepte les globs POSIX étendus : ``**`` = n'importe
    quel nombre de répertoires, ``*`` = un segment.  ``Path.glob``
    couvre exactement cette syntaxe.
    """
    # Path.glob ne supporte pas les patterns commençant par '/'.
    rel = pattern.lstrip("/")
    try:
        return any(True for _ in REPO_ROOT.glob(rel))
    except (ValueError, OSError):
        return False


def _list_workflow_files() -> list[Path]:
    if not WORKFLOWS_DIR.is_dir():
        return []
    return sorted(WORKFLOWS_DIR.glob("*.yml")) + sorted(
        WORKFLOWS_DIR.glob("*.yaml")
    )


def test_workflows_directory_exists() -> None:
    """Sanity check : ``.github/workflows/`` existe et contient au
    moins un fichier."""
    files = _list_workflow_files()
    assert files, (
        f"Aucun workflow trouvé sous {WORKFLOWS_DIR}.  Si le repo n'a "
        "pas de CI, supprimer ce test."
    )


def test_all_workflow_paths_resolve() -> None:
    """Tout pattern listé dans ``on.<event>.paths`` / ``paths-ignore``
    doit matcher au moins un fichier réel (ou être dans
    ``ALLOWED_EMPTY_PATTERNS``)."""
    offenders: list[str] = []
    for wf in _list_workflow_files():
        try:
            doc = yaml.safe_load(wf.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            pytest.fail(f"{wf.name} : YAML invalide ({e})")
        if not isinstance(doc, dict):
            continue
        for event, pattern in _iter_workflow_paths(doc):
            if pattern in ALLOWED_EMPTY_PATTERNS:
                continue
            if not _pattern_matches_any_file(pattern):
                offenders.append(
                    f"{wf.name} ({event}) : {pattern!r} ne matche aucun fichier"
                )

    assert not offenders, (
        "Workflows référencent des paths inexistants — la CI surveille "
        "du code supprimé ou renommé :\n  "
        + "\n  ".join(offenders)
        + "\n→ corriger les paths ou ajouter le pattern à "
        "ALLOWED_EMPTY_PATTERNS avec justification."
    )


def test_workflow_yaml_has_no_unparsable_python_paths() -> None:
    """Anti-régression secondaire : un path qui ressemble à un module
    Python (``picarones/<...>``) doit pouvoir exister dans
    l'arborescence.  Détecte les fautes de frappe avant qu'elles ne
    propagent en CI silencieuse."""
    offenders: list[str] = []
    py_path_re = re.compile(r"^picarones/[\w/_*.-]+$")
    for wf in _list_workflow_files():
        try:
            doc = yaml.safe_load(wf.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            continue
        if not isinstance(doc, dict):
            continue
        for event, pattern in _iter_workflow_paths(doc):
            if not py_path_re.match(pattern):
                continue
            # Le pattern ressemble à un chemin Python — il doit matcher.
            if not _pattern_matches_any_file(pattern):
                offenders.append(f"{wf.name} ({event}) : {pattern}")

    assert not offenders, (
        "Patterns ``picarones/...`` qui ne matchent rien (probable "
        "renommage non répercuté en CI) :\n  " + "\n  ".join(offenders)
    )
