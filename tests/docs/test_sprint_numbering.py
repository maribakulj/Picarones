"""Audit de la numérotation des fichiers de tests sprint-numérotés.

Sprint A2 (item m-12 de l'audit institutional-readiness-2026-05).

Picarones a évolué sur ~97 sprints. Le pattern historique est de créer
un fichier ``test_sprintNN_<topic>.py`` à chaque sprint. Avec le temps,
des trous de numérotation peuvent apparaître :
- soit parce que le sprint n'a pas eu de test dédié
  (cas légitime — son contenu est testé via d'autres fichiers) ;
- soit parce qu'un fichier a été supprimé sans cleanup
  (cas problématique).

Ce test produit un **rapport informatif** : il liste les trous mais
n'échoue jamais. Il sert d'audit lisible en CI pour repérer une
suppression accidentelle (signal humain, pas blocage automatique).

Pour fail-on-warning : ``pytest -W error::UserWarning tests/docs/test_sprint_numbering.py``.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = REPO_ROOT / "tests"


def _collect_sprint_numbered_tests() -> dict[int, list[Path]]:
    """Retourne ``{numero: [chemins]}`` pour tous les fichiers
    ``test_sprintNN_*.py`` du repo."""
    out: dict[int, list[Path]] = {}
    pattern = re.compile(r"test_sprint(\d{1,3})_")
    for path in TESTS_DIR.rglob("test_sprint*.py"):
        m = pattern.match(path.name)
        if not m:
            continue
        num = int(m.group(1))
        out.setdefault(num, []).append(path)
    return out


def test_sprint_test_numbering_audit() -> None:
    """Audit informatif de la numérotation sprint des fichiers de tests.

    Émet un ``UserWarning`` (ne fail pas) si des trous suspects sont
    détectés. Pour rendre bloquant : utiliser
    ``pytest -W error::UserWarning ...``.
    """
    by_num = _collect_sprint_numbered_tests()

    if not by_num:
        pytest.skip("Aucun fichier test_sprintNN_*.py — convention non utilisée")

    nums = sorted(by_num.keys())
    smallest = nums[0]
    largest = nums[-1]
    holes = [n for n in range(smallest, largest + 1) if n not in by_num]

    # Doublons : plusieurs fichiers pour un même numéro
    duplicates = {n: paths for n, paths in by_num.items() if len(paths) > 1}

    if holes:
        warnings.warn(
            f"Numérotation sprint avec trous : {len(holes)} numéros manquants "
            f"entre {smallest} et {largest}. "
            f"Premiers : {holes[:10]}{'…' if len(holes) > 10 else ''}. "
            f"Vérifier qu'il s'agit bien de sprints sans test dédié et "
            f"non de fichiers supprimés accidentellement.",
            UserWarning,
            stacklevel=2,
        )

    if duplicates:
        warnings.warn(
            f"Plusieurs fichiers de tests pour un même numéro de sprint : "
            f"{ {k: [p.name for p in v] for k, v in duplicates.items()} }. "
            f"Renommer pour préserver l'unicité.",
            UserWarning,
            stacklevel=2,
        )


def test_sprint_test_files_have_docstring() -> None:
    """Audit informatif : tout fichier ``test_sprintNN_*.py`` devrait
    commencer par un docstring mentionnant le sprint correspondant.

    Émet un ``UserWarning`` (ne fail pas) pour signaler les fichiers
    sans contexte. Pour rendre bloquant : ``pytest -W error::UserWarning``.
    Le mass-update de l'existant est planifié dans un sprint dédié de
    polissage post-A14."""
    by_num = _collect_sprint_numbered_tests()
    if not by_num:
        pytest.skip("Aucun fichier test_sprintNN_*.py")

    missing_docstring: list[str] = []
    for num, paths in by_num.items():
        for path in paths:
            head = path.read_text(encoding="utf-8", errors="ignore")[:500]
            # Premier docstring du fichier
            ds_match = re.search(r'^\s*"""([^"]+)"""', head, re.MULTILINE | re.DOTALL)
            if not ds_match:
                missing_docstring.append(str(path.relative_to(REPO_ROOT)))
                continue
            # Le docstring doit mentionner le sprint
            if not re.search(rf"\b[Ss]print\s+{num}\b", ds_match.group(1)):
                missing_docstring.append(
                    f"{path.relative_to(REPO_ROOT)} (docstring sans 'Sprint {num}')"
                )

    if missing_docstring:
        warnings.warn(
            f"{len(missing_docstring)} fichier(s) test_sprintNN_*.py sans "
            f"docstring mentionnant leur sprint. "
            f"Premiers : {missing_docstring[:5]}{'…' if len(missing_docstring) > 5 else ''}. "
            f"Ajouter un docstring de tête au format "
            f'\'\'\'Tests Sprint NN — <description courte>.\'\'\' '
            f"lors d'un sprint de polissage dédié.",
            UserWarning,
            stacklevel=2,
        )
