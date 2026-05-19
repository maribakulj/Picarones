"""Garde-fou anti-régression — plus aucun test « sprint-nommé ».

Audit prod Chantier 1 : les 183 fichiers ``test_sprintNN_*`` /
``test_s<N>_*`` ont été dé-sprintés (nommés par *comportement*, pas
par épisode de chantier — l'historique vit dans git/CHANGELOG).  Ce
test verrouille la convention : un nouveau fichier sprint-nommé fait
échouer la CI (ratchet, comme ruff/budgets/doc_paths).  Remplace
l'ancien ``tests/docs/test_sprint_numbering.py`` qui *auditait* la
numérotation sprint — devenu sans objet.

Règle : un fichier de test ne doit pas encoder un numéro de sprint
dans son nom.  ``test_sprint*`` interdit ; ``test_s<chiffre>*``
interdit (raccourci sprint historique).  ``test_s<lettre>*`` reste
autorisé (``test_scientific_audit``, ``test_specs_consistency``,
``test_storage_keys`` … : le ``s`` y est une lettre de mot, pas un
marqueur de sprint).
"""

from __future__ import annotations

import re
from pathlib import Path

_TESTS = Path(__file__).resolve().parents[1]
_SPRINT_NAME = re.compile(r"^test_(sprint|s[0-9])")


def test_no_sprint_named_test_files() -> None:
    offenders = sorted(
        p.relative_to(_TESTS).as_posix()
        for p in _TESTS.rglob("test_*.py")
        if _SPRINT_NAME.match(p.name)
    )
    assert not offenders, (
        "Fichiers de test sprint-nommés détectés (Chantier 1 = zéro) "
        "— nommer par COMPORTEMENT, pas par sprint ; l'historique "
        "vit dans git/CHANGELOG :\n  " + "\n  ".join(offenders)
    )


def test_no_residual_sprint_module_imports() -> None:
    """Aucun import inter-tests (bare ou dotted) ne doit encore
    pointer un module ``test_sprint*`` / ``test_s<N>*`` (un renommage
    sans patch de l'importateur casserait la collecte)."""
    pat = re.compile(
        r"\b(?:import|from)\s+"
        r"(?:[\w.]+\.)?(test_(?:sprint[0-9_a-z]+|s[0-9][0-9_a-z]*))\b",
    )
    offenders: list[str] = []
    for p in _TESTS.rglob("*.py"):
        for ln in p.read_text(encoding="utf-8").splitlines():
            s = ln.strip()
            if s.startswith("#"):
                continue
            m = pat.search(s)
            if m:
                offenders.append(
                    f"{p.relative_to(_TESTS).as_posix()}: {s}"
                )
    assert not offenders, (
        "Imports vers un module test sprint-nommé encore présents :\n  "
        + "\n  ".join(offenders)
    )
