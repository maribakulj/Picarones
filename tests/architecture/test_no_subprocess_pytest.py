"""Garde-fou : aucun test ne doit lancer pytest ou mypy via subprocess.

Pourquoi ce test existe
-----------------------

Lancer ``subprocess.run([sys.executable, "-m", "pytest", ...])``
depuis un test pytest cause un deadlock potentiel sur le lock du
fichier ``.coverage`` quand le test parent tourne lui-même sous
``pytest --cov`` (cas standard de la CI).

L'historique du repo contient des commentaires comme « ``-p
no:cacheprovider`` + ``--no-cov`` évitent les deadlocks de récursion »
— c'est précisément ce que ce test prévient en bloquant la cause
plutôt qu'en mitigeant les symptômes.

Les outils en ligne de commande (``mypy``, ``pytest``, ``ruff``,
``bandit``) exposent tous une API programmatique :

- ``from mypy import api ; api.run([...])``
- ``import pytest ; pytest.main([...])`` (rare, généralement
  remplaçable par une assertion directe sur ``collect_only``)
- ``import ruff`` non exposé, mais le besoin est rare en test

Périmètre
---------

On scanne tous les fichiers ``tests/**/*.py`` à la recherche de
patterns qui correspondent à un appel subprocess vers ces outils.
On accepte :

- les ``subprocess`` qui lancent des binaires système (``tesseract``,
  ``docker``, etc.) ;
- les ``subprocess`` qui lancent un script Python du repo
  (``scripts/...``) tant que ce script ne ré-invoque pas pytest.

On refuse :

- ``subprocess.run([..., "pytest", ...])``
- ``subprocess.run([sys.executable, "-m", "pytest", ...])``
- ``pytest.main(...)`` (récursion potentielle)
- ``subprocess.run([..., "mypy", ...])`` (utiliser ``mypy.api.run``)

Exceptions
----------

Aucune n'est tolérée.  Si un cas vraiment indispensable apparaît,
l'ajouter ici **avec justification** plutôt que de le laisser
fragiliser une partie du repo.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = REPO_ROOT / "tests"

#: Fichiers tolérés explicitement.  Le scanner lui-même contient les
#: patterns qu'il interdit (sinon il ne pourrait pas les chercher) ;
#: il s'auto-exclut.  Toute autre addition demande une justification
#: en revue.
ALLOWLIST: frozenset[str] = frozenset({
    "tests/architecture/test_no_subprocess_pytest.py",
})

#: Patterns refusés.  L'ordre importe : on retient le premier match
#: pour un message d'erreur clair.
_FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "subprocess.run([..., 'pytest', ...])",
        re.compile(
            r'subprocess\.(?:run|Popen|check_call|check_output|call)'
            r'\s*\([^)]*["\']pytest["\']',
            re.DOTALL,
        ),
    ),
    (
        "subprocess.run([sys.executable, '-m', 'pytest', ...])",
        re.compile(
            r'subprocess\.(?:run|Popen|check_call|check_output|call)'
            r'\s*\(\s*\[[^\]]*sys\.executable[^\]]*["\']pytest["\']',
            re.DOTALL,
        ),
    ),
    (
        "subprocess.run([..., 'mypy', ...])",
        re.compile(
            r'subprocess\.(?:run|Popen|check_call|check_output|call)'
            r'\s*\([^)]*["\']mypy["\']',
            re.DOTALL,
        ),
    ),
    (
        "pytest.main(...)",
        re.compile(r'\bpytest\.main\s*\('),
    ),
)


def _strip_comments_and_docstrings(text: str) -> str:
    """Retire les commentaires Python et les docstrings triple-quoted
    pour éviter les faux positifs sur les fichiers qui *décrivent* le
    motif interdit en prose (cas typique d'un commentaire ``# Historique :
    ce test lançait subprocess.run(..., 'pytest', ...) ...``).

    L'heuristique est volontairement simple — pas de parser Python
    complet — parce qu'on ne veut pas matcher un motif qui apparaît
    uniquement dans du texte non exécutable."""
    # Triple-quoted strings (docstrings et chaînes multi-lignes)
    text = re.sub(r'"""[\s\S]*?"""', "", text)
    text = re.sub(r"'''[\s\S]*?'''", "", text)
    # Commentaires single-line : tout ce qui suit un ``#`` sur la ligne.
    # On ignore le cas pathologique d'un ``#`` dans une chaîne car le
    # fichier scanné est du code de test (pas de littérature
    # défensive nécessaire à ce stade).
    text = re.sub(r"#[^\n]*", "", text)
    return text


def _scan_file(path: Path) -> list[str]:
    """Retourne la liste des patterns interdits trouvés dans ``path``."""
    text = _strip_comments_and_docstrings(
        path.read_text(encoding="utf-8")
    )
    return [
        label
        for label, pattern in _FORBIDDEN_PATTERNS
        if pattern.search(text)
    ]


def test_no_test_invokes_pytest_or_mypy_via_subprocess() -> None:
    offenders: list[str] = []
    for path in sorted(TESTS_DIR.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in ALLOWLIST:
            continue
        found = _scan_file(path)
        if found:
            offenders.append(f"{rel} : {', '.join(found)}")

    assert not offenders, (
        "Tests qui invoquent pytest/mypy en subprocess (risque de "
        "deadlock pytest-dans-pytest et / ou de skip silencieux) :\n  "
        + "\n  ".join(offenders)
        + "\n\n→ Utiliser l'API programmatique :\n"
        "    from mypy import api ; stdout, stderr, rc = api.run([...])\n"
        "    # ou supprimer le test s'il duplique un check existant"
    )
