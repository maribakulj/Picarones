"""Garde-fou contre la dérive doc-vs-code.

Scanne ``CLAUDE.md``, ``README.md``, ``docs/**/*.md`` à la recherche de
chemins de la forme ``picarones/.../X.py`` et vérifie qu'ils existent
dans le repo.

Snapshot v1.0.0 (2026-05-02) : **119 chemins cassés**, presque tous
dans ``CLAUDE.md`` et ``CHANGELOG.md`` qui décrivent systématiquement
des modules sous ``picarones/core/...`` alors qu'ils vivent dans
``picarones/measurements/...``. C'est une dette documentaire connue
qu'il faut résorber par paliers.

Test ratchet : le nombre de chemins cassés ne peut que diminuer. Pour
le faire baisser :

1. Soit corriger le chemin dans la doc.
2. Soit déplacer le module au chemin documenté (rare — la doc se
   trompe presque toujours).
3. Soit retirer la référence devenue obsolète.

Puis abaisser :data:`BROKEN_PATHS_BASELINE` du même montant.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Snapshot. Doit baisser, jamais monter.
#:
#: Historique :
#: - 119 (initial v1.0.0, dette pré-existante CLAUDE.md/CHANGELOG.md
#:   qui décrivent des modules sous ``picarones/core/...`` alors qu'ils
#:   vivent dans ``picarones/measurements/...``).
#: - 122 (sprint « découpage de statistics.py », 2026-05-02) : 3 audits
#:   historiques référencent ``picarones/measurements/statistics.py``
#:   qui est maintenant un sous-package. Baseline relevée.
#: - 72 (sprint « zéro dette actionnable », 2026-05-02) : 50 chemins
#:   massivement corrigés — 44 dans CLAUDE.md + 6 dans docs vivants.
#: - 73 (sprint « découpage de runner.py », 2026-05-03) :
#:   ``picarones/measurements/runner.py`` est désormais un sous-package
#:   ``runner/``. ``docs/user/writing-a-pipeline-module.md`` a été
#:   corrigé en place ; un audit historique
#:   (``docs/audits/institutional-readiness-2026-05.md``) référence
#:   l'ancien chemin et reste intouché par convention.
#:
#: Les 73 restants sont **TOUS** dans :
#: - ``CHANGELOG.md`` (67) : journal historique versionné, intouchable.
#: - ``docs/audits/*.md`` (6) : audits historiques, intouchables.
BROKEN_PATHS_BASELINE = 73

#: Patrons de fichiers de documentation à scanner.
DOC_GLOBS: tuple[str, ...] = (
    "CLAUDE.md",
    "README.md",
    "CHANGELOG.md",
    "SPECS.md",
    "docs/**/*.md",
)

#: Pattern minimal d'un chemin Python dans le repo.
PATH_PATTERN: re.Pattern[str] = re.compile(
    r"picarones/[a-z_][a-z_0-9]*(?:/[a-z_][a-z_0-9]*)*\.py"
)


def _doc_files() -> list[Path]:
    files: list[Path] = []
    for glob in DOC_GLOBS:
        files.extend(REPO_ROOT.glob(glob))
    return sorted({f for f in files if f.is_file()})


def _broken_paths() -> list[tuple[str, str]]:
    """Liste des (doc_relatif, chemin_cassé), dédoublonnée et triée."""
    broken: set[tuple[str, str]] = set()
    for doc in _doc_files():
        try:
            text = doc.read_text(encoding="utf-8")
        except OSError:
            continue
        rel_doc = doc.relative_to(REPO_ROOT).as_posix()
        for match in PATH_PATTERN.findall(text):
            if not (REPO_ROOT / match).exists():
                broken.add((rel_doc, match))
    return sorted(broken)


def test_broken_doc_paths_below_baseline() -> None:
    """Le nombre de chemins cassés ne peut que diminuer."""
    broken = _broken_paths()
    if len(broken) > BROKEN_PATHS_BASELINE:
        sample = "\n".join(f"  {doc} → {path}" for doc, path in broken[:30])
        more = f"\n  ... ({len(broken) - 30} de plus)" if len(broken) > 30 else ""
        raise AssertionError(
            f"\n{len(broken)} chemins de doc cassés (baseline "
            f"{BROKEN_PATHS_BASELINE}).\n"
            f"Régression : la doc référence un fichier qui n'existe pas.\n\n"
            f"Échantillon :\n{sample}{more}\n\n"
            "Soit corrige le chemin, soit le code, soit retire la référence."
        )


def test_baseline_must_be_tightened_when_progress_made() -> None:
    """Si on est sous le baseline, mettre à jour :data:`BROKEN_PATHS_BASELINE`.

    Verrouille chaque correction de doc pour empêcher une régression
    future de glisser sous le seuil obsolète.
    """
    broken = _broken_paths()
    assert len(broken) >= BROKEN_PATHS_BASELINE, (
        f"\nExcellent : {len(broken)} chemins cassés vs baseline "
        f"{BROKEN_PATHS_BASELINE}.\n\n"
        f"Mets à jour BROKEN_PATHS_BASELINE = {len(broken)} dans "
        "tests/architecture/test_doc_paths.py pour verrouiller le gain."
    )
