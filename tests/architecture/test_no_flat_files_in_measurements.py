"""Sprint A14-S3 — geler la fragmentation à plat de ``measurements/``.

Constat de l'audit (cf. ``BACKLOG_POST_LIVRAISON.md`` §2.4) : le
package ``picarones.measurements`` contient ~60 fichiers ``.py`` à
plat, accumulés au fil des Sprints 5-97.  Cette fragmentation rend
le code illisible (60 modules sans hiérarchie) et complique la
migration vers la nouvelle structure ``evaluation/metrics/``.

Cette règle **fige** la liste actuelle (snapshot au Sprint S3) et
**interdit** tout nouveau fichier ``.py`` à plat dans
``measurements/``.  Toute nouvelle métrique / hook / agrégateur
doit aller dans ``picarones/evaluation/metrics/`` (ou un sous-package
approprié).

Comportement attendu en pratique :

- **Nouveau fichier dans evaluation/metrics/** : OK.
- **Nouveau fichier dans measurements/<sous-package>/** (sous-dossier
  comme ``narrative/`` ou ``statistics/`` ou ``runner/``) : OK, le
  test ne regarde que le top-level.
- **Nouveau fichier à plat measurements/<nom>.py** : ÉCHEC.  Soit
  le mettre dans evaluation/metrics/ (préférence forte), soit
  dans un sous-package thématique de measurements/.

La whitelist est intentionnellement gelée à la date du Sprint S3.
Si un fichier de la whitelist est supprimé pendant le rewrite (par
exemple migré vers evaluation/metrics/ au Sprint S10), un autre
test (``test_no_orphaned_whitelist_entries``) le détecte.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MEASUREMENTS_DIR = REPO_ROOT / "picarones" / "measurements"


#: Snapshot de l'état au Sprint A14-S3 (mai 2026).  59 fichiers
#: ``.py`` à plat.  **Ne pas ajouter d'entrée** sans avoir d'abord
#: tenté de placer le fichier dans evaluation/metrics/ ou dans un
#: sous-package thématique.
WHITELIST_FLAT_FILES_S3: frozenset[str] = frozenset({
    "__init__.py",
    "abbreviations.py",
    "alto_metrics.py",
    "builtin_hooks.py",
    "builtin_metrics.py",
    "early_modern_typography.py",
    "equivalence_profile.py",
    "history.py",
    "metrics.py",
    "modern_archives.py",
    "mufi.py",
    "ner.py",
    "numerical_sequences_hooks.py",
    "philological_hooks.py",
    "readability.py",
    "readability_hooks.py",
    "reading_order.py",
    "reliability.py",
    "robustness.py",
    "searchability.py",
    "searchability_hooks.py",
    "unicode_blocks.py",
})


def _flat_python_files() -> set[str]:
    """Liste des fichiers ``.py`` directement dans ``measurements/``.

    Exclut les sous-packages (``narrative/``, ``statistics/``,
    ``runner/``) et les fichiers ``__pycache__``.
    """
    return {
        p.name for p in MEASUREMENTS_DIR.glob("*.py")
        if "__pycache__" not in p.parts
    }


def test_no_new_flat_file_in_measurements() -> None:
    """Toute addition à plat dans ``measurements/`` est interdite.

    Si ce test échoue après l'ajout d'un fichier, deux options :

    1. **Préférée** : déplacer le fichier dans
       ``picarones/evaluation/metrics/`` (ou un sous-package
       approprié).
    2. **Acceptable seulement avec justification** : si le fichier
       *doit* vivre dans ``measurements/`` pendant la transition
       (ex : refactor d'un fichier de la whitelist qui se scinde),
       l'ajouter à WHITELIST_FLAT_FILES_S3 dans ce fichier en
       expliquant pourquoi dans le message de commit.
    """
    actual = _flat_python_files()
    new_files = actual - WHITELIST_FLAT_FILES_S3
    assert not new_files, (
        "\nNouveaux fichiers ``.py`` à plat dans ``picarones/measurements/`` "
        "(plan rewrite-2026 §S3 — fragmentation gelée) :\n"
        + "\n".join(f"  - {f}" for f in sorted(new_files))
        + "\n\nDéplacer ces fichiers vers ``picarones/evaluation/metrics/`` "
        "ou un sous-package approprié.  Voir docs/roadmap/rewrite-2026.md."
    )


def test_no_orphaned_whitelist_entries() -> None:
    """La whitelist ne doit pas contenir d'entrée pointant vers un
    fichier qui n'existe plus.

    Garantit que la migration des fichiers vers ``evaluation/metrics/``
    (Sprint S10) entraîne automatiquement la mise à jour de cette
    whitelist — pas de dette qui s'accumule.
    """
    actual = _flat_python_files()
    orphans = WHITELIST_FLAT_FILES_S3 - actual
    assert not orphans, (
        "\nWhitelist contient des fichiers qui n'existent plus dans "
        "``picarones/measurements/`` :\n"
        + "\n".join(f"  - {f}" for f in sorted(orphans))
        + "\n\nLe fichier a été déplacé/supprimé — retirer l'entrée "
        "de WHITELIST_FLAT_FILES_S3 dans ce fichier."
    )


def test_subpackages_not_affected() -> None:
    """Méta-test : les sous-packages existants de ``measurements/``
    (narrative, statistics, runner) restent intouchés par ce test."""
    expected_subpackages = {"narrative", "runner"}
    actual = {
        p.name for p in MEASUREMENTS_DIR.iterdir()
        if p.is_dir() and not p.name.startswith("_") and "__pycache__" not in p.name
    }
    missing = expected_subpackages - actual
    assert not missing, (
        f"Sous-packages attendus dans measurements/ absents : {missing}. "
        "Si l'un d'eux a été migré vers la nouvelle architecture (S10+), "
        "retirer son nom de ``expected_subpackages`` ici."
    )
