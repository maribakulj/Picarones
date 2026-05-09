"""Garde-fou : aucun module du rewrite n'importe depuis le legacy.

L'arborescence post-rewrite (``domain → formats → evaluation →
pipeline → adapters → app → reports_v2 → interfaces``) doit être
**autonome**.  Le legacy peut s'appuyer sur le rewrite (re-exports),
mais l'inverse romprait l'invariant — chaque retrait de paquet
legacy au cours des phases 1-11 ferait planter le rewrite.

Ce test scanne tous les fichiers Python des paquets rewrite et
rejette toute déclaration d'import qui pointe vers un paquet
legacy.

Listes de référence
-------------------

Les paquets sont déclarés ici de manière explicite — un nouveau
paquet rewrite ou legacy doit être inscrit consciemment, pas
auto-détecté.  Cela évite qu'une erreur de structure (un paquet
posé au mauvais endroit) ne soit silencieusement classée par
heuristique.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Paquets de l'arborescence rewrite (cible 2.0).  Ne doivent
#: jamais importer depuis :data:`LEGACY_PACKAGES`.
REWRITE_PACKAGES: tuple[str, ...] = (
    "domain",
    "formats",
    "evaluation",
    "pipeline",
    "adapters",
    "app",
    "reports_v2",
    "interfaces",
)

#: Paquets legacy.  Importables uniquement depuis l'intérieur du
#: legacy lui-même (ou depuis les tests, qui valident la migration
#: en cours).
LEGACY_PACKAGES: tuple[str, ...] = ()

#: Pattern qui matche un import déclaré dans le code source.
#:
#: Couvre :
#:
#: - ``from picarones.X import ...``
#: - ``import picarones.X``
#: - ``import picarones.X as Y``
#:
#: Ne couvre PAS les imports différés via ``importlib.import_module``
#: ou ``__import__`` — le test architectural cible la déclaration
#: statique, pas la résolution dynamique.
_IMPORT_RE = re.compile(
    r"^\s*(?:from|import)\s+picarones\.([a-z_][a-z_0-9]*)",
    re.MULTILINE,
)


def _rewrite_modules() -> list[Path]:
    """Liste tous les fichiers ``.py`` des paquets rewrite."""
    out: list[Path] = []
    for pkg in REWRITE_PACKAGES:
        root = REPO_ROOT / "picarones" / pkg
        if not root.exists():
            continue
        out.extend(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(out)


def _scan_legacy_imports(path: Path) -> list[tuple[int, str]]:
    """Retourne la liste des ``(numéro_de_ligne, import_legacy)``
    trouvés dans ``path``.

    Utilise l'AST pour capturer les imports indentés (à l'intérieur
    de fonctions, ``TYPE_CHECKING``, etc.) — un grep simple raterait
    ces cas.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    offenders: list[tuple[int, str]] = []
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        # On laisse les autres tests d'archi attraper les fichiers
        # cassés.
        return []
    legacy_set = set(LEGACY_PACKAGES)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            parts = mod.split(".")
            if len(parts) >= 2 and parts[0] == "picarones" and parts[1] in legacy_set:
                offenders.append((node.lineno, f"from {mod} import ..."))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if (
                    len(parts) >= 2
                    and parts[0] == "picarones"
                    and parts[1] in legacy_set
                ):
                    offenders.append((node.lineno, f"import {alias.name}"))
    return offenders


def test_rewrite_modules_dont_import_from_legacy() -> None:
    """Aucun fichier des paquets rewrite n'a d'import legacy.

    Si ce test échoue, le rewrite a une dépendance qui empêchera
    le retrait du paquet legacy concerné.  Deux fixes possibles :

    1. Le code legacy importé existe en équivalent dans le rewrite
       → migrer l'import.
    2. Il n'existe pas encore → la fonctionnalité doit être inscrite
       au plan ``docs/migration/legacy-retirement-plan.md`` comme
       bloquante avant le retrait du paquet legacy concerné.
    """
    offenders: list[tuple[str, int, str]] = []
    for path in _rewrite_modules():
        rel = path.relative_to(REPO_ROOT).as_posix()
        for lineno, import_text in _scan_legacy_imports(path):
            offenders.append((rel, lineno, import_text))

    if offenders:
        sample = "\n".join(
            f"  {p}:{n} → {s}" for p, n, s in offenders[:30]
        )
        more = (
            f"\n  ... ({len(offenders) - 30} de plus)"
            if len(offenders) > 30
            else ""
        )
        raise AssertionError(
            f"\n{len(offenders)} import(s) legacy détecté(s) dans le "
            "rewrite.  Le retrait du legacy en sera bloqué.\n\n"
            f"{sample}{more}\n\n"
            "Soit migrer l'import vers l'équivalent rewrite, soit "
            "inscrire la fonctionnalité manquante dans "
            "``docs/migration/legacy-retirement-plan.md`` comme "
            "bloquante.",
        )


def test_legacy_packages_match_directory_structure() -> None:
    """Cohérence : les noms déclarés dans :data:`LEGACY_PACKAGES`
    correspondent à des dossiers réels.

    Quand un paquet legacy est supprimé (au fil des phases), il faut
    le retirer aussi de cette liste — sinon le test ci-dessus ne
    refusera plus les imports vers ce paquet désormais inexistant
    (ce serait quand même un import cassé, pris par d'autres tests,
    mais incohérent).
    """
    missing = []
    for pkg in LEGACY_PACKAGES:
        if not (REPO_ROOT / "picarones" / pkg).is_dir():
            missing.append(pkg)
    assert not missing, (
        f"Paquet(s) déclaré(s) dans LEGACY_PACKAGES mais sans "
        f"dossier correspondant : {missing}.  Si ces paquets ont été "
        "retirés au cours d'une phase de migration, mettre à jour "
        "LEGACY_PACKAGES ici."
    )


def test_rewrite_packages_match_directory_structure() -> None:
    """Cohérence : les paquets cibles existent."""
    missing = []
    for pkg in REWRITE_PACKAGES:
        if not (REPO_ROOT / "picarones" / pkg).is_dir():
            missing.append(pkg)
    assert not missing, (
        f"Paquet(s) du rewrite déclaré(s) mais absent(s) du "
        f"filesystem : {missing}."
    )
