"""Garde-fou : aucun module du rewrite n'importe depuis le legacy
**et** aucun paquet legacy ne ressuscite sous son ancien nom.

L'arborescence canonique 8 couches (``domain → formats → evaluation
→ pipeline → adapters → app → reports → interfaces``) est autonome
depuis la v2.0 (mai 2026).  Tous les paquets legacy historiquement
listés ont été supprimés au cours des sprints A-H.

Phase 4.5 de l'audit code-quality (2026-05) : avant cette refonte,
``LEGACY_PACKAGES = ()`` rendait ``test_rewrite_modules_dont_import_from_legacy``
**vacuement vrai** (boucle sur un itérable vide → toujours OK).  Le
test passait mais ne vérifiait rien.

Refonte en deux invariants actifs :

1. ``test_no_resurrected_legacy_package_directory`` : aucun dossier
   au nom d'un paquet legacy historique ne réapparaît dans
   ``picarones/``.  Si quelqu'un recrée ``picarones/core/`` ou
   ``picarones/web/``, le test plante.

2. ``test_no_imports_of_resurrected_legacy_module`` : aucun fichier
   du rewrite n'importe depuis ces noms supprimés (même par
   ``picarones.web.app:app`` dans une string, scope du sister test
   ``test_no_legacy_picarones_web_references`` côté entry points).

Le garde-fou structurel (``test_layer_dependencies.py``) couvre
les imports inter-couches sains ; ce fichier couvre spécifiquement
la **non-régression du retrait du legacy**.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Paquets de l'arborescence canonique v2.0.
REWRITE_PACKAGES: tuple[str, ...] = (
    "domain",
    "formats",
    "evaluation",
    "pipeline",
    "adapters",
    "app",
    "reports",
    "interfaces",
)

#: Noms de paquets historiquement legacy, supprimés au cours des
#: sprints A-H (mai 2026).  Si l'un d'eux réapparaît :
#:
#: - Soit le retrait a été partiellement annulé (régression).
#: - Soit un nouveau paquet a réutilisé un nom homonyme.  Dans ce
#:   cas, le retirer de cette liste avec un commentaire expliquant
#:   pourquoi le sens a changé.
#:
#: Source : CHANGELOG v2.0 et ``docs/archives/migration/``.
RESURRECTED_LEGACY_NAMES: tuple[str, ...] = (
    "core",
    "measurements",
    "engines",
    "modules",
    "report",
    "llm",
    "pipelines",
    "cli",
    "web",
    "extras",
)

#: Sous-paquets transitoires retirés en parallèle.  Format
#: ``parent/sub`` pour matcher le chemin filesystem.
RESURRECTED_LEGACY_SUBPACKAGES: tuple[tuple[str, str], ...] = (
    ("adapters", "legacy_engines"),
    ("adapters", "legacy_pipelines"),
    ("interfaces/cli", "_legacy"),
    ("interfaces/web", "_legacy"),
)

#: Pattern qui matche un import déclaré dans le code source.
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


def _scan_legacy_imports(
    path: Path,
    forbidden_top_levels: set[str],
) -> list[tuple[int, str]]:
    """``(lineno, import_text)`` pour chaque import qui pointe vers
    un nom legacy ressuscité."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    offenders: list[tuple[int, str]] = []
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            parts = mod.split(".")
            if (
                len(parts) >= 2
                and parts[0] == "picarones"
                and parts[1] in forbidden_top_levels
            ):
                offenders.append((node.lineno, f"from {mod} import ..."))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if (
                    len(parts) >= 2
                    and parts[0] == "picarones"
                    and parts[1] in forbidden_top_levels
                ):
                    offenders.append((node.lineno, f"import {alias.name}"))
    return offenders


# --------------------------------------------------------------------------
# Invariant 1 — aucun dossier legacy ne réapparaît dans picarones/
# --------------------------------------------------------------------------


def test_no_resurrected_legacy_package_directory() -> None:
    """Aucun dossier au nom d'un paquet legacy historique n'existe
    sous ``picarones/``.

    Si ce test échoue, soit un retrait a été annulé (régression),
    soit un nouveau paquet a homonymie accidentelle — dans les deux
    cas, l'attention du reviewer est requise.
    """
    resurrected: list[str] = []
    for name in RESURRECTED_LEGACY_NAMES:
        path = REPO_ROOT / "picarones" / name
        if path.is_dir():
            resurrected.append(f"picarones/{name}/")

    for parent, sub in RESURRECTED_LEGACY_SUBPACKAGES:
        path = REPO_ROOT / "picarones" / parent / sub
        if path.is_dir():
            resurrected.append(f"picarones/{parent}/{sub}/")

    assert not resurrected, (
        "Paquet(s) legacy ressuscité(s) :\n"
        + "\n".join(f"  - {p}" for p in resurrected)
        + "\n\nLe retrait v2.0 (sprints A-H) avait acté la suppression "
        "définitive.  Si la réintroduction est intentionnelle, retirer "
        "le nom de ``RESURRECTED_LEGACY_NAMES`` / "
        "``RESURRECTED_LEGACY_SUBPACKAGES`` avec un commentaire dans "
        "ce fichier (pourquoi le sens a changé)."
    )


# --------------------------------------------------------------------------
# Invariant 2 — aucun import du rewrite ne cible un nom legacy
# --------------------------------------------------------------------------


def test_no_imports_of_resurrected_legacy_module() -> None:
    """Aucun fichier du rewrite n'importe ``picarones.<legacy_name>...``.

    Même si l'invariant 1 garantit l'absence du dossier, un import
    statique pourrait subsister dans le code et planter à l'exécution.
    Ce test attrape ce drift plus tôt.
    """
    forbidden = set(RESURRECTED_LEGACY_NAMES)
    offenders: list[tuple[str, int, str]] = []
    for path in _rewrite_modules():
        rel = path.relative_to(REPO_ROOT).as_posix()
        for lineno, import_text in _scan_legacy_imports(path, forbidden):
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
            f"\n{len(offenders)} import(s) ciblant un nom legacy "
            f"ressuscité :\n\n{sample}{more}\n\n"
            "Le code source rewrite ne doit pas importer depuis les "
            "paquets supprimés en v2.0.  Migrer l'import vers la "
            "couche canonique correspondante."
        )


# --------------------------------------------------------------------------
# Cohérence : les paquets rewrite existent bien
# --------------------------------------------------------------------------


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
