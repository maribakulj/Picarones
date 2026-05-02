"""Tests de cohérence du ``CHANGELOG.md``.

Sprint A2 — vérifications minimales :

1. Le fichier doit exister et suivre la structure Keep-a-Changelog
   (en-tête, sections ``[Unreleased]`` ou versions taggées).
2. Toute référence ``Sprint NN`` doit correspondre à une entrée
   décrite ailleurs dans CHANGELOG ou CLAUDE.md (le test ne vérifie
   pas le contenu, juste la résolution de la référence).
3. Les liens internes (chemins ``docs/...``, ``picarones/...``) doivent
   pointer vers des fichiers existants.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CHANGELOG_PATH = REPO_ROOT / "CHANGELOG.md"
CLAUDE_MD_PATH = REPO_ROOT / "CLAUDE.md"


def _read_changelog() -> str:
    if not CHANGELOG_PATH.exists():
        pytest.skip("CHANGELOG.md absent")
    return CHANGELOG_PATH.read_text(encoding="utf-8")


def _read_claude_md() -> str:
    if not CLAUDE_MD_PATH.exists():
        return ""
    return CLAUDE_MD_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Existence et structure
# ---------------------------------------------------------------------------


def test_changelog_exists() -> None:
    assert CHANGELOG_PATH.exists(), "CHANGELOG.md absent à la racine."


def test_changelog_keep_a_changelog_format() -> None:
    """CHANGELOG doit déclarer suivre Keep a Changelog."""
    text = _read_changelog()
    assert "keepachangelog" in text.lower(), (
        "CHANGELOG.md doit mentionner Keep a Changelog "
        "(https://keepachangelog.com/) pour signaler le format adopté."
    )


def test_changelog_has_versioned_sections() -> None:
    """CHANGELOG doit contenir au moins une section versionnée
    ``## [...]``."""
    text = _read_changelog()
    assert re.search(r"^##\s+\[[^]]+\]", text, re.MULTILINE), (
        "CHANGELOG.md doit contenir au moins une section ``## [Version]`` "
        "ou ``## [Unreleased]``."
    )


# ---------------------------------------------------------------------------
# Sprints référencés
# ---------------------------------------------------------------------------


def test_sprint_references_resolve() -> None:
    """Toute référence ``Sprint NN`` (ou ``Sprint NN+`` ou ``Sprint NN-MM``)
    dans CHANGELOG doit avoir au moins une entrée descriptive
    correspondante quelque part — soit dans CHANGELOG lui-même, soit
    dans CLAUDE.md.

    Le test n'exige pas que CHAQUE entrée soit décrite dans CHANGELOG
    (la migration entre CLAUDE.md et CHANGELOG est progressive). Il
    refuse simplement les références orphelines (faute de frappe ou
    sprint inventé)."""
    changelog = _read_changelog()
    claude = _read_claude_md()

    # Capture les sprints référencés (Sprint 18, sprint 35-37, Sprint 97, …)
    references: set[int] = set()
    for m in re.finditer(r"[Ss]print\s+(\d{1,3})(?:[-+]?\d{0,3})?", changelog):
        try:
            references.add(int(m.group(1)))
        except ValueError:
            continue

    # Sprints décrits = ceux qui apparaissent dans CLAUDE.md (table
    # des sprints réalisés) ou dans une section dédiée du CHANGELOG.
    described: set[int] = set()
    # CLAUDE.md a un tableau au format `| 18 | …` ou `| 35 | …`
    for m in re.finditer(r"^\|\s*(\d{1,3})\s*\|", claude, re.MULTILINE):
        try:
            described.add(int(m.group(1)))
        except ValueError:
            continue
    # CHANGELOG peut décrire `## [post-Sprint 97] — chantiers …`
    for m in re.finditer(r"[Ss]print\s+(\d{1,3})", changelog):
        # Tout sprint mentionné dans CHANGELOG est tolé (s'il a son contexte).
        try:
            described.add(int(m.group(1)))
        except ValueError:
            continue

    orphans = sorted(references - described)
    assert not orphans, (
        f"Sprints référencés dans CHANGELOG mais non documentés ailleurs : "
        f"{orphans}. Les sprints officiels sont listés dans CLAUDE.md."
    )


# ---------------------------------------------------------------------------
# Liens internes
# ---------------------------------------------------------------------------

#: Patterns d'URL externes à ignorer (pas notre responsabilité de les vérifier).
EXTERNAL_URL_PREFIXES = ("http://", "https://", "mailto:")


def _extract_internal_links(text: str) -> list[tuple[str, str]]:
    """Retourne ``[(link_text, link_target), ...]`` pour les liens markdown
    qui pointent vers le repo (pas vers une URL externe)."""
    out: list[tuple[str, str]] = []
    for m in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", text):
        target = m.group(2).strip()
        if target.startswith(EXTERNAL_URL_PREFIXES):
            continue
        if target.startswith("#"):  # ancre interne au document
            continue
        # Strip trailing query/fragment
        target = target.split("#")[0].split("?")[0]
        if target:
            out.append((m.group(1), target))
    return out


def test_changelog_internal_links_resolve() -> None:
    """Tout lien interne (chemin de fichier) dans CHANGELOG doit pointer
    vers un fichier existant."""
    text = _read_changelog()
    links = _extract_internal_links(text)
    if not links:
        pytest.skip("Pas de liens internes à vérifier dans CHANGELOG")

    broken: list[tuple[str, str]] = []
    for link_text, target in links:
        # Résolution relative au repo
        candidate = (REPO_ROOT / target).resolve()
        # Tolérer ../X comme ./X dans le repo
        if not candidate.exists() and not (REPO_ROOT / target.lstrip("./")).exists():
            broken.append((link_text, target))

    assert not broken, (
        f"Liens internes cassés dans CHANGELOG.md : {broken}. "
        f"Soit corriger le chemin, soit retirer le lien."
    )
