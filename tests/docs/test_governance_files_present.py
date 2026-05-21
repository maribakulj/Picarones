"""Tests Sprint A10 — gouvernance institutionnelle (M-10 + M-11).

Garde-fou : les 6 fichiers de gouvernance doivent exister, ne pas
être vides, et couvrir les sections clés. Empêche une suppression
accidentelle ou un PR qui viderait l'un d'eux par inadvertance.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


GOVERNANCE_FILES = [
    "CODEOWNERS",          # cherché à la racine ET dans .github/
    "GOVERNANCE.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "LICENSE",
    "CONTRIBUTING.md",
    # Phase 1 D5 : ACCESSIBILITY.md a quitté la racine pour
    # docs/operations/accessibility.md.  Le fichier reste obligatoire
    # mais résolu via le path ``docs/operations/`` (cf. _resolve).
    "accessibility.md",
]


def _resolve(name: str) -> Path | None:
    """Cherche un fichier à la racine, dans ``.github/`` ou
    ``docs/`` (récursif).  La recherche récursive sous ``docs/``
    couvre les fichiers de gouvernance déplacés dans des
    sous-dossiers (operations/, legal/, etc.)."""
    candidates = [
        REPO_ROOT / name,
        REPO_ROOT / ".github" / name,
        REPO_ROOT / "docs" / name,
    ]
    for c in candidates:
        if c.exists():
            return c
    # Recherche récursive sous docs/ pour les fichiers déplacés
    for match in (REPO_ROOT / "docs").rglob(name):
        if match.is_file():
            return match
    return None


@pytest.mark.parametrize("name", GOVERNANCE_FILES)
def test_governance_file_exists(name: str) -> None:
    """Chaque fichier de gouvernance doit exister."""
    path = _resolve(name)
    assert path is not None, (
        f"{name} introuvable — recherché à racine/, .github/, docs/."
    )


@pytest.mark.parametrize("name", GOVERNANCE_FILES)
def test_governance_file_not_empty(name: str) -> None:
    """Chaque fichier de gouvernance doit avoir un contenu non trivial
    (≥ 200 octets — un README dégradé serait < 100 octets)."""
    path = _resolve(name)
    if path is None:
        pytest.skip(f"{name} absent — couvert par test_governance_file_exists")
    assert path.stat().st_size >= 200, (
        f"{path.name} fait seulement {path.stat().st_size} octets — "
        "fichier vide ou tronqué."
    )


def test_codeowners_has_catchall() -> None:
    """``.github/CODEOWNERS`` doit avoir une ligne catch-all ``*  @user``
    pour qu'aucun chemin ne soit sans reviewer."""
    f = REPO_ROOT / ".github" / "CODEOWNERS"
    assert f.exists()
    text = f.read_text(encoding="utf-8")
    # Ligne qui commence par `*` suivi d'au moins un @
    has_catchall = any(
        line.strip().startswith("*") and "@" in line
        for line in text.splitlines()
        if not line.strip().startswith("#") and line.strip()
    )
    assert has_catchall, (
        "CODEOWNERS doit contenir une ligne catch-all `*  @user` "
        "pour garantir un reviewer par défaut."
    )


def test_governance_documents_release_cadence() -> None:
    """``GOVERNANCE.md`` doit documenter la cadence de release."""
    f = REPO_ROOT / "GOVERNANCE.md"
    text = f.read_text(encoding="utf-8")
    for keyword in ["Patch", "Mineure", "Majeure", "release", "SLO"]:
        assert keyword in text, (
            f"GOVERNANCE.md doit mentionner la notion `{keyword}`"
        )


def test_governance_documents_coi() -> None:
    """``GOVERNANCE.md`` doit contenir une section Conflicts of interest
    (item M-10 de l'audit)."""
    f = REPO_ROOT / "GOVERNANCE.md"
    text = f.read_text(encoding="utf-8")
    # Cherche l'une des formulations acceptables
    coi_markers = [
        "Conflicts of interest",
        "Conflits d'intérêt",
        "indépendance",
        "Affiliations des mainteneurs",
    ]
    found = [m for m in coi_markers if m in text]
    assert found, (
        f"GOVERNANCE.md doit contenir une section COI. "
        f"Aucun marqueur trouvé parmi {coi_markers}."
    )


def test_coi_mentions_pricing_independence() -> None:
    """La section COI doit explicitement traiter l'indépendance des
    données de pricing vis-à-vis des fournisseurs benchmarkés."""
    f = REPO_ROOT / "GOVERNANCE.md"
    text = f.read_text(encoding="utf-8")
    assert "pricing" in text.lower()
    # Doit mentionner au moins un fournisseur cloud benchmarké
    providers = ["OpenAI", "Anthropic", "Mistral", "Google", "Azure"]
    found = [p for p in providers if p in text]
    assert len(found) >= 3, (
        f"COI doit citer ≥ 3 fournisseurs cloud benchmarkés ; "
        f"trouvé : {found}"
    )


def test_code_of_conduct_uses_contributor_covenant() -> None:
    """``CODE_OF_CONDUCT.md`` doit s'appuyer sur Contributor Covenant
    (standard de facto, traduit en 30+ langues)."""
    f = REPO_ROOT / "CODE_OF_CONDUCT.md"
    text = f.read_text(encoding="utf-8")
    assert "Contributor Covenant" in text
    # Doit mentionner les 4 niveaux d'application standard
    levels = ["Correction", "Avertissement", "Bannissement"]
    for level in levels:
        assert level in text, f"CoC doit décrire le niveau `{level}`"


def test_governance_links_other_docs() -> None:
    """``GOVERNANCE.md`` doit lier les autres docs de gouvernance pour
    cohérence du système."""
    f = REPO_ROOT / "GOVERNANCE.md"
    text = f.read_text(encoding="utf-8")
    expected_links = [
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "SECURITY.md",
        "accessibility.md",  # déplacé en D5 dans docs/operations/
    ]
    missing = [link for link in expected_links if link not in text]
    assert not missing, (
        f"GOVERNANCE.md doit linker : {missing}"
    )
