"""Tests Sprint A11 — parité de traduction FR/EN (item M-17).

Garde-fou : pour chaque fichier ``X.md`` listé comme prioritaire,
la version EN ``X.en.md`` doit exister et avoir des sections de
premier niveau (``##``) qui se correspondent. Empêche une
divergence éditoriale silencieuse entre les deux versions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


# Surface EN canonique du projet.  La politique éditoriale (cf.
# ``docs/index.md``) est : langue canonique = français, surface EN
# réduite à l'entrée du projet (README), aux fichiers institutionnels
# (CONTRIBUTING, SECURITY) et au tutoriel le plus exposé
# (reading-a-report).  Tout autre fichier ``X.en.md`` créé sans
# être inscrit ici est considéré comme orpheline et le test
# ``test_no_orphan_en_md`` ci-dessous le rejette.
TRANSLATION_PAIRS: list[tuple[str, str]] = [
    ("docs/tutorials/reading-a-report.md", "docs/tutorials/reading-a-report.en.md"),
    ("SECURITY.md", "SECURITY.en.md"),
    ("CONTRIBUTING.md", "CONTRIBUTING.en.md"),
]


def _h2_titles(path: Path) -> list[str]:
    """Retourne la liste ordonnée des titres ``##`` d'un fichier markdown."""
    text = path.read_text(encoding="utf-8")
    return [
        line[3:].strip()
        for line in text.splitlines()
        if line.startswith("## ") and not line.startswith("###")
    ]


@pytest.mark.parametrize("fr_path,en_path", TRANSLATION_PAIRS)
def test_en_version_exists(fr_path: str, en_path: str) -> None:
    """Pour chaque paire, la version EN doit exister."""
    fr = REPO_ROOT / fr_path
    en = REPO_ROOT / en_path
    if not fr.exists():
        pytest.skip(f"Source FR absente : {fr_path}")
    assert en.exists(), f"Traduction EN manquante : {en_path}"


@pytest.mark.parametrize("fr_path,en_path", TRANSLATION_PAIRS)
def test_en_version_marks_translation_status(
    fr_path: str, en_path: str
) -> None:
    """La version EN doit marquer son statut (translation: ...) en
    tête, pour qu'un lecteur sache que la version FR fait foi en
    cas de divergence."""
    en = REPO_ROOT / en_path
    if not en.exists():
        pytest.skip(f"EN manquant : {en_path}")
    text = en.read_text(encoding="utf-8")
    assert "translation:" in text, (
        f"{en_path} doit contenir un marqueur HTML "
        "<!-- translation: ... --> en tête."
    )
    # Doit aussi linker vers la version FR canonique
    fr_basename = Path(fr_path).name
    assert fr_basename in text, (
        f"{en_path} doit linker vers la version FR canonique "
        f"({fr_basename})."
    )


@pytest.mark.parametrize("fr_path,en_path", TRANSLATION_PAIRS)
def test_section_count_consistent(fr_path: str, en_path: str) -> None:
    """Les versions FR et EN doivent avoir un nombre cohérent de
    sections de premier niveau (``##``).

    Tolérances :

    - **±1 section** par défaut (légère adaptation rédactionnelle).
    - **±5 sections** si la version EN porte le marqueur
      ``<!-- translation: machine + human review pending -->``
      (Sprint A11 livre des traductions synthétiques que le sprint
      A15 ou un sprint dédié de revue humaine alignera ensuite).

    Au-delà, c'est une divergence sérieuse à corriger.
    """
    fr = REPO_ROOT / fr_path
    en = REPO_ROOT / en_path
    if not (fr.exists() and en.exists()):
        pytest.skip(f"Une des deux versions manque : {fr_path} / {en_path}")
    fr_sections = _h2_titles(fr)
    en_sections = _h2_titles(en)
    diff = abs(len(fr_sections) - len(en_sections))

    en_text = en.read_text(encoding="utf-8")
    is_pending_review = "machine + human review pending" in en_text
    threshold = 5 if is_pending_review else 1
    assert diff <= threshold, (
        f"Divergence de structure trop forte : FR a {len(fr_sections)} sections, "
        f"EN a {len(en_sections)} (diff {diff} > {threshold}). "
        f"Pending-review : {is_pending_review}.\n"
        f"FR: {fr_sections}\nEN: {en_sections}"
    )


def test_no_orphan_en_translations() -> None:
    """Aucun fichier ``*.en.md`` ne doit exister sans son pendant FR
    canonique (sinon dérive éditoriale)."""
    en_files = list(REPO_ROOT.glob("**/*.en.md"))
    # Exclure les __pycache__ et venv
    en_files = [
        f for f in en_files
        if not any(part.startswith(".") or part == "__pycache__"
                   for part in f.parts)
    ]
    orphans: list[str] = []
    for en in en_files:
        # Pendant FR : remplacer .en.md → .md
        fr_name = en.name.replace(".en.md", ".md")
        fr = en.with_name(fr_name)
        if not fr.exists():
            orphans.append(str(en.relative_to(REPO_ROOT)))
    assert not orphans, (
        f"Traductions EN orphelines (sans version FR) : {orphans}"
    )


def test_translation_pairs_listed_actually_exist() -> None:
    """Méta-test : tous les couples listés ci-dessus doivent référencer
    des fichiers existants (au moins le FR)."""
    missing: list[str] = []
    for fr, _ in TRANSLATION_PAIRS:
        if not (REPO_ROOT / fr).exists():
            missing.append(fr)
    assert not missing, (
        f"Fichiers FR listés mais absents : {missing}. "
        f"Mettre à jour TRANSLATION_PAIRS si la doc canonique a bougé."
    )


def test_no_en_md_outside_translation_pairs() -> None:
    """Tout fichier ``X.en.md`` présent dans le repo doit être listé
    dans ``TRANSLATION_PAIRS``.

    Sans ce test, on peut créer une traduction EN « libre » sans
    contrat éditorial (pas de marqueur ``translation:``, pas de
    parité de structure vérifiée).  C'est exactement la classe de
    fuite qui avait laissé survivre ``extending-glossary.en.md``,
    ``extending-i18n.en.md``, ``developer/index.en.md`` et
    ``narrative-engine.en.md`` en dehors de la politique annoncée.

    Effet : ajouter un ``.en.md`` exige d'ajouter sa paire dans
    ``TRANSLATION_PAIRS``, ce qui force le marqueur et la parité.
    """
    listed_en = {en for _, en in TRANSLATION_PAIRS}

    en_files: list[str] = []
    for en in REPO_ROOT.glob("**/*.en.md"):
        if any(part.startswith(".") or part == "__pycache__"
               for part in en.parts):
            continue
        # Exclure les archives — elles peuvent contenir des EN
        # historiques sans engagement éditorial.
        rel = en.relative_to(REPO_ROOT).as_posix()
        if "docs/archive/" in rel or "docs/archives/" in rel:
            continue
        en_files.append(rel)

    extra = sorted(set(en_files) - listed_en)
    assert not extra, (
        "Fichiers ``.en.md`` présents mais NON listés dans "
        "``TRANSLATION_PAIRS`` :\n  "
        + "\n  ".join(extra)
        + "\n\n→ Soit les inscrire dans TRANSLATION_PAIRS (et leur "
        "ajouter un marqueur ``<!-- translation: ... -->``), soit "
        "les supprimer.  La politique éditoriale n'autorise pas de "
        "traduction EN « libre »."
    )
