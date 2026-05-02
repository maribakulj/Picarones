"""Tests Sprint A13 — bilinguisme du README + auto-génération.

Items M-22 (project structure), M-25 (CLI table), M-26 (API table)
de l'audit institutional-readiness-2026-05.

Ces tests valident :

1. Le README contient à la fois les taglines anglaise et française
   (B-13 : taglines cassées par markdown invalide → réparées).
2. Les balises ``<!-- generated:* -->`` qui délimitent les sections
   auto-générées sont bien en place et cohérentes.
3. Le contenu généré par ``scripts/gen_readme_tables.py`` correspond
   à l'état du code (engines, CLI commands, endpoints) — équivalent
   au mode ``--check`` du script.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"
GEN_SCRIPT = REPO_ROOT / "scripts" / "gen_readme_tables.py"


def _read_readme() -> str:
    return README.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# B-13 — taglines bilingues réparées
# ---------------------------------------------------------------------------


def test_taglines_have_closing_bold() -> None:
    """Les deux taglines en blockquote doivent être correctement
    fermées (B-13 : avant Sprint A13 le ``> **`` n'était jamais
    fermé, taglines tronquées à mi-ligne)."""
    text = _read_readme()
    # Cherche les blockquotes dans la zone d'en-tête (avant le premier ## )
    header = text.split("\n## ")[0]
    blockquotes = [
        line for line in header.splitlines()
        if line.startswith("> ")
    ]
    bold_blocks = [b for b in blockquotes if "**" in b]
    assert bold_blocks, "Au moins un blockquote en gras attendu en tête"
    for b in bold_blocks:
        # Compte le nombre de paires ``**`` — doit être pair
        n_stars = b.count("**")
        assert n_stars > 0 and n_stars % 2 == 0, (
            f"Blockquote avec ``**`` non fermé : {b!r}"
        )


def test_readme_has_french_section() -> None:
    """Le README doit garder une section française pour la
    rétrocompat (B-13 : les deux taglines + section ``En français``)."""
    text = _read_readme()
    assert "En français" in text or "En Français" in text, (
        "Section française attendue (### En français)"
    )
    # Vérifie que le mot ``patrimoniaux`` apparaît (sceau d'un texte
    # français cohérent)
    assert "patrimoniaux" in text


def test_readme_has_english_section() -> None:
    """Le README doit avoir une intro anglaise pour les lecteurs
    internationaux."""
    text = _read_readme()
    assert "What is Picarones" in text or "Picarones is " in text, (
        "Intro anglaise attendue"
    )


# ---------------------------------------------------------------------------
# Balises de génération
# ---------------------------------------------------------------------------


GENERATED_MARKERS = ["engines", "cli", "endpoints"]


def test_generated_markers_paired() -> None:
    """Chaque ``<!-- generated:X -->`` doit avoir son
    ``<!-- /generated:X -->`` jumeau."""
    text = _read_readme()
    for marker in GENERATED_MARKERS:
        opens = text.count(f"<!-- generated:{marker} -->")
        closes = text.count(f"<!-- /generated:{marker} -->")
        assert opens == closes == 1, (
            f"Balise ``generated:{marker}`` mal appariée : "
            f"{opens} ouverture(s), {closes} fermeture(s)."
        )


def test_generated_sections_not_empty() -> None:
    """Le contenu entre les balises doit être non vide (si vide,
    le générateur n'a jamais tourné)."""
    text = _read_readme()
    for marker in GENERATED_MARKERS:
        pattern = re.compile(
            rf"<!--\s*generated:{marker}\s*-->(.*?)<!--\s*/generated:{marker}\s*-->",
            re.DOTALL,
        )
        m = pattern.search(text)
        assert m is not None, f"Section ``{marker}`` introuvable"
        content = m.group(1).strip()
        assert len(content) > 50, (
            f"Section ``{marker}`` trop courte : {len(content)} caractères. "
            "Lancer ``python scripts/gen_readme_tables.py``."
        )


# ---------------------------------------------------------------------------
# scripts/gen_readme_tables.py --check
# ---------------------------------------------------------------------------


def test_gen_readme_tables_script_exists() -> None:
    """Le script de génération doit exister et être exécutable."""
    assert GEN_SCRIPT.exists(), (
        f"{GEN_SCRIPT} manquant — pas de gate anti-dérive en CI."
    )


def test_readme_tables_consistent_with_code() -> None:
    """``python scripts/gen_readme_tables.py --check`` doit retourner 0
    (le README est synchronisé avec le code)."""
    result = subprocess.run(
        [sys.executable, str(GEN_SCRIPT), "--check"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )
    assert result.returncode == 0, (
        "Le README diverge du contenu généré par scripts/gen_readme_tables.py.\n"
        f"stdout: {result.stdout[-500:]}\n"
        f"stderr: {result.stderr[-500:]}\n"
        "Lancer ``python scripts/gen_readme_tables.py`` puis committer."
    )


# ---------------------------------------------------------------------------
# Footer / métadonnées
# ---------------------------------------------------------------------------


def test_copyright_year_range() -> None:
    """Le copyright doit refléter une plage d'années (pas seulement 2024
    — m-18 de l'audit)."""
    text = _read_readme()
    assert re.search(r"Copyright\s+202[0-9]\s*[-–]\s*202[5-9]", text), (
        "Copyright doit être au format ``Copyright YYYY-YYYY`` ; "
        "actuellement absent ou figé à une seule année."
    )


def test_readme_under_500_lines() -> None:
    """Le README doit rester compact (Sprint A13 vise < 500 lignes,
    versus 786 avant la refonte)."""
    text = _read_readme()
    n_lines = len(text.splitlines())
    assert n_lines < 500, (
        f"README à {n_lines} lignes — au-dessus du seuil 500. "
        "Déléguer le détail vers docs/."
    )


def test_readme_links_to_audits() -> None:
    """Le README doit pointer vers ``docs/audits/`` pour la traçabilité
    des plans de remédiation (M-21 : suppression de Known Issues
    obsolète + redirection vers audits)."""
    text = _read_readme()
    assert "docs/audits" in text, (
        "README doit linker vers docs/audits/ pour les rapports "
        "d'audit institutionnels."
    )


def test_readme_links_to_governance() -> None:
    """Le README doit linker GOVERNANCE, CONTRIBUTING, CoC."""
    text = _read_readme()
    expected = ["GOVERNANCE.md", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "SECURITY.md"]
    missing = [m for m in expected if m not in text]
    assert not missing, (
        f"README doit linker : {missing}"
    )


def test_no_obsolete_known_issues_section() -> None:
    """La section ``Known Issues`` (héritée de Sprint 22) doit avoir
    été retirée — son contenu était obsolète (références à
    web/app.py 3072 lignes alors que c'est 131, etc.)."""
    text = _read_readme()
    assert "Known Issues & Improvement Opportunities" not in text, (
        "Section ``Known Issues`` héritée doit être supprimée — "
        "remplacée par un lien vers docs/audits/."
    )
