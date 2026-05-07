"""Génère les tableaux Markdown du README depuis le code réel.

Sprint A13 (item M-22 / M-23 / M-25 / M-26 du plan de remédiation).

Ce script remplace les listes manuelles qui dérivaient silencieusement
(le bug typique : un nouvel engine ajouté → README pas mis à jour →
``test_readme_consistency`` casse au prochain CI).

Trois tableaux sont produits :

1. **Engines** : un par fichier ``picarones/engines/*.py`` (hors base /
   factory / __init__).
2. **CLI commands** : depuis ``picarones --help``.
3. **API endpoints** : depuis ``app.openapi()["paths"]``.

Le script écrit chaque tableau dans le README entre des balises HTML
``<!-- generated:engines -->`` … ``<!-- /generated:engines -->`` (idem
``cli`` et ``endpoints``). En CI, un job re-exécute ce script et
échoue si le diff Git est non vide — garantissant l'absence de dérive.

Usage :

.. code-block:: bash

    python scripts/gen_readme_tables.py            # met à jour README.md
    python scripts/gen_readme_tables.py --check    # CI : exit 1 si diff
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"

#: Fichiers où ``N tests`` / ``N passed`` est mentionné en prose et
#: doit converger vers le compte réel.  L'audit doc S60 avait
#: identifié 5 chiffres divergents dans 5 docs (1072 / 1244 / 3354 /
#: ~3600 / ~5030).  Liste explicite plutôt qu'un glob — un mainteneur
#: qui ajoute un nouveau doc doit l'inscrire ici consciemment.
TEST_COUNT_FILES: tuple[Path, ...] = (
    README,
    REPO_ROOT / "CLAUDE.md",
    REPO_ROOT / "GOVERNANCE.md",
    REPO_ROOT / "docs" / "developer" / "index.md",
    REPO_ROOT / "docs" / "developer" / "index.en.md",
)

# Permet l'invocation du script en subprocess sans avoir besoin
# d'un ``pip install -e .`` préalable (cas CI / test pytest).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------


_ENGINE_DESCRIPTIONS: dict[str, tuple[str, str, str]] = {
    # name → (display_name, type, install_hint)
    "tesseract": ("Tesseract 5", "Local CLI", "`pip install pytesseract` + system binary"),
    "pero_ocr": ("Pero OCR", "Local Python", "`pip install -e .[pero]`"),
    "mistral_ocr": ("Mistral OCR", "Cloud API", "`MISTRAL_API_KEY` env var"),
    "google_vision": ("Google Vision", "Cloud API", "`GOOGLE_APPLICATION_CREDENTIALS` env var"),
    "azure_doc_intel": ("Azure Doc Intelligence", "Cloud API", "`AZURE_DOC_INTEL_ENDPOINT` + `AZURE_DOC_INTEL_KEY`"),
}


def _engine_files() -> list[str]:
    """Retourne la liste triée des modules d'engines (sans base / factory).

    Lot E (2026-05) : ``picarones/engines/`` a été retiré, son canonique
    est ``picarones/adapters/legacy_engines/``.
    """
    out: list[str] = []
    engines_dir = REPO_ROOT / "picarones" / "adapters" / "legacy_engines"
    for path in sorted(engines_dir.glob("*.py")):
        name = path.stem
        if name in {"__init__", "base", "factory"}:
            continue
        out.append(name)
    return out


def build_engines_table() -> str:
    rows = [
        "| Engine | Type | Installation |",
        "|--------|------|-------------|",
    ]
    for name in _engine_files():
        display, kind, install = _ENGINE_DESCRIPTIONS.get(
            name,
            (name, "Unknown", "—"),
        )
        rows.append(f"| **{display}** | {kind} | {install} |")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


_CLI_DESCRIPTIONS: dict[str, str] = {
    "run": "Run a full benchmark on a corpus",
    "report": "Generate an HTML report from JSON results",
    "demo": "Generate a demo report with synthetic data (no engine required)",
    "metrics": "Compute CER/WER between two text files",
    "engines": "List available OCR engines and LLM adapters",
    "info": "Display version and system information",
    "serve": "Launch the FastAPI web interface",
    "history": "Query longitudinal benchmark history (SQLite)",
    "robustness": "Run robustness analysis with degraded images",
    "import": "Import a corpus from a remote source (IIIF, HF, HTR-United)",
    "compare": "Compare two benchmark JSON runs and flag regressions (Sprint 28)",
    "pipeline": "Run / compare composed pipelines from a YAML spec (Sprint 70)",
    "diagnose": "Pre-wired workflow: bench + improvement levers + factual recommendations",
    "economics": "Pre-wired workflow: bench + effective throughput + cost projection",
    "edition": "Pre-wired workflow: bench + philological metrics for critical editing",
}


def build_cli_table() -> str:
    from picarones.cli import cli

    rows = [
        "| Command | Description |",
        "|---------|-------------|",
    ]
    for name in sorted(cli.commands.keys()):
        desc = _CLI_DESCRIPTIONS.get(name, "—")
        rows.append(f"| `picarones {name}` | {desc} |")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


def build_endpoints_table() -> str:
    from picarones.web.app import app

    spec = app.openapi()
    rows = [
        "| Method | Endpoint | Summary |",
        "|--------|----------|---------|",
    ]
    for path in sorted(spec.get("paths", {})):
        methods = spec["paths"][path]
        for method, definition in sorted(methods.items()):
            if method.upper() not in ("GET", "POST", "PUT", "DELETE", "PATCH"):
                continue
            summary = (
                definition.get("summary")
                or (definition.get("description", "") or "—").split("\n")[0]
            )
            # Tronque à 60 caractères pour le tableau.
            if len(summary) > 80:
                summary = summary[:77] + "…"
            rows.append(
                f"| `{method.upper()}` | `{path}` | {summary} |"
            )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Test count
# ---------------------------------------------------------------------------


def collect_test_count() -> int | None:
    """Lance ``pytest --collect-only`` et extrait le compteur."""
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "--no-cov",
                "-p",
                "no:cacheprovider",
                "tests/",
            ],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return None
    for line in reversed(result.stdout.strip().split("\n")):
        m = re.search(r"(\d+)\s+tests?\s+collected", line)
        if m:
            return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Insertion dans le README
# ---------------------------------------------------------------------------


def _replace_section(text: str, marker: str, content: str) -> str:
    """Remplace le contenu entre ``<!-- generated:<marker> -->`` et
    ``<!-- /generated:<marker> -->`` ; conserve le reste du fichier
    intact. Si les balises sont absentes, retourne le texte inchangé
    (le README doit être mis à jour avec les balises au moins une fois
    manuellement avant que ce script puisse opérer)."""
    pattern = re.compile(
        rf"(<!--\s*generated:{marker}\s*-->)(.*?)(<!--\s*/generated:{marker}\s*-->)",
        re.DOTALL,
    )
    replacement = f"\\1\n\n{content}\n\n\\3"
    new_text, n = pattern.subn(replacement, text)
    if n == 0:
        sys.stderr.write(
            f"[gen_readme_tables] Marqueurs <!-- generated:{marker} --> "
            f"absents du README — section non mise à jour.\n"
        )
        return text
    return new_text


def _replace_test_count(text: str, count: int) -> str:
    """Remplace les mentions ``N tests`` ou ``N passed`` qui citent un
    nombre dans la fenêtre [count*0.5, count*2]. Garde la formulation
    exacte (espace, ponctuation) intacte.

    Le count est **arrondi à la dizaine inférieure** pour rendre le
    résultat OS-déterministe : sur Windows certains tests POSIX-only
    sont skipés (cf. ``pytest.importorskip``) ce qui décale le
    compteur de quelques unités.  Le floor à la dizaine absorbe ces
    écarts mineurs sans masquer une vraie évolution (le seuil de
    tolérance des tests consistency reste à ±5 %).

    Note : utilise ``(count // 10) * 10`` plutôt que
    ``round(count, -1)``.  Le ``round()`` Python applique le
    "banker's rounding" (round half to even) qui n'est pas
    monotone — un écart d'1 test entre Linux et Windows peut
    produire des dizaines différentes (ex : 5035 → 5040 sur Linux,
    5034 → 5030 sur Windows), faisant échouer le test
    ``test_readme_tables_consistent_with_code``.
    """
    rounded_count = (count // 10) * 10

    def _sub(match: re.Match) -> str:
        cited = int(match.group(1))
        # Ne touche pas si le nombre cité est complètement hors plage —
        # c'est probablement une autre référence (un chiffre dans une
        # phrase qui parle d'autre chose).
        if cited < count * 0.5 or cited > count * 2:
            return match.group(0)
        return match.group(0).replace(str(cited), str(rounded_count))

    return re.sub(r"(\d{3,5})\s+(?:tests|passed)\b", _sub, text)


def render_readme(check_only: bool = False) -> int:
    """Met à jour les sections générées du README. Retourne 0 ou 1."""
    if not README.exists():
        sys.stderr.write(f"README absent : {README}\n")
        return 1

    original = README.read_text(encoding="utf-8")
    text = original
    text = _replace_section(text, "engines", build_engines_table())
    text = _replace_section(text, "cli", build_cli_table())
    text = _replace_section(text, "endpoints", build_endpoints_table())

    count = collect_test_count()
    if count is not None:
        text = _replace_test_count(text, count)

    if check_only:
        if text != original:
            sys.stderr.write(
                "[gen_readme_tables] README divergent du code généré. "
                "Lancer ``python scripts/gen_readme_tables.py`` puis "
                "committer.\n"
            )
            return 1
        return 0

    if text != original:
        README.write_text(text, encoding="utf-8")
        print(f"[gen_readme_tables] README mis à jour ({len(text)} octets).")
    else:
        print("[gen_readme_tables] README déjà à jour.")
    return 0


def render_test_counts(check_only: bool = False) -> int:
    """Synchronise le compte de tests dans tous les ``TEST_COUNT_FILES``.

    Audit doc S60 : 5 chiffres divergents (1072 / 1244 / 3354 /
    ~3600 / ~5030) selon les docs.  Cette fonction lit le compte
    réel via ``pytest --collect-only`` et l'injecte dans chaque
    fichier de la liste.

    Returns
    -------
    int
        0 si tout est synchronisé, 1 si divergence (en mode check)
        ou erreur d'écriture.
    """
    count = collect_test_count()
    if count is None:
        # ``pytest --collect-only`` indisponible (env CI minimal,
        # virtualenv dégradé).  On ne casse pas le build pour ça.
        sys.stderr.write(
            "[gen_readme_tables] collect_test_count indisponible — "
            "skip mise à jour des compteurs de tests.\n",
        )
        return 0

    divergent = False
    for path in TEST_COUNT_FILES:
        if not path.exists():
            continue
        original = path.read_text(encoding="utf-8")
        updated = _replace_test_count(original, count)
        if updated == original:
            continue
        divergent = True
        if check_only:
            sys.stderr.write(
                f"[gen_readme_tables] {path.relative_to(REPO_ROOT)} "
                "diverge du compteur de tests réel.\n",
            )
        else:
            path.write_text(updated, encoding="utf-8")
            print(
                f"[gen_readme_tables] {path.relative_to(REPO_ROOT)} "
                "test count mis à jour.",
            )
    if check_only and divergent:
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="N'écrit rien ; sort 1 si le README diverge du code généré.",
    )
    args = parser.parse_args()
    rc_readme = render_readme(check_only=args.check)
    rc_counts = render_test_counts(check_only=args.check)
    return rc_readme or rc_counts


if __name__ == "__main__":
    sys.exit(main())
