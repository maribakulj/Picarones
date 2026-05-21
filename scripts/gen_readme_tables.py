"""Génère les tableaux Markdown du README depuis le code réel.

Le script remplace les listes manuelles qui dérivaient silencieusement
(le bug typique : un nouvel engine ajouté → README pas mis à jour →
``test_readme_consistency`` casse au prochain CI).

Trois tableaux sont produits :

1. **Engines** : un par adapter sous ``picarones/adapters/ocr/`` (hors
   base / factory / __init__).
2. **CLI commands** : depuis ``picarones --help``.
3. **API endpoints** : depuis ``app.openapi()["paths"]``.

Le script écrit chaque tableau dans le README entre des balises HTML
``<!-- generated:engines -->`` … ``<!-- /generated:engines -->`` (idem
``cli`` et ``endpoints``). En CI, un job re-exécute ce script et
échoue si le diff Git est non vide — garantissant l'absence de dérive.

Le compteur de tests n'est PAS géré ici : il dérivait selon l'OS et
les binaires système installés (4509 vs 4510 selon que tesseract est
présent ou non), donc on l'a sorti de la prose.  La règle actuelle :
le README dit ``5000+ tests`` (formulation non quantifiée) et le
chiffre exact vit dans le badge CI / Codecov.

Usage :

.. code-block:: bash

    python scripts/gen_readme_tables.py            # met à jour README.md
    python scripts/gen_readme_tables.py --check    # CI : exit 1 si diff
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"

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
    "kraken": ("Kraken HTR", "Local Python", "`pip install -e .[kraken]` + modèle `.mlmodel`"),
    "calamari": ("Calamari OCR", "Local Python", "`pip install -e .[calamari]` + checkpoint"),
    "mistral_ocr": ("Mistral OCR", "Cloud API", "`MISTRAL_API_KEY` env var"),
    "google_vision": ("Google Vision", "Cloud API", "`GOOGLE_APPLICATION_CREDENTIALS` env var"),
    "azure_doc_intel": ("Azure Doc Intelligence", "Cloud API", "`AZURE_DOC_INTEL_ENDPOINT` + `AZURE_DOC_INTEL_KEY`"),
}


def _engine_files() -> list[str]:
    """Retourne la liste triée des modules d'OCR engines (sans helpers).

    Sprint H.2.d (2026-05) : ``picarones/adapters/legacy_engines/`` a été
    supprimé, le canonique est ``picarones/adapters/ocr/``.  On filtre
    aussi les modules helpers (``confidences``, ``precomputed``) qui ne
    sont pas des engines OCR à proprement parler.
    """
    out: list[str] = []
    engines_dir = REPO_ROOT / "picarones" / "adapters" / "ocr"
    skip = {"__init__", "base", "factory", "confidences", "precomputed"}
    for path in sorted(engines_dir.glob("*.py")):
        name = path.stem
        if name in skip:
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
    from picarones.interfaces.cli import cli

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
    from picarones.interfaces.web.app import app

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


def render_readme(check_only: bool = False) -> int:
    """Met à jour les sections générées du README. Retourne 0 ou 1.

    Le compteur de tests n'est plus injecté en prose : il dérivait
    selon l'OS et les binaires système installés, et la stratégie
    actuelle est ``5000+ tests`` (formulation non quantifiée) avec le
    chiffre exact porté par le badge CI.
    """
    if not README.exists():
        sys.stderr.write(f"README absent : {README}\n")
        return 1

    original = README.read_text(encoding="utf-8")
    text = original
    text = _replace_section(text, "engines", build_engines_table())
    text = _replace_section(text, "cli", build_cli_table())
    text = _replace_section(text, "endpoints", build_endpoints_table())

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="N'écrit rien ; sort 1 si le README diverge du code généré.",
    )
    args = parser.parse_args()
    return render_readme(check_only=args.check)


if __name__ == "__main__":
    sys.exit(main())
