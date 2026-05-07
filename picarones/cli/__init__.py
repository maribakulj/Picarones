"""Interface en ligne de commande Picarones (Click).

Commandes disponibles
---------------------
picarones run         — Lance un benchmark complet
picarones report      — Génère le rapport HTML depuis un JSON de résultats
picarones demo        — Génère un rapport de démonstration avec données fictives
picarones metrics     — Calcule CER/WER entre deux fichiers texte
picarones engines     — Liste les moteurs disponibles
picarones info        — Informations de version
picarones history     — Consulte l'historique des benchmarks (suivi longitudinal)
picarones robustness  — Lance une analyse de robustesse sur un corpus

Exemples d'usage
----------------
    picarones run --corpus ./corpus/ --engines tesseract --output results.json
    picarones metrics --reference gt.txt --hypothesis ocr.txt
    picarones history --engine tesseract
    picarones robustness --corpus ./gt/ --engine tesseract
    picarones engines
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click

from picarones import __version__

if TYPE_CHECKING:
    from picarones.adapters.legacy_engines.base import BaseOCREngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


def _engine_from_name(engine_name: str, lang: str, psm: int) -> "BaseOCREngine":
    """Instancie un moteur OCR par son nom (wrapper Click).

    Délègue à :func:`picarones.adapters.legacy_engines.factory.engine_from_name`
    (couche adapters) puis traduit toute ``ValueError`` en
    ``click.BadParameter`` pour rester compatible avec les sous-modules
    CLI (``_workflows.py``, ``_robustness.py``) qui attrappent ce type
    d'exception. Les commandes CLI continuent donc à fonctionner sans
    aucune modification.
    """
    from picarones.adapters.legacy_engines.factory import engine_from_name

    try:
        return engine_from_name(engine_name, lang=lang, psm=psm)
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc


# ---------------------------------------------------------------------------
# Groupe principal
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Groupe principal
# ---------------------------------------------------------------------------

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "-V", "--version", prog_name="picarones")
def cli() -> None:
    """Picarones — Plateforme de comparaison de moteurs OCR pour documents patrimoniaux."""


# ---------------------------------------------------------------------------
# picarones metrics
# ---------------------------------------------------------------------------

@cli.command("metrics")
@click.option(
    "--reference", "-r",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Fichier vérité terrain (texte brut UTF-8)",
)
@click.option(
    "--hypothesis", "-H",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Fichier transcription OCR (texte brut UTF-8)",
)
@click.option("--json-output", is_flag=True, default=False, help="Sortie en JSON")
def metrics_cmd(reference: str, hypothesis: str, json_output: bool) -> None:
    """Calcule CER et WER entre deux fichiers texte."""
    from picarones.measurements.metrics import compute_metrics

    ref_text = Path(reference).read_text(encoding="utf-8").strip()
    hyp_text = Path(hypothesis).read_text(encoding="utf-8").strip()

    result = compute_metrics(ref_text, hyp_text)

    if json_output:
        click.echo(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    else:
        click.echo(f"CER            : {result.cer_percent:.2f}%")
        click.echo(f"CER (NFC)      : {result.cer_nfc * 100:.2f}%")
        click.echo(f"CER (caseless) : {result.cer_caseless * 100:.2f}%")
        click.echo(f"WER            : {result.wer_percent:.2f}%")
        click.echo(f"WER (normalisé): {result.wer_normalized * 100:.2f}%")
        click.echo(f"MER            : {result.mer * 100:.2f}%")
        click.echo(f"WIL            : {result.wil * 100:.2f}%")
        click.echo(f"Longueur GT    : {result.reference_length} chars")
        click.echo(f"Longueur OCR   : {result.hypothesis_length} chars")
        if result.error:
            click.echo(f"Erreur         : {result.error}", err=True)


# ---------------------------------------------------------------------------
# picarones engines
# ---------------------------------------------------------------------------

@cli.command("engines")
def engines_cmd() -> None:
    """Liste les moteurs OCR disponibles et vérifie leur installation."""
    engines = [
        ("tesseract", "Tesseract 5 (pytesseract)", "pytesseract"),
        ("pero_ocr", "Pero OCR", "pero_ocr"),
    ]

    click.echo("Moteurs OCR disponibles :\n")
    for engine_id, label, module in engines:
        try:
            __import__(module)
            status = click.style("✓ disponible", fg="green")
        except ImportError:
            status = click.style("✗ non installé", fg="red")
        click.echo(f"  {engine_id:<15} {label:<35} {status}")

    click.echo(
        "\nPour installer un moteur manquant :\n"
        "  pip install pytesseract\n"
        "  pip install pero-ocr"
    )


# ---------------------------------------------------------------------------
# picarones info
# ---------------------------------------------------------------------------

@cli.command("info")
def info_cmd() -> None:
    """Affiche les informations de version de Picarones et de ses dépendances."""
    click.echo(f"Picarones v{__version__}")
    click.echo("")

    deps = [
        ("click", "click"),
        ("jiwer", "jiwer"),
        ("Pillow", "PIL"),
        ("pytesseract", "pytesseract"),
        ("tqdm", "tqdm"),
        ("numpy", "numpy"),
        ("pyyaml", "yaml"),
    ]

    click.echo("Dépendances :")
    for name, module in deps:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "installé")
            status = click.style(f"v{version}", fg="green")
        except ImportError:
            status = click.style("non installé", fg="red")
        click.echo(f"  {name:<15} {status}")


# ---------------------------------------------------------------------------
# picarones report
# ---------------------------------------------------------------------------

@cli.command("report")
@click.option(
    "--results", "-r",
    required=True,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="Fichier JSON de résultats produit par 'picarones run'",
)
@click.option(
    "--output", "-o",
    default="rapport.html",
    show_default=True,
    type=click.Path(resolve_path=True),
    help="Fichier HTML de sortie",
)
@click.option(
    "--lazy-images/--inline-images",
    default=False,
    show_default=True,
    help=(
        "Sprint A5 (M-16) : si activé, externalise les images dans un dossier "
        "report-assets/ à côté du HTML (au lieu de les embarquer en base64). "
        "Recommandé pour un corpus > 50 documents (rapport monolithique > 100 MB "
        "sinon). Le rapport reste auto-portant si vous copiez aussi report-assets/."
    ),
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Mode verbeux")
def report_cmd(results: str, output: str, lazy_images: bool, verbose: bool) -> None:
    """Génère le rapport HTML interactif depuis un fichier JSON de résultats.

    Le rapport est un fichier HTML auto-contenu, lisible hors-ligne,
    avec tableau de classement, galerie, vue document et graphiques.

    En mode --lazy-images, les images sont externalisées en
    ``report-assets/`` à côté du HTML pour les corpus volumineux.
    """
    _setup_logging(verbose)

    from picarones.reports_v2.html.generator import ReportGenerator

    click.echo(f"Chargement des résultats : {results}")
    try:
        gen = ReportGenerator.from_json(results, lazy_images=lazy_images)
    except Exception as exc:
        click.echo(f"Erreur lors du chargement : {exc}", err=True)
        sys.exit(1)

    click.echo("Génération du rapport HTML…")
    path = gen.generate(output)
    click.echo(f"Rapport généré : {path}")
    click.echo(f"Ouvrez-le dans un navigateur : file://{path}")


# ---------------------------------------------------------------------------
# picarones demo
# ---------------------------------------------------------------------------

@cli.command("demo")
@click.option(
    "--output", "-o",
    default="rapport_demo.html",
    show_default=True,
    type=click.Path(resolve_path=True),
    help="Fichier HTML de sortie",
)
@click.option(
    "--docs", "-n",
    default=12,
    show_default=True,
    type=click.IntRange(1, 12),
    help="Nombre de documents fictifs (1–12)",
)
@click.option(
    "--json-output", "-j",
    default=None,
    type=click.Path(resolve_path=True),
    help="Exporte aussi les résultats JSON",
)
@click.option(
    "--with-history",
    is_flag=True,
    default=False,
    help="Inclut une démonstration du suivi longitudinal (8 runs fictifs)",
)
@click.option(
    "--with-robustness",
    is_flag=True,
    default=False,
    help="Inclut une démonstration de l'analyse de robustesse",
)
@click.option(
    "--lang",
    default="fr",
    show_default=True,
    type=click.Choice(["fr", "en"], case_sensitive=False),
    help="Langue du rapport HTML généré (fr = français, en = anglais patrimonial)",
)
def demo_cmd(
    output: str,
    docs: int,
    json_output: str | None,
    with_history: bool,
    with_robustness: bool,
    lang: str,
) -> None:
    """Génère un rapport de démonstration avec des données fictives réalistes.

    Utile pour tester le rendu HTML sans installer Tesseract ni Pero OCR.

    \b
    Exemples :
        picarones demo
        picarones demo --lang en
        picarones demo --with-history
        picarones demo --with-robustness
        picarones demo --with-history --with-robustness --docs 8
    """
    from picarones.fixtures import generate_sample_benchmark
    from picarones.reports_v2.html.generator import ReportGenerator

    click.echo(f"Génération des données fictives ({docs} documents, 3 moteurs)…")
    benchmark = generate_sample_benchmark(n_docs=docs)

    if json_output:
        bm_path = benchmark.to_json(json_output)
        click.echo(f"Résultats JSON : {bm_path}")

    gen = ReportGenerator(benchmark, lang=lang)
    path = gen.generate(output)
    click.echo(f"Rapport de démonstration : {path}")
    click.echo(f"Ouvrez-le dans un navigateur : file://{path}")

    # Suivi longitudinal
    if with_history:
        click.echo("\n── Démonstration suivi longitudinal ──────────────")
        from picarones.measurements.history import BenchmarkHistory, generate_demo_history
        history = BenchmarkHistory(":memory:")
        generate_demo_history(history, n_runs=8)
        entries = history.query(engine="tesseract")
        click.echo(f"  {history.count()} entrées générées (8 runs, 3 moteurs).")
        click.echo("\n  Évolution du CER — tesseract :")
        for e in entries:
            cer_str = f"{e.cer_percent:.2f}%" if e.cer_percent is not None else "N/A"
            bar = "█" * int((e.cer_percent or 0) * 2)
            click.echo(f"    {e.timestamp[:10]}  {cer_str:<8}  {bar}")
        regression = history.detect_regression("tesseract", threshold=0.01)
        if regression and regression.is_regression:
            click.echo(
                click.style(
                    f"\n  RÉGRESSION détectée ! delta CER = +{regression.delta_cer * 100:.2f}%",
                    fg="red",
                )
            )
        else:
            click.echo(click.style("\n  Aucune régression détectée.", fg="green"))

    # Analyse de robustesse
    if with_robustness:
        click.echo("\n── Démonstration analyse de robustesse ───────────")
        from picarones.measurements.robustness import generate_demo_robustness_report
        report = generate_demo_robustness_report(
            engine_names=["tesseract", "pero_ocr"]
        )
        for curve in report.curves:
            if curve.degradation_type == "noise":
                click.echo(f"\n  {curve.engine_name} / bruit gaussien :")
                for label, cer in zip(curve.labels, curve.cer_values):
                    cer_pct = f"{(cer or 0) * 100:.1f}%"
                    bar = "█" * int((cer or 0) * 40)
                    click.echo(f"    {label:<12} {cer_pct:<8} {bar}")
                if curve.critical_threshold_level is not None:
                    click.echo(
                        click.style(
                            f"    Niveau critique (CER>20%) : σ={curve.critical_threshold_level}",
                            fg="yellow",
                        )
                    )


# ---------------------------------------------------------------------------
# Imports en cascade des sous-modules — chantier 5 post-Sprint 97
# ---------------------------------------------------------------------------
#
# Chaque sous-module enregistre ses commandes via ``@cli.command(...)`` au
# chargement.  L'import ici suffit à peupler le groupe principal ``cli``.
# Ordre des imports : aucun n'a de dépendance entre eux, seul le groupe
# ``cli`` (déjà créé ci-dessus) est partagé.

from picarones.cli import _workflows   # noqa: F401, E402  (run, diagnose, economics, edition, compare)
from picarones.cli import _imports     # noqa: F401, E402  (import group + import_iiif)
from picarones.cli import _serve       # noqa: F401, E402  (serve)
from picarones.cli import _history     # noqa: F401, E402  (history)
from picarones.cli import _robustness  # noqa: F401, E402  (robustness)


__all__ = [
    "cli",
    "_setup_logging",
    "_engine_from_name",
]
