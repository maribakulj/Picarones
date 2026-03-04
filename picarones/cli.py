"""Interface en ligne de commande Picarones (Click).

Commandes disponibles
---------------------
picarones run      — Lance un benchmark complet
picarones report   — Génère le rapport HTML depuis un JSON de résultats
picarones demo     — Génère un rapport de démonstration avec données fictives
picarones metrics  — Calcule CER/WER entre deux fichiers texte
picarones engines  — Liste les moteurs disponibles
picarones info     — Informations de version

Exemples d'usage
----------------
    picarones run --corpus ./corpus/ --engines tesseract --output results.json
    picarones metrics --reference gt.txt --hypothesis ocr.txt
    picarones engines
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from picarones import __version__

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
    """Instancie un moteur par son nom."""
    from picarones.engines.tesseract import TesseractEngine

    if engine_name in {"tesseract", "tess"}:
        return TesseractEngine(config={"lang": lang, "psm": psm})

    try:
        from picarones.engines.pero_ocr import PeroOCREngine

        if engine_name in {"pero_ocr", "pero"}:
            return PeroOCREngine(config={"name": "pero_ocr"})
    except ImportError:
        pass

    raise click.BadParameter(
        f"Moteur inconnu ou non disponible : '{engine_name}'. "
        "Moteurs supportés : tesseract, pero_ocr"
    )


# ---------------------------------------------------------------------------
# Groupe principal
# ---------------------------------------------------------------------------

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "-V", "--version", prog_name="picarones")
def cli() -> None:
    """Picarones — Plateforme de comparaison de moteurs OCR pour documents patrimoniaux.

    Bibliothèque nationale de France — Département numérique.
    """


# ---------------------------------------------------------------------------
# picarones run
# ---------------------------------------------------------------------------

@cli.command("run")
@click.option(
    "--corpus", "-c",
    required=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Dossier contenant les paires image / .gt.txt",
)
@click.option(
    "--engines", "-e",
    default="tesseract",
    show_default=True,
    help="Liste de moteurs séparés par des virgules (ex : tesseract,pero_ocr)",
)
@click.option(
    "--output", "-o",
    default="results.json",
    show_default=True,
    type=click.Path(resolve_path=True),
    help="Fichier JSON de sortie",
)
@click.option(
    "--lang", "-l",
    default="fra",
    show_default=True,
    help="Code langue Tesseract (fra, lat, eng…)",
)
@click.option("--psm", default=6, show_default=True, help="Page Segmentation Mode Tesseract (0-13)")
@click.option("--no-progress", is_flag=True, default=False, help="Désactive la barre de progression")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Mode verbeux")
@click.option(
    "--fail-if-cer-above",
    default=None,
    type=float,
    metavar="THRESHOLD",
    help="Quitte avec code 1 si CER moyen > THRESHOLD (usage CI/CD)",
)
def run_cmd(
    corpus: str,
    engines: str,
    output: str,
    lang: str,
    psm: int,
    no_progress: bool,
    verbose: bool,
    fail_if_cer_above: float | None,
) -> None:
    """Lance un benchmark OCR sur un corpus de documents.

    Le corpus doit être un dossier contenant des paires
    <image>.<ext> + <image>.gt.txt (vérité terrain).
    """
    _setup_logging(verbose)

    from picarones.core.corpus import load_corpus_from_directory
    from picarones.core.runner import run_benchmark

    # Chargement du corpus
    try:
        corp = load_corpus_from_directory(corpus)
    except (FileNotFoundError, ValueError) as exc:
        click.echo(f"Erreur corpus : {exc}", err=True)
        sys.exit(1)

    click.echo(f"Corpus '{corp.name}' — {len(corp)} documents chargés.")

    # Instanciation des moteurs
    engine_names = [e.strip() for e in engines.split(",") if e.strip()]
    ocr_engines = []
    for name in engine_names:
        try:
            engine = _engine_from_name(name, lang=lang, psm=psm)
            ocr_engines.append(engine)
        except click.BadParameter as exc:
            click.echo(f"Erreur moteur : {exc}", err=True)
            sys.exit(1)

    if not ocr_engines:
        click.echo("Aucun moteur valide spécifié.", err=True)
        sys.exit(1)

    click.echo(f"Moteurs : {', '.join(e.name for e in ocr_engines)}")

    # Lancement du benchmark
    result = run_benchmark(
        corpus=corp,
        engines=ocr_engines,
        output_json=output,
        show_progress=not no_progress,
    )

    # Affichage du classement
    click.echo("\n── Classement ──────────────────────────────────")
    for rank, entry in enumerate(result.ranking(), 1):
        cer_pct = f"{entry['mean_cer'] * 100:.2f}%" if entry["mean_cer"] is not None else "N/A"
        wer_pct = f"{entry['mean_wer'] * 100:.2f}%" if entry["mean_wer"] is not None else "N/A"
        failed = entry["failed"]
        failed_str = f" ({failed} erreur(s))" if failed else ""
        click.echo(f"  {rank}. {entry['engine']:<20} CER={cer_pct:<8} WER={wer_pct}{failed_str}")

    click.echo(f"\nRésultats écrits dans : {output}")

    # Mode CI/CD : exit code non-zero si CER > seuil
    if fail_if_cer_above is not None:
        for entry in result.ranking():
            if entry["mean_cer"] is not None and entry["mean_cer"] * 100 > fail_if_cer_above:
                click.echo(
                    f"\nECHEC : {entry['engine']} CER={entry['mean_cer']*100:.2f}% "
                    f"> seuil {fail_if_cer_above:.2f}%",
                    err=True,
                )
                sys.exit(1)


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
    from picarones.core.metrics import compute_metrics

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
    click.echo("BnF — Département numérique\n")

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
@click.option("--verbose", "-v", is_flag=True, default=False, help="Mode verbeux")
def report_cmd(results: str, output: str, verbose: bool) -> None:
    """Génère le rapport HTML interactif depuis un fichier JSON de résultats.

    Le rapport est un fichier HTML auto-contenu, lisible hors-ligne,
    avec tableau de classement, galerie, vue document et graphiques.
    """
    _setup_logging(verbose)

    from picarones.report.generator import ReportGenerator

    click.echo(f"Chargement des résultats : {results}")
    try:
        gen = ReportGenerator.from_json(results)
    except Exception as exc:
        click.echo(f"Erreur lors du chargement : {exc}", err=True)
        sys.exit(1)

    click.echo(f"Génération du rapport HTML…")
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
def demo_cmd(output: str, docs: int, json_output: str | None) -> None:
    """Génère un rapport de démonstration avec des données fictives réalistes.

    Utile pour tester le rendu HTML sans installer Tesseract ni Pero OCR.
    """
    from picarones.fixtures import generate_sample_benchmark
    from picarones.report.generator import ReportGenerator

    click.echo(f"Génération des données fictives ({docs} documents, 3 moteurs)…")
    benchmark = generate_sample_benchmark(n_docs=docs)

    if json_output:
        bm_path = benchmark.to_json(json_output)
        click.echo(f"Résultats JSON : {bm_path}")

    gen = ReportGenerator(benchmark)
    path = gen.generate(output)
    click.echo(f"Rapport de démonstration : {path}")
    click.echo(f"Ouvrez-le dans un navigateur : file://{path}")


# ---------------------------------------------------------------------------
# picarones import (groupe de sous-commandes)
# ---------------------------------------------------------------------------

@cli.group("import")
def import_group() -> None:
    """Importe un corpus depuis une source distante (IIIF, HuggingFace…)."""


@import_group.command("iiif")
@click.argument("manifest_url")
@click.option(
    "--pages", "-p",
    default="all",
    show_default=True,
    help=(
        "Pages à importer. Formats : '1-10', '1,3,5', '1-5,10,15-20', 'all'. "
        "Les numéros sont 1-based (1 = première page du manifeste)."
    ),
)
@click.option(
    "--output", "-o",
    default="./corpus_iiif/",
    show_default=True,
    type=click.Path(resolve_path=True),
    help="Dossier de destination pour les images et les fichiers .gt.txt",
)
@click.option(
    "--max-resolution",
    default=0,
    type=int,
    show_default=True,
    help="Résolution maximale des images téléchargées (largeur en pixels). 0 = max disponible.",
)
@click.option("--no-progress", is_flag=True, default=False, help="Désactive la barre de progression")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Mode verbeux")
def import_iiif_cmd(
    manifest_url: str,
    pages: str,
    output: str,
    max_resolution: int,
    no_progress: bool,
    verbose: bool,
) -> None:
    """Importe un corpus depuis un manifeste IIIF (v2 ou v3).

    MANIFEST_URL : URL du manifeste IIIF (Gallica, Bodleian, BL, BSB…)

    Exemples :

    \b
        picarones import iiif https://gallica.bnf.fr/ark:/12148/xxx/manifest.json
        picarones import iiif https://gallica.bnf.fr/ark:/12148/xxx/manifest.json --pages 1-10
        picarones import iiif https://gallica.bnf.fr/ark:/12148/xxx/manifest.json --pages 1,3,5-8 --output ./mon_corpus/

    Les images sont téléchargées dans le dossier de sortie.
    Des fichiers .gt.txt vides (ou remplis si le manifeste contient des annotations
    de transcription) sont créés à côté de chaque image.
    """
    _setup_logging(verbose)

    from picarones.importers.iiif import IIIFImporter

    click.echo(f"Manifeste IIIF : {manifest_url}")

    try:
        importer = IIIFImporter(manifest_url, max_resolution=max_resolution)
        importer.load()

        all_canvases = importer.parser.canvases()
        click.echo(
            f"Manifeste IIIF v{importer.parser.version} — "
            f"titre : {importer.parser.label} — "
            f"{len(all_canvases)} canvas disponibles"
        )

        selected = importer.list_canvases(pages)
        click.echo(f"Pages sélectionnées : {len(selected)} sur {len(all_canvases)}")

        corpus = importer.import_corpus(
            pages=pages,
            output_dir=output,
            show_progress=not no_progress,
        )

    except (ValueError, RuntimeError) as exc:
        click.echo(f"Erreur import IIIF : {exc}", err=True)
        sys.exit(1)

    click.echo(f"\n{len(corpus)} documents importés dans : {output}")

    # Résumé
    gt_filled = sum(1 for d in corpus.documents if d.ground_truth.strip())
    if gt_filled:
        click.echo(f"Transcriptions trouvées dans le manifeste : {gt_filled}/{len(corpus)}")
    else:
        click.echo(
            "Aucune transcription dans le manifeste — "
            "les fichiers .gt.txt sont vides (à remplir manuellement ou via OCR)."
        )

    click.echo(f"\nPour lancer un benchmark sur ce corpus :")
    click.echo(f"  picarones run --corpus {output} --engines tesseract")


if __name__ == "__main__":
    cli()
