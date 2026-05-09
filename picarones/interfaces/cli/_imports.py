"""Commandes d'import depuis sources externes : IIIF, HTR-United, ...

Sous-module CLI extrait de l'ancien ``picarones/cli.py`` (1519 lignes)
lors du chantier 5 post-Sprint 97.  Les commandes ici s'enregistrent
automatiquement sur le groupe ``cli`` à l'import.

Comportement et signatures inchangés — uniquement de la modularisation.
"""

from __future__ import annotations

import sys

import click

from picarones.interfaces.cli import cli, _setup_logging

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

    from picarones.adapters.corpus.iiif import IIIFImporter

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

    click.echo("\nPour lancer un benchmark sur ce corpus :")
    click.echo(f"  picarones run --corpus {output} --engines tesseract")
