"""``picarones-rewrite import-corpus`` — extraction sandboxée d'un ZIP.

Sprint A14-S22.

Wrapper CLI minimal autour du ``CorpusService`` (S20) :

::

    python -m picarones.app.cli import-corpus mon_corpus.zip \\
        --output-dir ./workspaces/sess1 \\
        --corpus-name bnf_xviiie \\
        --metadata language=fr \\
        --metadata period=early_modern

Comportement
------------
- Lit le ZIP (path utilisateur, sans validation préalable — la CLI
  fait confiance au filesystem local de l'opérateur).
- Crée un ``WorkspaceManager`` dans ``--output-dir`` (créé s'il
  n'existe pas).
- Appelle ``CorpusService.import_zip``.
- Affiche un résumé lisible : n_documents, n_images sans GT, GT
  orphelines, warnings.
- Code de sortie ``0`` succès, ``1`` erreur typée
  (``CorpusImportError``), ``2`` erreur d'usage Click (gérée par
  Click).
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from picarones.app.services import (
    CorpusImportError,
    CorpusService,
    WorkspaceManager,
)


@click.command()
@click.argument(
    "zip_path",
    type=click.Path(
        exists=True, dir_okay=False, file_okay=True, path_type=Path,
    ),
)
@click.option(
    "--output-dir",
    "output_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help=(
        "Répertoire parent où créer le workspace sandboxé.  Créé "
        "s'il n'existe pas."
    ),
)
@click.option(
    "--corpus-name",
    default=None,
    help=(
        "Nom du corpus (défaut : nom du fichier ZIP sans "
        "extension).  Sera sanitizé automatiquement."
    ),
)
@click.option(
    "--metadata",
    "metadata_pairs",
    multiple=True,
    help=(
        "Paires ``clé=valeur`` (option répétable).  Ex : "
        "``--metadata language=fr --metadata period=medieval``."
    ),
)
@click.option(
    "--max-zip-mb",
    default=100,
    type=int,
    show_default=True,
    help="Plafond taille du blob ZIP (Mo).",
)
@click.option(
    "--max-entries",
    default=5000,
    type=int,
    show_default=True,
    help="Plafond nombre d'entrées dans le ZIP (anti zip bomb).",
)
@click.option(
    "--max-uncompressed-mb",
    default=500,
    type=int,
    show_default=True,
    help="Plafond taille décompressée totale (Mo).",
)
@click.option(
    "--quiet",
    is_flag=True,
    default=False,
    help="N'affiche que le chemin du dossier extrait, rien d'autre.",
)
def import_corpus_command(
    zip_path: Path,
    output_dir: Path,
    corpus_name: str | None,
    metadata_pairs: tuple[str, ...],
    max_zip_mb: int,
    max_entries: int,
    max_uncompressed_mb: int,
    quiet: bool,
) -> None:
    """Extrait un ZIP de corpus dans un workspace sandboxé."""
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace = WorkspaceManager(output_dir)

    if corpus_name is None:
        corpus_name = zip_path.stem

    metadata = _parse_metadata_pairs(metadata_pairs)

    service = CorpusService(
        workspace,
        max_zip_size_bytes=max_zip_mb * 1024 * 1024,
        max_entry_count=max_entries,
        max_uncompressed_bytes=max_uncompressed_mb * 1024 * 1024,
    )
    try:
        report = service.import_zip(
            zip_path.read_bytes(),
            corpus_name=corpus_name,
            metadata=metadata,
        )
    except CorpusImportError as exc:
        click.echo(f"erreur : {exc}", err=True)
        sys.exit(1)

    if quiet:
        click.echo(str(report.extracted_dir))
        return

    click.echo(f"Corpus extrait dans : {report.extracted_dir}")
    click.echo(f"  documents      : {report.n_documents}")
    click.echo(f"  sans GT        : {report.n_images_without_gt}")
    click.echo(f"  GT orphelines  : {report.n_gt_without_image}")
    click.echo(f"  bruit OS sauté : {report.n_skipped_noise}")
    if report.warnings:
        click.echo("Avertissements :")
        for w in report.warnings:
            click.echo(f"  - {w}")


def _parse_metadata_pairs(
    pairs: tuple[str, ...],
) -> dict[str, str]:
    """Parse ``("k1=v1", "k2=v2")`` → ``{"k1": "v1", "k2": "v2"}``.

    Lève ``click.BadParameter`` si une paire ne contient pas ``=``.
    """
    out: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise click.BadParameter(
                f"métadonnée invalide : {pair!r} (attendu ``clé=valeur``).",
                param_hint="--metadata",
            )
        key, _, value = pair.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            raise click.BadParameter(
                f"métadonnée à clé vide : {pair!r}.",
                param_hint="--metadata",
            )
        out[key] = value
    return out


__all__ = ["import_corpus_command"]
