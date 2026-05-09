"""``picarones-rewrite report`` — génère le HTML d'un run persisté.

Wrapper Click mince autour du :class:`HtmlReportRenderer` (couche
``reports/html/``).

::

    python -m picarones.interfaces.cli report ./runs/run_001 \\
        --output rapport.html \\
        --lang fr

Comportement
------------
- Lit les 3 fichiers persistés par ``BenchmarkService.persist`` :
  ``run_manifest.json``, ``pipeline_results.jsonl``,
  ``view_results.jsonl``.
- Reconstruit le ``RunResult`` via
  :meth:`HtmlReportRenderer.load_run_result`.
- Rend le HTML autonome via :meth:`HtmlReportRenderer.render`.
- Écrit dans ``--output`` (chemin filesystem libre), ou affiche sur
  stdout si ``--output`` est omis.
- Code de sortie ``0`` succès, ``1`` fichiers persistés
  introuvables, ``2`` erreur d'usage Click.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from picarones.reports.html import HtmlReportRenderer


@click.command()
@click.argument(
    "run_dir",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, path_type=Path,
    ),
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Chemin du fichier HTML à écrire.  Si omis, le HTML est "
        "affiché sur stdout."
    ),
)
@click.option(
    "--lang",
    type=click.Choice(["fr", "en"]),
    default="fr",
    show_default=True,
    help="Langue des labels du rapport.",
)
def report_command(
    run_dir: Path,
    output_path: Path | None,
    lang: str,
) -> None:
    """Génère le rapport HTML d'un run persisté."""
    renderer = HtmlReportRenderer(lang=lang)
    try:
        html = renderer.render_from_dir(run_dir)
    except FileNotFoundError as exc:
        click.echo(f"erreur : {exc}", err=True)
        sys.exit(1)

    if output_path is None:
        click.echo(html)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    click.echo(f"Rapport HTML écrit dans : {output_path}")


__all__ = ["report_command"]
