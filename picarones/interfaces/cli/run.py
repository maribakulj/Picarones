"""``picarones-rewrite run`` — exécute un benchmark depuis un YAML.

Wrapper Click mince autour du :class:`RunOrchestrator` (couche
``app/services/``) — toute la logique métier vit dans le service,
ce module ne fait que du parsing CLI, l'injection du renderer HTML
(:class:`HtmlReportRenderer` de la couche ``reports/``) et le
formatage de sortie.

Usage
-----

::

    python -m picarones.interfaces.cli run --spec ./run.yaml

Codes de sortie : 0 succès, 1 erreur métier (typée
``PicaronesError``), 2 erreur Click (option mal formée).
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from picarones.app.results import RunResult
from picarones.app.schemas import RunSpecLoadError, load_run_spec_from_yaml
from picarones.app.services.corpus_service import CorpusImportError
from picarones.app.services.run_orchestrator import RunOrchestrator
from picarones.reports.html import HtmlReportRenderer


def _render_html_report(
    result: RunResult, output_path: Path, lang: str,
) -> Path:
    """Adapte :class:`HtmlReportRenderer` au protocole ``ReportRenderer``
    attendu par :meth:`RunOrchestrator.execute`."""
    renderer = HtmlReportRenderer(lang=lang)
    output_path.write_text(renderer.render(result), encoding="utf-8")
    return output_path


@click.command()
@click.option(
    "--spec",
    "spec_path",
    type=click.Path(
        exists=True, dir_okay=False, file_okay=True, path_type=Path,
    ),
    required=True,
    help="Chemin du fichier YAML décrivant le run.",
)
@click.option(
    "--no-report",
    is_flag=True,
    default=False,
    help=(
        "Ne génère pas le rapport HTML, même si ``report_html`` "
        "est défini dans la spec."
    ),
)
def run_command(spec_path: Path, no_report: bool) -> None:
    """Exécute un benchmark complet depuis une spec YAML."""
    # 1. Parsing de la spec.
    try:
        spec = load_run_spec_from_yaml(spec_path.read_text(encoding="utf-8"))
    except RunSpecLoadError as exc:
        click.echo(f"erreur : spec invalide : {exc}", err=True)
        sys.exit(1)

    # 2. Délégation au service d'orchestration avec injection du
    # renderer HTML (sauf si --no-report).
    orchestrator = RunOrchestrator(output_dir=Path(spec.output_dir))
    renderer = None if no_report else _render_html_report
    try:
        result = orchestrator.execute(spec, report_renderer=renderer)
    except CorpusImportError as exc:
        click.echo(f"erreur : import corpus : {exc}", err=True)
        sys.exit(1)
    except RunSpecLoadError as exc:
        click.echo(f"erreur : résolution pipeline : {exc}", err=True)
        sys.exit(1)

    # 3. Formatage de la sortie utilisateur.
    click.echo(
        f"Corpus chargé : {result.run_result.manifest.corpus_name} "
        f"({result.run_result.n_documents} docs, "
        f"{result.extracted_corpus_dir})",
    )
    click.echo(
        f"Lancement du run : "
        f"{len(result.run_result.manifest.pipeline_names)} pipeline(s) × "
        f"{len(result.run_result.manifest.view_specs)} vue(s) × "
        f"{result.run_result.n_documents} doc(s)…",
    )
    persist_dir = next(iter(result.persisted_files.values())).parent
    click.echo(f"Run persisté dans : {persist_dir}")
    for kind, path in result.persisted_files.items():
        click.echo(f"  {kind}: {path}")
    if result.report_path is not None:
        click.echo(f"Rapport : {result.report_path}")
    click.echo("OK")


__all__ = ["run_command"]
