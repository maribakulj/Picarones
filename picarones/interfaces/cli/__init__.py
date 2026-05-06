"""CLI du rewrite ciblé — couche ``interfaces/cli``.

Point d'entrée Click ``cli`` qui regroupe les commandes consommant
les services applicatifs du rewrite (``CorpusService``,
``ReportService``, ``BenchmarkService``).

Usage
-----

::

    python -m picarones.interfaces.cli import-corpus mon_corpus.zip \\
        --output-dir ./workspaces/sess1
    python -m picarones.interfaces.cli report ./runs/run_001 \\
        --output rapport.html
    python -m picarones.interfaces.cli run --spec ./run.yaml

Distinct du legacy
------------------
``picarones.cli`` (legacy) reste opérationnel — il est appelé par le
script ``picarones`` installé via ``pyproject.toml``.  Cette nouvelle
CLI vit dans ``picarones.interfaces.cli`` et s'invoque via
``python -m``.  Quand le rewrite atteindra la parité fonctionnelle,
on basculera l'entry point ``console_scripts`` vers ce module et le
legacy sera supprimé.
"""

from __future__ import annotations

import click

from picarones.interfaces.cli.import_corpus import import_corpus_command
from picarones.interfaces.cli.report import report_command
from picarones.interfaces.cli.run import run_command


@click.group(
    name="picarones-rewrite",
    help=(
        "CLI du rewrite ciblé Picarones.  Sous-commandes : "
        "import-corpus, report, run."
    ),
)
@click.version_option(package_name="picarones")
def cli() -> None:
    """Groupe principal."""


cli.add_command(import_corpus_command, name="import-corpus")
cli.add_command(report_command, name="report")
cli.add_command(run_command, name="run")


__all__ = ["cli"]
