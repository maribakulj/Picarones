"""CLI du nouveau monde — Sprint A14-S22.

Point d'entrée Click ``cli`` qui regroupe les commandes consommant
les services applicatifs du rewrite (S20 ``CorpusService``, S21
``ReportService``).

Usage en S22
------------

::

    python -m picarones.app.cli import-corpus mon_corpus.zip \\
        --output-dir ./workspaces/sess1
    python -m picarones.app.cli report ./runs/run_001 \\
        --output rapport.html

Distinct du legacy
------------------
``picarones.cli`` (legacy) reste opérationnel — il est appelé par le
script ``picarones`` installé via ``pyproject.toml``.  Cette nouvelle
CLI vit dans ``picarones.app.cli`` et n'est pas (encore) exposée
comme commande shell ; elle s'invoque via ``python -m``.

Quand le rewrite atteindra la parité fonctionnelle avec le legacy,
on basculera l'entry point ``console_scripts`` vers ce module et le
legacy sera supprimé.

Pourquoi pas ``picarones run``
------------------------------
La commande ``run`` exige un registre d'adapters OCR/LLM + un
chargeur de spec YAML — elle dépend d'un ``RegistryService`` non
encore livré (S23+).  S22 livre les deux commandes qui n'ont besoin
d'aucun registre : ``import-corpus`` (S20) et ``report`` (S21).
"""

from __future__ import annotations

import click

from picarones.app.cli.import_corpus import import_corpus_command
from picarones.app.cli.report import report_command
from picarones.app.cli.run import run_command


@click.group(
    name="picarones-rewrite",
    help=(
        "CLI du rewrite ciblé Picarones (S22-S24).  Sous-commandes : "
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
