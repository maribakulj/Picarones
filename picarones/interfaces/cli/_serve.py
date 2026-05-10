"""Commande ``serve`` : lance l'interface web FastAPI Picarones.

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
# picarones serve
# ---------------------------------------------------------------------------

@cli.command("serve")
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Adresse d'écoute du serveur web",
)
@click.option(
    "--port", "-p",
    default=8000,
    show_default=True,
    type=click.IntRange(1, 65535),
    help="Port d'écoute du serveur web",
)
@click.option("--reload", is_flag=True, default=False, help="Mode rechargement automatique (développement)")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Mode verbeux")
def serve_cmd(host: str, port: int, reload: bool, verbose: bool) -> None:
    """Lance l'interface web locale Picarones sur localhost.

    Accessible dans le navigateur à l'adresse : http://HOST:PORT

    \b
    Exemples :
        picarones serve
        picarones serve --port 8080
        picarones serve --host 0.0.0.0 --port 8000
    """
    _setup_logging(verbose)

    try:
        import uvicorn
    except ImportError:
        click.echo(
            "uvicorn n'est pas installé. Installez-le avec :\n"
            "  pip install uvicorn[standard]\n"
            "ou :\n"
            "  pip install picarones[web]",
            err=True,
        )
        sys.exit(1)

    url = f"http://{host}:{port}"
    click.echo("Picarones — Interface web locale")
    click.echo(f"Démarrage du serveur sur {url}")
    click.echo("Appuyez sur Ctrl+C pour arrêter.\n")

    log_level = "debug" if verbose else "info"
    uvicorn.run(
        "picarones.interfaces.web.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )
