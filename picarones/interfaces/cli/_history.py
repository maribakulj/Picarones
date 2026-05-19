"""Commande ``history`` : consultation de l'historique SQLite

Sous-module CLI extrait de l'ancien ``picarones/cli.py`` (1519 lignes)
lors du chantier 5 post-Sprint 97.  Les commandes ici s'enregistrent
automatiquement sur le groupe ``cli`` à l'import.

Comportement et signatures inchangés — uniquement de la modularisation.
"""

from __future__ import annotations


import click

from picarones.interfaces.cli import cli, _setup_logging

# ---------------------------------------------------------------------------
# picarones history
# ---------------------------------------------------------------------------

@cli.command("history")
@click.option(
    "--db",
    default="~/.picarones/history.db",
    show_default=True,
    type=click.Path(resolve_path=False),
    help="Chemin vers la base SQLite d'historique",
)
@click.option(
    "--engine", "-e",
    default=None,
    help="Filtre sur le nom du moteur",
)
@click.option(
    "--corpus", "-c",
    default=None,
    help="Filtre sur le nom du corpus",
)
@click.option(
    "--since",
    default=None,
    metavar="DATE",
    help="Date minimale ISO 8601 (ex: 2025-01-01)",
)
@click.option(
    "--limit", "-n",
    default=50,
    show_default=True,
    type=click.IntRange(1, 10000),
    help="Nombre maximum d'entrées à afficher",
)
@click.option(
    "--regression",
    is_flag=True,
    default=False,
    help="Détecter automatiquement les régressions (compare au run précédent)",
)
@click.option(
    "--regression-threshold",
    default=0.01,
    show_default=True,
    type=float,
    metavar="DELTA",
    help="Seuil de régression en points de CER absolus (ex: 0.01 = 1%)",
)
@click.option(
    "--export-json",
    default=None,
    type=click.Path(resolve_path=True),
    help="Exporte l'historique complet en JSON",
)
@click.option(
    "--demo",
    is_flag=True,
    default=False,
    help="Pré-remplir la base avec des données fictives de démonstration",
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Mode verbeux")
def history_cmd(
    db: str,
    engine: str | None,
    corpus: str | None,
    since: str | None,
    limit: int,
    regression: bool,
    regression_threshold: float,
    export_json: str | None,
    demo: bool,
    verbose: bool,
) -> None:
    """Consulte l'historique des benchmarks (suivi longitudinal).

    Affiche l'évolution du CER dans le temps pour chaque moteur et corpus.
    Permet de détecter automatiquement les régressions entre deux runs.

    \b
    Exemples :
        picarones history
        picarones history --engine tesseract --corpus "Chroniques médiévales"
        picarones history --regression --regression-threshold 0.02
        picarones history --demo   # données fictives de démonstration
        picarones history --export-json historique.json
    """
    _setup_logging(verbose)

    from picarones.evaluation.metrics.history import BenchmarkHistory, generate_demo_history

    history = BenchmarkHistory(db)

    if demo:
        click.echo("Insertion de données fictives de démonstration dans l'historique…")
        generate_demo_history(history, n_runs=8)
        click.echo(f"  {history.count()} entrées insérées.")

    if export_json:
        path = history.export_json(export_json)
        click.echo(f"Historique exporté : {path}")
        return

    entries = history.query(engine=engine, corpus=corpus, since=since, limit=limit)

    if not entries:
        click.echo("Aucun benchmark dans l'historique.")
        click.echo(
            "\nPour enregistrer automatiquement les runs, utilisez :\n"
            "  picarones run --corpus ./gt/ --engines tesseract --save-history\n"
            "\nOu pour tester avec des données fictives :\n"
            "  picarones history --demo"
        )
        return

    # Regrouper par moteur
    by_engine: dict[str, list] = {}
    for entry in entries:
        by_engine.setdefault(entry.engine_name, []).append(entry)

    click.echo(f"\n── Historique des benchmarks ({'filtré' if engine or corpus else 'tous'}) ──")
    click.echo(f"  Base : {history.db_path}")
    click.echo(f"  Total entrées : {len(entries)}\n")

    for eng_name, eng_entries in by_engine.items():
        click.echo(click.style(f"  Moteur : {eng_name}", bold=True))
        for e in eng_entries:
            cer_str = f"{e.cer_percent:.2f}%" if e.cer_percent is not None else "N/A"
            wer_str = f"{e.wer_mean * 100:.2f}%" if e.wer_mean is not None else "N/A"
            ts = e.timestamp[:10]  # date uniquement
            click.echo(f"    {ts}  CER={cer_str:<8} WER={wer_str:<8} docs={e.doc_count}  corpus={e.corpus_name}")
        click.echo()

    # Détection de régression
    if regression:
        click.echo("── Détection de régressions ──────────────────────")
        regressions = history.detect_all_regressions(threshold=regression_threshold)
        if not regressions:
            click.echo(
                click.style(
                    f"  Aucune régression détectée (seuil={regression_threshold*100:.1f}%)",
                    fg="green",
                )
            )
        else:
            for r in regressions:
                delta_str = f"+{r.delta_cer * 100:.2f}%" if r.delta_cer else "N/A"
                click.echo(
                    click.style(
                        f"  RÉGRESSION {r.engine_name} / {r.corpus_name} : "
                        f"delta CER={delta_str} "
                        f"({r.baseline_timestamp[:10]} → {r.current_timestamp[:10]})",
                        fg="red",
                    )
                )
