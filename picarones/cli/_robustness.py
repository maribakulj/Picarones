"""Commande ``robustness`` : analyse de robustesse aux dégradations d'image.

Sous-module CLI extrait de l'ancien ``picarones/cli.py`` (1519 lignes)
lors du chantier 5 post-Sprint 97.  Les commandes ici s'enregistrent
automatiquement sur le groupe ``cli`` à l'import.

Comportement et signatures inchangés — uniquement de la modularisation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from picarones.cli import cli, _engine_from_name, _setup_logging

# ---------------------------------------------------------------------------
# picarones robustness
# ---------------------------------------------------------------------------

@cli.command("robustness")
@click.option(
    "--corpus", "-c",
    required=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Dossier contenant les paires image / .gt.txt",
)
@click.option(
    "--engine", "-e",
    default="tesseract",
    show_default=True,
    help="Moteur OCR à tester (tesseract, pero_ocr…)",
)
@click.option(
    "--degradations", "-d",
    default="noise,blur,rotation,resolution,binarization",
    show_default=True,
    help="Types de dégradation séparés par des virgules",
)
@click.option(
    "--cer-threshold",
    default=0.20,
    show_default=True,
    type=float,
    metavar="THRESHOLD",
    help="Seuil CER pour définir le niveau critique (0-1)",
)
@click.option(
    "--max-docs",
    default=10,
    show_default=True,
    type=click.IntRange(1, 1000),
    help="Nombre maximum de documents à traiter",
)
@click.option(
    "--output-json", "-o",
    default=None,
    type=click.Path(resolve_path=True),
    help="Exporte le rapport de robustesse en JSON",
)
@click.option(
    "--lang", "-l",
    default="fra",
    show_default=True,
    help="Code langue Tesseract",
)
@click.option("--no-progress", is_flag=True, default=False, help="Désactive la barre de progression")
@click.option("--demo", is_flag=True, default=False, help="Mode démo avec données fictives (sans OCR réel)")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Mode verbeux")
def robustness_cmd(
    corpus: str,
    engine: str,
    degradations: str,
    cer_threshold: float,
    max_docs: int,
    output_json: str | None,
    lang: str,
    no_progress: bool,
    demo: bool,
    verbose: bool,
) -> None:
    """Lance une analyse de robustesse d'un moteur OCR face aux dégradations d'image.

    Génère des versions dégradées des images (bruit, flou, rotation,
    réduction de résolution, binarisation) et mesure le CER à chaque niveau.

    \b
    Exemples :
        picarones robustness --corpus ./gt/ --engine tesseract
        picarones robustness --corpus ./gt/ --engine pero_ocr --degradations noise,blur
        picarones robustness --corpus ./gt/ --engine tesseract --output-json robustness.json
        picarones robustness --corpus ./gt/ --engine tesseract --demo
    """
    _setup_logging(verbose)

    import json as _json

    deg_types = [d.strip() for d in degradations.split(",") if d.strip()]

    from picarones.core.robustness import (
        RobustnessAnalyzer, ALL_DEGRADATION_TYPES, generate_demo_robustness_report
    )

    # Valider les types de dégradation
    invalid = [d for d in deg_types if d not in ALL_DEGRADATION_TYPES]
    if invalid:
        click.echo(
            f"Types de dégradation invalides : {', '.join(invalid)}\n"
            f"Types valides : {', '.join(ALL_DEGRADATION_TYPES)}",
            err=True,
        )
        sys.exit(1)

    click.echo(f"Corpus       : {corpus}")
    click.echo(f"Moteur       : {engine}")
    click.echo(f"Dégradations : {', '.join(deg_types)}")
    click.echo(f"Seuil CER    : {cer_threshold * 100:.0f}%")

    if demo:
        click.echo("\nMode démo : génération d'un rapport fictif réaliste…")
        report = generate_demo_robustness_report(engine_names=[engine])
    else:
        # Charger le corpus
        from picarones.core.corpus import load_corpus_from_directory
        try:
            corp = load_corpus_from_directory(corpus)
        except (FileNotFoundError, ValueError) as exc:
            click.echo(f"Erreur corpus : {exc}", err=True)
            sys.exit(1)

        click.echo(f"\n{len(corp)} documents chargés. Début de l'analyse…\n")

        # Instancier le moteur
        try:
            ocr_engine = _engine_from_name(engine, lang=lang, psm=6)
        except click.BadParameter as exc:
            click.echo(f"Erreur moteur : {exc}", err=True)
            sys.exit(1)

        from picarones.core.robustness import RobustnessAnalyzer
        analyzer = RobustnessAnalyzer(
            engines=[ocr_engine],
            degradation_types=deg_types,
            cer_threshold=cer_threshold,
        )
        report = analyzer.analyze(
            corpus=corp,
            show_progress=not no_progress,
            max_docs=max_docs,
        )

    # Affichage des résultats
    click.echo("\n── Résultats de robustesse ──────────────────────────")
    for curve in report.curves:
        click.echo(f"\n  {curve.engine_name} / {curve.degradation_type}")
        for label, cer in zip(curve.labels, curve.cer_values):
            if cer is not None:
                bar_len = int(cer * 40)
                bar = "█" * bar_len
                cer_pct = f"{cer * 100:.1f}%"
                threshold_marker = " ← CRITIQUE" if curve.critical_threshold_level is not None and \
                    curve.levels[curve.labels.index(label)] == curve.critical_threshold_level else ""
                click.echo(f"    {label:<12} {cer_pct:<8} {bar}{threshold_marker}")
        if curve.critical_threshold_level is not None:
            click.echo(
                click.style(
                    f"    Niveau critique (CER>{cer_threshold*100:.0f}%) : {curve.critical_threshold_level}",
                    fg="yellow",
                )
            )
        else:
            click.echo(click.style("    Robuste jusqu'au niveau max.", fg="green"))

    # Résumé
    click.echo("\n── Résumé ──────────────────────────────────────────")
    for key, val in report.summary.items():
        if key.startswith("most_robust_"):
            deg = key.replace("most_robust_", "")
            click.echo(f"  Moteur le plus robuste ({deg}) : {val}")

    # Export JSON
    if output_json:
        report_dict = report.as_dict()
        Path(output_json).write_text(
            _json.dumps(report_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        click.echo(f"\nRapport JSON exporté : {output_json}")


# ---------------------------------------------------------------------------
# Mise à jour de picarones demo pour illustrer suivi longitudinal + robustesse
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Sprint 70 — sous-groupe `pipeline` : runner et compare de pipelines
