"""Commande ``pipeline`` : exécution + comparaison de pipelines composées (axe B).

Sous-module CLI extrait de l'ancien ``picarones/cli.py`` (1519 lignes)
lors du chantier 5 post-Sprint 97.  Les commandes ici s'enregistrent
automatiquement sur le groupe ``cli`` à l'import.

Comportement et signatures inchangés — uniquement de la modularisation.
"""

from __future__ import annotations

from pathlib import Path

import click

from picarones.cli import cli

# composées (axe B), pilotables depuis des fichiers YAML déclaratifs.
# ---------------------------------------------------------------------------


@cli.group("pipeline")
def pipeline_group() -> None:
    """Banc d'essai de pipelines composées (modules tiers)."""


@pipeline_group.command("run")
@click.argument(
    "spec_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--corpus",
    "corpus_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Répertoire du corpus à évaluer.",
)
@click.option(
    "--output-json",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Chemin de sortie JSON (résultats par document + agrégats).",
)
@click.option(
    "--output-html",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Chemin de sortie HTML (rapport autonome).",
)
@click.option(
    "--lang",
    type=click.Choice(["fr", "en"]),
    default="fr",
    show_default=True,
    help="Langue du rapport HTML.",
)
def pipeline_run_cmd(
    spec_path: Path,
    corpus_dir: Path,
    output_json: Path | None,
    output_html: Path | None,
    lang: str,
) -> None:
    """Exécute la pipeline décrite dans SPEC_PATH sur un corpus."""
    import json as _json

    from picarones.core.corpus import load_corpus_from_directory
    from picarones.measurements.pipeline_benchmark import run_pipeline_benchmark
    from picarones.measurements.pipeline_spec_loader import load_pipeline_spec_from_yaml

    spec = load_pipeline_spec_from_yaml(spec_path)
    corpus = load_corpus_from_directory(str(corpus_dir))
    click.echo(
        f"Pipeline {spec.name!r} sur {corpus.name!r} "
        f"({len(list(corpus.documents))} docs)"
    )
    bench = run_pipeline_benchmark(spec, corpus)
    click.echo(
        f"Terminé : {bench.n_pipelines_succeeded}/{bench.n_docs} succès "
        f"en {bench.total_duration_seconds:.2f}s"
    )
    for agg in bench.per_step_aggregates:
        click.echo(
            f"  {agg.step_name}: succès={agg.n_succeeded}/{agg.n_docs} "
            f"({agg.success_rate * 100:.0f}%)"
        )
    if output_json is not None:
        payload = {
            "pipeline_name": bench.pipeline_name,
            "corpus_name": bench.corpus_name,
            "n_docs": bench.n_docs,
            "n_pipelines_succeeded": bench.n_pipelines_succeeded,
            "n_pipelines_failed": bench.n_pipelines_failed,
            "total_duration_seconds": bench.total_duration_seconds,
            "per_step_aggregates": [
                {
                    "step_name": a.step_name,
                    "n_docs": a.n_docs,
                    "n_succeeded": a.n_succeeded,
                    "n_failed": a.n_failed,
                    "duration_seconds_mean": a.duration_seconds_mean,
                    "duration_seconds_median": a.duration_seconds_median,
                    "junction_metrics": a.junction_metrics,
                    "error_breakdown": a.error_breakdown,
                    "failing_doc_ids": a.failing_doc_ids,
                }
                for a in bench.per_step_aggregates
            ],
        }
        Path(output_json).write_text(
            _json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        click.echo(f"JSON exporté : {output_json}")
    if output_html is not None:
        from picarones.report.pipeline_render import build_pipeline_report_html
        Path(output_html).write_text(
            build_pipeline_report_html(bench, lang=lang),
            encoding="utf-8",
        )
        click.echo(f"HTML exporté : {output_html}")


@pipeline_group.command("compare")
@click.argument(
    "specs_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--corpus",
    "corpus_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Répertoire du corpus à évaluer.",
)
@click.option(
    "--output-html",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Chemin de sortie HTML (rapport comparatif autonome).",
)
@click.option(
    "--baseline",
    type=str,
    default=None,
    help="Nom de la pipeline baseline pour la table de gain.",
)
@click.option(
    "--lang",
    type=click.Choice(["fr", "en"]),
    default="fr",
    show_default=True,
    help="Langue du rapport HTML.",
)
def pipeline_compare_cmd(
    specs_path: Path,
    corpus_dir: Path,
    output_html: Path | None,
    baseline: str | None,
    lang: str,
) -> None:
    """Compare N pipelines décrites dans SPECS_PATH sur le même corpus."""
    from picarones.core.corpus import load_corpus_from_directory
    from picarones.core.modules import ArtifactType
    from picarones.measurements.pipeline_comparison import compare_pipelines
    from picarones.measurements.pipeline_spec_loader import (
        load_comparison_specs_from_yaml,
    )

    specs, extras = load_comparison_specs_from_yaml(specs_path)
    corpus = load_corpus_from_directory(str(corpus_dir))
    click.echo(
        f"Comparaison de {len(specs)} pipelines sur {corpus.name!r} "
        f"({len(list(corpus.documents))} docs)"
    )
    comparison = compare_pipelines(specs, corpus)
    click.echo(
        f"Terminé en {comparison.total_duration_seconds:.2f}s"
    )
    ranked = comparison.ranking_by_final_metric(
        ArtifactType.TEXT, "cer",
    )
    if ranked:
        click.echo("\nClassement par CER (TEXT) :")
        for i, (name, value) in enumerate(ranked, 1):
            shown = f"{value:.4f}" if value is not None else "N/A"
            click.echo(f"  {i}. {name}: {shown}")
    if output_html is not None:
        from picarones.report.pipeline_render import (
            RankingSpec,
            build_pipeline_comparison_report_html,
        )
        rankings_yaml = (
            extras.get("rankings") if isinstance(extras, dict) else None
        )
        ranking_specs: list[RankingSpec] = []
        if rankings_yaml and isinstance(rankings_yaml, list):
            for r in rankings_yaml:
                if not isinstance(r, dict):
                    continue
                try:
                    at = ArtifactType(r["artifact_type"])
                except (KeyError, ValueError):
                    continue
                ranking_specs.append(RankingSpec(
                    artifact_type=at,
                    metric_name=r.get("metric", "cer"),
                    higher_is_better=bool(r.get("higher_is_better", False)),
                    label=r.get("label"),
                ))
        if not ranking_specs:
            ranking_specs = [
                RankingSpec(ArtifactType.TEXT, "cer", label="CER"),
            ]
        Path(output_html).write_text(
            build_pipeline_comparison_report_html(
                comparison,
                ranking_specs=ranking_specs,
                baseline_pipeline=baseline,
                lang=lang,
            ),
            encoding="utf-8",
        )
        click.echo(f"\nHTML exporté : {output_html}")


if __name__ == "__main__":
    cli()
