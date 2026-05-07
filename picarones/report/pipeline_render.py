"""Rendu HTML server-side d'un benchmark de pipeline composée
(Sprint 67).

Suite directe Sprints 63-66 (axe B) — produit les blocs HTML qui
exposent le résultat d'une pipeline composée.

Pattern identique aux Sprints 41 (NER), 43 (calibration) et 62
(philologie) : rendu **server-side**, pas de JavaScript,
déterministe, anti-injection systématique via ``html.escape``.

Vue distincte du rapport OCR historique
---------------------------------------
Le rapport HTML OCR (``picarones/report/generator.py``) attend un
``BenchmarkResult`` (axe A).  Pour les pipelines composées, on
travaille avec ``PipelineBenchmarkResult`` (axe B, Sprint 64).

Ce module fournit donc un rapport **autonome** : la fonction
``build_pipeline_report_html`` produit un document HTML complet
(``<!doctype html>...``) que l'utilisateur peut écrire directement
sur disque, sans dépendre du générateur OCR.

Sprint 67 — périmètre
---------------------
Inclus :

- ``build_pipeline_summary_html(bench)`` — encart résumé global
  (corpus, n_docs, taux de succès, durée totale).
- ``build_pipeline_steps_table_html(bench)`` — tableau par étape
  (durée mean/median, n_succeeded/failed, error_breakdown,
  métriques aux jonctions).
- ``build_pipeline_report_html(bench, lang)`` — document HTML
  complet à sauver sur disque.

Reporté à Sprint 68 :

- Rendu d'un ``PipelineComparisonResult`` (ranking entre N
  pipelines + gain table).

Toujours pas de classification automatique
------------------------------------------
On affiche les chiffres bruts ; le chercheur lit et conclut.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape as _e
from typing import Optional

from picarones.domain.artifacts import ArtifactType
from picarones.measurements.pipeline_benchmark import PipelineBenchmarkResult
from picarones.measurements.pipeline_comparison import PipelineComparisonResult
from picarones.reports_v2._helpers.render_helpers import color_traffic_light


# ──────────────────────────────────────────────────────────────────────────
# Helpers communs
# ──────────────────────────────────────────────────────────────────────────


def _format_duration(seconds: float) -> str:
    """Formate une durée en ms si < 1s, en s sinon."""
    if seconds < 1.0:
        return f"{seconds * 1000:.1f} ms"
    if seconds < 60.0:
        return f"{seconds:.2f} s"
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    return f"{minutes}min {rest:.1f}s"


# ──────────────────────────────────────────────────────────────────────────
# Encart résumé corpus-wide
# ──────────────────────────────────────────────────────────────────────────


def build_pipeline_summary_html(
    bench: PipelineBenchmarkResult,
    labels: Optional[dict[str, str]] = None,
) -> str:
    """Construit l'encart résumé global du benchmark."""
    labels = labels or {}
    title = labels.get("pipeline_summary_title", "Résumé du benchmark")
    pipeline_label = labels.get("pipeline_name_label", "Pipeline")
    corpus_label = labels.get("pipeline_corpus_label", "Corpus")
    n_docs_label = labels.get("pipeline_n_docs_label", "Documents")
    succeeded_label = labels.get(
        "pipeline_succeeded_label", "Pipelines réussies",
    )
    failed_label = labels.get("pipeline_failed_label", "Pipelines échouées")
    duration_label = labels.get("pipeline_duration_label", "Durée totale")

    success = bench.n_pipelines_succeeded
    failed = bench.n_pipelines_failed
    total = bench.n_docs
    rate = success / total if total > 0 else 0.0
    color = color_traffic_light(rate)

    parts = [
        '<div class="pipeline-summary" '
        'style="margin:1rem 0;padding:.75rem;'
        'background:var(--bg-secondary,#f7f7f7);border-radius:6px">',
        f'<div style="font-weight:600;margin-bottom:.5rem">{_e(title)}</div>',
        '<table style="border-collapse:collapse;font-size:.9rem">',
    ]
    rows = [
        (pipeline_label, _e(bench.pipeline_name)),
        (corpus_label, _e(bench.corpus_name)),
        (n_docs_label, str(total)),
        (
            succeeded_label,
            f'<span style="background:{color};padding:.1rem .4rem;'
            f'border-radius:3px">{success} / {total}</span>',
        ),
        (failed_label, str(failed)),
        (duration_label, _e(_format_duration(bench.total_duration_seconds))),
    ]
    for label, value in rows:
        parts.append(
            f'<tr>'
            f'<td style="padding:.2rem .5rem;font-weight:500;'
            f'color:#555">{_e(label)}</td>'
            f'<td style="padding:.2rem .5rem">{value}</td>'
            f'</tr>'
        )
    parts.append("</table></div>")
    return "".join(parts)


# ──────────────────────────────────────────────────────────────────────────
# Tableau par étape
# ──────────────────────────────────────────────────────────────────────────


def build_pipeline_steps_table_html(
    bench: PipelineBenchmarkResult,
    labels: Optional[dict[str, str]] = None,
) -> str:
    """Construit le tableau par étape de la pipeline.

    Colonnes : nom de l'étape, n_succeeded, n_failed, taux de
    succès (cellule colorée), durée mean/median, métriques aux
    jonctions (mean) regroupées par type, error_breakdown
    catégorisé.
    """
    if not bench.per_step_aggregates:
        return ""
    labels = labels or {}
    title = labels.get("pipeline_steps_title", "Détail par étape")
    name_label = labels.get("pipeline_step_name_label", "Étape")
    succ_label = labels.get("pipeline_succeeded_label", "Réussies")
    fail_label = labels.get("pipeline_failed_label", "Échouées")
    rate_label = labels.get("pipeline_success_rate_label", "Taux succès")
    dmean_label = labels.get("pipeline_duration_mean_label", "Durée moyenne")
    dmedian_label = labels.get(
        "pipeline_duration_median_label", "Durée médiane",
    )
    metrics_label = labels.get(
        "pipeline_junction_metrics_label", "Métriques aux jonctions",
    )
    errors_label = labels.get("pipeline_error_breakdown_label", "Erreurs")

    parts = [
        '<div class="pipeline-steps" style="margin:1rem 0">',
        f'<div style="font-weight:600;margin-bottom:.4rem">{_e(title)}</div>',
        '<table style="border-collapse:collapse;font-size:.85rem;'
        'width:100%">',
        '<thead><tr>',
    ]
    for col in (
        name_label, succ_label, fail_label, rate_label,
        dmean_label, dmedian_label, metrics_label, errors_label,
    ):
        parts.append(
            f'<th scope=\"col\" style="padding:.3rem .5rem;text-align:left;'
            f'border-bottom:1px solid #ccc;font-weight:600">'
            f'{_e(col)}</th>'
        )
    parts.append("</tr></thead><tbody>")

    for agg in bench.per_step_aggregates:
        rate = agg.success_rate
        rate_color = color_traffic_light(rate)
        # Métriques aux jonctions : pour chaque type d'artefact,
        # liste des métriques mean
        metrics_cells: list[str] = []
        for at_value, type_metrics in sorted(agg.junction_metrics.items()):
            type_str = _e(at_value)
            for mname, stats in sorted(type_metrics.items()):
                mean = stats["mean"]
                n = stats["n"]
                metrics_cells.append(
                    f'<div style="font-size:.8rem;line-height:1.3">'
                    f'<code>{type_str}.{_e(mname)}</code>: '
                    f'{mean:.3f} '
                    f'<span style="opacity:.6">(n={n})</span></div>'
                )
        metrics_html = "".join(metrics_cells) or (
            '<span style="opacity:.5">—</span>'
        )
        # Error breakdown
        err_cells: list[str] = []
        for label, count in sorted(agg.error_breakdown.items()):
            err_cells.append(
                f'<div style="font-size:.8rem;line-height:1.3">'
                f'<code>{_e(label)}</code>: {count}</div>'
            )
        err_html = "".join(err_cells) or (
            '<span style="opacity:.5">—</span>'
        )

        parts.append(
            f'<tr>'
            f'<td style="padding:.3rem .5rem;font-weight:500">'
            f'{_e(agg.step_name)}</td>'
            f'<td style="padding:.3rem .5rem;text-align:right">'
            f'{agg.n_succeeded}</td>'
            f'<td style="padding:.3rem .5rem;text-align:right">'
            f'{agg.n_failed}</td>'
            f'<td style="padding:.3rem .5rem;text-align:center;'
            f'background:{rate_color}">{rate * 100:.0f}%</td>'
            f'<td style="padding:.3rem .5rem;text-align:right">'
            f'{_e(_format_duration(agg.duration_seconds_mean))}</td>'
            f'<td style="padding:.3rem .5rem;text-align:right">'
            f'{_e(_format_duration(agg.duration_seconds_median))}</td>'
            f'<td style="padding:.3rem .5rem">{metrics_html}</td>'
            f'<td style="padding:.3rem .5rem">{err_html}</td>'
            f'</tr>'
        )
    parts.append("</tbody></table></div>")
    return "".join(parts)


# ──────────────────────────────────────────────────────────────────────────
# Document HTML autonome
# ──────────────────────────────────────────────────────────────────────────


_DOC_STYLES = """
:root {
  --bg-primary: #ffffff;
  --bg-secondary: #f7f7f7;
  --text-primary: #222;
  --text-muted: #666;
  --border: #ddd;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: var(--bg-primary);
  color: var(--text-primary);
  line-height: 1.5;
}
header {
  padding: 1.5rem 2rem;
  border-bottom: 1px solid var(--border);
}
header h1 { margin: 0 0 .3rem 0; font-size: 1.4rem; }
header .subtitle { color: var(--text-muted); font-size: .9rem; }
main { padding: 1rem 2rem 3rem 2rem; max-width: 1400px; margin: 0 auto; }
table { border: 1px solid var(--border); }
code { background: #f0f0f0; padding: 0 .2rem; border-radius: 2px; font-size: .85em; }
.note {
  font-size: .85rem;
  color: var(--text-muted);
  font-style: italic;
  margin: .5rem 0 1.5rem 0;
}
"""


def build_pipeline_report_html(
    bench: PipelineBenchmarkResult,
    labels: Optional[dict[str, str]] = None,
    lang: str = "fr",
) -> str:
    """Construit un document HTML autonome pour un benchmark de
    pipeline composée.

    Le document est complet (``<!doctype html>...``) et peut être
    sauvé directement sur disque par l'utilisateur :

    >>> html = build_pipeline_report_html(bench)
    >>> Path("rapport_pipeline.html").write_text(html)
    """
    labels = labels or {}
    main_title = labels.get(
        "pipeline_report_title", "Rapport de pipeline composée",
    )
    note = labels.get(
        "pipeline_report_note",
        "Données brutes par étape. L'outil mesure et agrège — il "
        "ne classe pas la pipeline « bonne » ou « mauvaise ». "
        "C'est au chercheur de juger les chiffres selon ses critères.",
    )
    summary = build_pipeline_summary_html(bench, labels)
    steps = build_pipeline_steps_table_html(bench, labels)

    title_text = f"{main_title} — {bench.pipeline_name}"
    parts = [
        "<!doctype html>",
        f'<html lang="{_e(lang)}">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>{_e(title_text)}</title>",
        "<style>", _DOC_STYLES, "</style>",
        "</head>",
        "<body>",
        "<header>",
        f"<h1>{_e(title_text)}</h1>",
        f'<div class="subtitle">{_e(bench.corpus_name)} — '
        f'{bench.n_docs} {_e(labels.get("pipeline_docs_short", "docs"))}'
        f'</div>',
        "</header>",
        "<main>",
        f'<p class="note">{_e(note)}</p>',
        summary,
        steps,
        "</main>",
        "</body>",
        "</html>",
    ]
    return "".join(parts)


# ──────────────────────────────────────────────────────────────────────────
# Sprint 68 — comparaison de N pipelines : ranking + gain table
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class RankingSpec:
    """Spec d'un classement à afficher.

    Décrit la jonction (``artifact_type``) et la métrique
    (``metric_name``) à utiliser pour classer les pipelines.

    Attributs
    ---------
    artifact_type:
        Type d'artefact où la métrique est calculée (typiquement
        ``ArtifactType.TEXT`` pour des métriques OCR).
    metric_name:
        Nom de la métrique dans le registre typé Sprint 34
        (``"cer"``, ``"wer"``, ``"flesch_delta_fr"``, etc.).
    higher_is_better:
        ``False`` (défaut) pour les métriques d'erreur (CER, WER) ;
        ``True`` pour les métriques de qualité (accuracy, F1,
        coverage…).
    label:
        Libellé optionnel à afficher dans le tableau ; sinon
        construit comme ``"<artifact_type>.<metric_name>"``.
    """

    artifact_type: ArtifactType
    metric_name: str
    higher_is_better: bool = False
    label: Optional[str] = None

    @property
    def display_label(self) -> str:
        if self.label:
            return self.label
        return f"{self.artifact_type.value}.{self.metric_name}"


def _bg_for_rank(rank: int, total: int) -> str:
    """Gradient vert (rang 1) → rouge (dernier rang).

    Mapping : ``rank ∈ [1, total]`` → ``color_traffic_light`` avec
    ``low_is_good=True`` (rang bas = bon).
    """
    if total <= 1:
        return color_traffic_light(1.0)
    return color_traffic_light(
        float(rank), low_is_good=True, scale_min=1.0, scale_max=float(total),
    )


def build_pipeline_ranking_table_html(
    comparison: PipelineComparisonResult,
    ranking_spec: RankingSpec,
    labels: Optional[dict[str, str]] = None,
) -> str:
    """Tableau de classement des pipelines selon une métrique finale.

    Colonnes : rang, nom du pipeline, valeur de la métrique (mean
    sur le corpus à la dernière jonction qui produit
    ``artifact_type``).  Les pipelines sans valeur sont listés en
    queue avec un tiret.
    """
    labels = labels or {}
    title_template = labels.get(
        "pipeline_ranking_title", "Classement par {label}",
    )
    title = title_template.format(label=ranking_spec.display_label)
    rank_label = labels.get("pipeline_rank_label", "Rang")
    name_label = labels.get("pipeline_name_label", "Pipeline")
    value_label = labels.get("pipeline_value_label", "Valeur")

    ranked = comparison.ranking_by_final_metric(
        ranking_spec.artifact_type,
        ranking_spec.metric_name,
        higher_is_better=ranking_spec.higher_is_better,
    )
    if not ranked:
        return ""

    n_with_value = sum(1 for _name, v in ranked if v is not None)

    parts = [
        '<div class="pipeline-ranking" style="margin:1rem 0">',
        f'<div style="font-weight:600;margin-bottom:.4rem">{_e(title)}</div>',
        '<table style="border-collapse:collapse;font-size:.85rem">',
        '<thead><tr>',
    ]
    for col in (rank_label, name_label, value_label):
        parts.append(
            f'<th scope=\"col\" style="padding:.3rem .5rem;text-align:left;'
            f'border-bottom:1px solid #ccc;font-weight:600">'
            f'{_e(col)}</th>'
        )
    parts.append("</tr></thead><tbody>")

    rank = 0
    for name, value in ranked:
        if value is None:
            rank_str = "—"
            value_str = "—"
            rank_color = "#f0f0f0"
        else:
            rank += 1
            rank_str = str(rank)
            value_str = f"{value:.4f}"
            rank_color = _bg_for_rank(rank, n_with_value)
        parts.append(
            f'<tr>'
            f'<td style="padding:.3rem .5rem;text-align:center;'
            f'background:{rank_color};font-weight:600">{rank_str}</td>'
            f'<td style="padding:.3rem .5rem">{_e(name)}</td>'
            f'<td style="padding:.3rem .5rem;text-align:right;'
            f'font-family:monospace">{value_str}</td>'
            f'</tr>'
        )
    parts.append("</tbody></table></div>")
    return "".join(parts)


def build_pipeline_gain_table_html(
    comparison: PipelineComparisonResult,
    ranking_spec: RankingSpec,
    baseline_pipeline: str,
    labels: Optional[dict[str, str]] = None,
) -> str:
    """Tableau gain vs baseline pour une métrique donnée.

    Colonnes : pipeline, valeur, gain absolu, gain relatif.  La
    baseline est marquée explicitement (cellule grisée).
    Convention de couleur : vert si gain favorable selon
    ``higher_is_better``, rouge sinon.
    """
    labels = labels or {}
    title_template = labels.get(
        "pipeline_gain_title", "Gain vs {baseline} sur {label}",
    )
    title = title_template.format(
        baseline=baseline_pipeline,
        label=ranking_spec.display_label,
    )
    name_label = labels.get("pipeline_name_label", "Pipeline")
    value_label = labels.get("pipeline_value_label", "Valeur")
    abs_label = labels.get("pipeline_gain_absolute_label", "Gain absolu")
    rel_label = labels.get("pipeline_gain_relative_label", "Gain relatif")
    baseline_label = labels.get(
        "pipeline_baseline_marker", "(référence)",
    )

    try:
        gains = comparison.gain_table(
            ranking_spec.artifact_type,
            ranking_spec.metric_name,
            baseline_pipeline,
        )
    except KeyError:
        return ""

    parts = [
        '<div class="pipeline-gain" style="margin:1rem 0">',
        f'<div style="font-weight:600;margin-bottom:.4rem">{_e(title)}</div>',
        '<table style="border-collapse:collapse;font-size:.85rem">',
        '<thead><tr>',
    ]
    for col in (name_label, value_label, abs_label, rel_label):
        parts.append(
            f'<th scope=\"col\" style="padding:.3rem .5rem;text-align:left;'
            f'border-bottom:1px solid #ccc;font-weight:600">'
            f'{_e(col)}</th>'
        )
    parts.append("</tr></thead><tbody>")

    for name, g in gains.items():
        is_baseline = name == baseline_pipeline
        value = g["value"]
        absolute = g["absolute"]
        relative = g["relative"]
        # Formatage des cellules
        value_str = "—" if value is None else f"{value:.4f}"
        abs_str = "—" if absolute is None else f"{absolute:+.4f}"
        rel_str = "—" if relative is None else f"{relative * 100:+.1f}%"
        # Couleur du gain : vert si favorable, rouge sinon, gris pour
        # la baseline.
        if is_baseline:
            gain_color = "#f0f0f0"
        elif absolute is None or absolute == 0:
            gain_color = "#f0f0f0"
        else:
            favorable = (
                absolute > 0 if ranking_spec.higher_is_better else absolute < 0
            )
            gain_color = "#cfe8cf" if favorable else "#f4cfcf"
        # Marqueur baseline
        name_cell = _e(name)
        if is_baseline:
            name_cell += (
                f' <span style="opacity:.6;font-size:.85em">'
                f'{_e(baseline_label)}</span>'
            )
        parts.append(
            f'<tr>'
            f'<td style="padding:.3rem .5rem;font-weight:500">{name_cell}</td>'
            f'<td style="padding:.3rem .5rem;text-align:right;'
            f'font-family:monospace">{value_str}</td>'
            f'<td style="padding:.3rem .5rem;text-align:right;'
            f'font-family:monospace;background:{gain_color}">{abs_str}</td>'
            f'<td style="padding:.3rem .5rem;text-align:right;'
            f'font-family:monospace;background:{gain_color}">{rel_str}</td>'
            f'</tr>'
        )
    parts.append("</tbody></table></div>")
    return "".join(parts)


def build_pipeline_comparison_summary_html(
    comparison: PipelineComparisonResult,
    labels: Optional[dict[str, str]] = None,
) -> str:
    """Encart de résumé global d'une comparaison de pipelines.

    Affiche corpus, n_docs, durée totale, nombre de pipelines, et
    pour chacune un mini-résumé n_succeeded / n_docs.
    """
    labels = labels or {}
    title = labels.get(
        "pipeline_comparison_summary_title", "Résumé de la comparaison",
    )
    corpus_label = labels.get("pipeline_corpus_label", "Corpus")
    n_docs_label = labels.get("pipeline_n_docs_label", "Documents")
    n_pipelines_label = labels.get(
        "pipeline_n_pipelines_label", "Pipelines comparées",
    )
    duration_label = labels.get("pipeline_duration_label", "Durée totale")

    parts = [
        '<div class="pipeline-comparison-summary" '
        'style="margin:1rem 0;padding:.75rem;'
        'background:var(--bg-secondary,#f7f7f7);border-radius:6px">',
        f'<div style="font-weight:600;margin-bottom:.5rem">{_e(title)}</div>',
        '<table style="border-collapse:collapse;font-size:.9rem">',
    ]
    rows = [
        (corpus_label, _e(comparison.corpus_name)),
        (n_docs_label, str(comparison.n_docs)),
        (n_pipelines_label, str(len(comparison.per_pipeline))),
        (duration_label, _e(_format_duration(comparison.total_duration_seconds))),
    ]
    for label, value in rows:
        parts.append(
            f'<tr>'
            f'<td style="padding:.2rem .5rem;font-weight:500;color:#555">'
            f'{_e(label)}</td>'
            f'<td style="padding:.2rem .5rem">{value}</td>'
            f'</tr>'
        )
    parts.append("</table>")
    # Mini-résumé par pipeline
    if comparison.per_pipeline:
        per_pipeline_label = labels.get(
            "pipeline_per_pipeline_label", "Par pipeline",
        )
        parts.append(
            f'<div style="margin-top:.6rem;font-size:.85rem">'
            f'<span style="font-weight:500;color:#555">'
            f'{_e(per_pipeline_label)} :</span>'
        )
        items: list[str] = []
        for name, bench in comparison.per_pipeline.items():
            items.append(
                f'<code>{_e(name)}</code> '
                f'({bench.n_pipelines_succeeded}/{bench.n_docs})'
            )
        parts.append(" — ".join(items))
        parts.append("</div>")
    parts.append("</div>")
    return "".join(parts)


def build_pipeline_comparison_report_html(
    comparison: PipelineComparisonResult,
    ranking_specs: Optional[list[RankingSpec]] = None,
    baseline_pipeline: Optional[str] = None,
    labels: Optional[dict[str, str]] = None,
    lang: str = "fr",
) -> str:
    """Document HTML autonome pour une comparaison de N pipelines.

    Parameters
    ----------
    comparison:
        Résultat de ``compare_pipelines`` (Sprint 65).
    ranking_specs:
        Liste explicite des classements à afficher.  Pour chaque
        spec, on rend un tableau de classement et, si
        ``baseline_pipeline`` est fourni, un tableau de gain.
        Si ``None`` ou vide, on affiche uniquement le résumé
        global et les résumés par pipeline (sans verdict).
    baseline_pipeline:
        Pipeline de référence pour les tableaux de gain.  Si
        ``None``, les tableaux de gain ne sont pas affichés.
    labels:
        Map i18n.
    lang:
        Code langue pour ``<html lang="…">``.

    Returns
    -------
    str
        Document HTML complet (``<!doctype html>`` + ``<html>``).
    """
    labels = labels or {}
    main_title = labels.get(
        "pipeline_comparison_report_title",
        "Rapport de comparaison de pipelines",
    )
    note = labels.get(
        "pipeline_comparison_report_note",
        "Données comparatives brutes. L'outil mesure et classe — il "
        "ne tranche pas le débat éditorial. C'est au chercheur de "
        "lire les chiffres et de conclure selon ses critères.",
    )
    title_text = f"{main_title} — {comparison.corpus_name}"
    summary = build_pipeline_comparison_summary_html(comparison, labels)

    rankings_html: list[str] = []
    for spec in (ranking_specs or []):
        rankings_html.append(
            build_pipeline_ranking_table_html(comparison, spec, labels),
        )
        if baseline_pipeline is not None:
            rankings_html.append(
                build_pipeline_gain_table_html(
                    comparison, spec, baseline_pipeline, labels,
                ),
            )

    parts = [
        "<!doctype html>",
        f'<html lang="{_e(lang)}">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>{_e(title_text)}</title>",
        "<style>", _DOC_STYLES, "</style>",
        "</head>",
        "<body>",
        "<header>",
        f"<h1>{_e(title_text)}</h1>",
        f'<div class="subtitle">{len(comparison.per_pipeline)} '
        f'{_e(labels.get("pipeline_n_pipelines_short", "pipelines"))} '
        f'— {comparison.n_docs} '
        f'{_e(labels.get("pipeline_docs_short", "docs"))}'
        f'</div>',
        "</header>",
        "<main>",
        f'<p class="note">{_e(note)}</p>',
        summary,
    ]
    parts.extend(rankings_html)
    parts.extend([
        "</main>",
        "</body>",
        "</html>",
    ])
    return "".join(parts)


__all__ = [
    "build_pipeline_summary_html",
    "build_pipeline_steps_table_html",
    "build_pipeline_report_html",
    "RankingSpec",
    "build_pipeline_ranking_table_html",
    "build_pipeline_gain_table_html",
    "build_pipeline_comparison_summary_html",
    "build_pipeline_comparison_report_html",
]
