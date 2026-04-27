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

from html import escape as _e
from typing import Optional

from picarones.core.pipeline_benchmark import PipelineBenchmarkResult


# ──────────────────────────────────────────────────────────────────────────
# Helpers communs
# ──────────────────────────────────────────────────────────────────────────


def _color_for_success_rate(rate: float) -> str:
    """Gradient rouge → jaune → vert pour le taux de succès."""
    f = max(0.0, min(1.0, rate))
    if f <= 0.5:
        ratio = f / 0.5
        r = int(220 + (240 - 220) * ratio)
        g = int(100 + (220 - 100) * ratio)
        b = int(100 + (130 - 100) * ratio)
    else:
        ratio = (f - 0.5) / 0.5
        r = int(240 + (130 - 240) * ratio)
        g = int(220 + (200 - 220) * ratio)
        b = int(130 + (130 - 130) * ratio)
    return f"#{r:02x}{g:02x}{b:02x}"


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
    color = _color_for_success_rate(rate)

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
            f'<th style="padding:.3rem .5rem;text-align:left;'
            f'border-bottom:1px solid #ccc;font-weight:600">'
            f'{_e(col)}</th>'
        )
    parts.append("</tr></thead><tbody>")

    for agg in bench.per_step_aggregates:
        rate = agg.success_rate
        rate_color = _color_for_success_rate(rate)
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


__all__ = [
    "build_pipeline_summary_html",
    "build_pipeline_steps_table_html",
    "build_pipeline_report_html",
]
