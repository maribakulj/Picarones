"""Rendu HTML « Déficit projeté de robustesse » — Sprint 88
(A.I.8 vue HTML).

Suite directe ``picarones/core/robustness_projection.py``
(Sprint 81).  Pattern identique aux autres rendus : server-
side, pas de JS, anti-injection systématique.

Note d'intégration
------------------
La robustesse synthétique (``picarones.evaluation.metrics.robustness``) est
exécutée par la CLI ``picarones robustness`` indépendamment du
benchmark principal.  Pour produire la vue de projection,
l'utilisateur compose :

.. code-block:: python

    from picarones.evaluation.metrics.robustness import analyze_robustness
    from picarones.evaluation.metrics.robustness_projection import (
        project_robustness_on_corpus,
        aggregate_projection_per_engine,
    )
    from picarones.reports.html.renderers.robustness_projection import (
        build_robustness_projection_html,
    )

    rob = analyze_robustness(corpus, [engine])         # Sprint 8
    projection = project_robustness_on_corpus(
        rob.curves,
        [doc.image_quality.as_dict() for doc in benchmark.docs],
    )                                                   # Sprint 81
    aggregated = aggregate_projection_per_engine(projection)
    html = build_robustness_projection_html(
        projection, aggregated, labels,
    )

Vue
---
1. **Tableau résumé par moteur** : déficit total attendu,
   nombre de types de dégradation, pire dégradation.
2. **Tableau détaillé par couple (moteur × dégradation)** :
   docs, docs avec data, déficit, % docs au-dessus du seuil
   critique.

Les cellules « déficit » sont colorées par gradient vert
(faible) → orange → rouge (≥ 5 points de CER projetés).

Adaptive : ``""`` si la projection est vide (aucune courbe ou
aucun document avec qualité).
"""

from __future__ import annotations

from html import escape as _e
from typing import Optional

from picarones.reports._helpers.render_helpers import color_traffic_light


def _build_summary_table(
    aggregated: dict,
    labels: dict[str, str],
) -> str:
    if not aggregated:
        return ""
    h_engine = labels.get("robproj_engine", "Moteur")
    h_total = labels.get("robproj_total", "Déficit total (pts CER)")
    h_n_types = labels.get("robproj_n_types", "Types évalués")
    h_worst = labels.get("robproj_worst", "Pire dégradation")
    parts = [
        '<table style="border-collapse:collapse;width:100%;'
        'font-size:.9rem;margin-bottom:.8rem">',
        '<thead><tr>',
    ]
    for col in (h_engine, h_total, h_n_types, h_worst):
        parts.append(
            f'<th scope=\"col\" style="padding:.4rem .6rem;text-align:left;'
            f'border-bottom:1px solid #ccc;font-weight:600">'
            f'{_e(col)}</th>'
        )
    parts.append("</tr></thead><tbody>")
    # Tri par déficit décroissant
    rows = sorted(
        aggregated.items(),
        key=lambda kv: -float(
            kv[1].get("total_expected_deficit") or 0.0
        ),
    )
    for engine, info in rows:
        deficit = float(info.get("total_expected_deficit") or 0.0)
        n_types = int(info.get("n_degradation_types") or 0)
        worst_type = info.get("worst_degradation_type")
        worst_deficit = info.get("worst_degradation_deficit")
        color = color_traffic_light(abs(deficit), low_is_good=True, scale_max=0.05)
        worst_str = (
            f"{_e(str(worst_type))} ({worst_deficit * 100:+.1f})"
            if worst_type and isinstance(worst_deficit, (int, float))
            else "—"
        )
        parts.append(
            f'<tr>'
            f'<td style="padding:.4rem .6rem">{_e(str(engine))}</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'background:{color};font-family:monospace;font-weight:600">'
            f'{deficit * 100:+.2f}</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'font-family:monospace">{n_types}</td>'
            f'<td style="padding:.4rem .6rem">{worst_str}</td>'
            f'</tr>'
        )
    parts.append("</tbody></table>")
    return "".join(parts)


def _build_detail_table(
    projection: dict,
    labels: dict[str, str],
) -> str:
    if not projection:
        return ""
    h_engine = labels.get("robproj_engine", "Moteur")
    h_deg_type = labels.get("robproj_deg_type", "Dégradation")
    h_n_docs = labels.get("robproj_n_docs", "Docs")
    h_n_with_data = labels.get("robproj_n_with_data", "Docs avec data")
    h_deficit = labels.get("robproj_deficit", "Δ CER projeté (pts)")
    h_above = labels.get("robproj_above", "Docs ≥ seuil critique")
    parts = [
        '<table style="border-collapse:collapse;width:100%;'
        'font-size:.9rem">',
        '<thead><tr>',
    ]
    for col in (h_engine, h_deg_type, h_n_docs,
                h_n_with_data, h_deficit, h_above):
        parts.append(
            f'<th scope=\"col\" style="padding:.4rem .6rem;text-align:left;'
            f'border-bottom:1px solid #ccc;font-weight:600">'
            f'{_e(col)}</th>'
        )
    parts.append("</tr></thead><tbody>")
    # Tri stable : par moteur puis type de dégradation
    for engine in sorted(projection):
        per_type = projection[engine] or {}
        for deg_type in sorted(per_type):
            entry = per_type[deg_type] or {}
            n_docs = int(entry.get("n_docs") or 0)
            n_with_data = int(entry.get("n_docs_with_data") or 0)
            deficit = entry.get("deficit_vs_baseline")
            n_above = int(entry.get("n_docs_above_critical") or 0)
            if isinstance(deficit, (int, float)):
                color = color_traffic_light(abs(float(deficit)), low_is_good=True, scale_max=0.05)
                deficit_str = f"{float(deficit) * 100:+.2f}"
                deficit_cell = (
                    f'<td style="padding:.4rem .6rem;text-align:right;'
                    f'background:{color};font-family:monospace">'
                    f'{deficit_str}</td>'
                )
            else:
                deficit_cell = (
                    '<td style="padding:.4rem .6rem;text-align:right;'
                    'opacity:.4">—</td>'
                )
            parts.append(
                f'<tr>'
                f'<td style="padding:.4rem .6rem">{_e(str(engine))}</td>'
                f'<td style="padding:.4rem .6rem">{_e(str(deg_type))}</td>'
                f'<td style="padding:.4rem .6rem;text-align:right;'
                f'font-family:monospace">{n_docs}</td>'
                f'<td style="padding:.4rem .6rem;text-align:right;'
                f'font-family:monospace">{n_with_data}</td>'
                f'{deficit_cell}'
                f'<td style="padding:.4rem .6rem;text-align:right;'
                f'font-family:monospace">{n_above}</td>'
                f'</tr>'
            )
    parts.append("</tbody></table>")
    return "".join(parts)


def build_robustness_projection_html(
    projection: Optional[dict],
    aggregated: Optional[dict] = None,
    labels: Optional[dict[str, str]] = None,
) -> str:
    """Construit la vue HTML « Déficit projeté de robustesse ».

    Parameters
    ----------
    projection:
        Sortie de ``project_robustness_on_corpus`` (Sprint 81),
        forme ``{engine: {deg_type: {...}}}``.  Si ``None`` ou
        vide, retourne ``""``.
    aggregated:
        Sortie de ``aggregate_projection_per_engine`` (Sprint
        81). Si ``None``, sera calculé à partir de
        ``projection``.
    labels:
        Dict i18n.  Clés sous le préfixe ``robproj_*``.

    Returns
    -------
    str
        Section HTML, ou ``""`` si projection vide.
    """
    if not projection:
        return ""
    if aggregated is None:
        from picarones.evaluation.metrics.robustness_projection import (
            aggregate_projection_per_engine,
        )
        aggregated = aggregate_projection_per_engine(projection)
    labels = labels or {}
    title = labels.get(
        "robproj_title",
        "Déficit projeté de robustesse sur le corpus réel",
    )
    note = labels.get(
        "robproj_note",
        "Projection des courbes de dégradation synthétique sur "
        "les caractéristiques d'image réelles. Le déficit total "
        "suppose l'indépendance des dégradations — c'est une "
        "approximation utile pour le diagnostic, pas un verdict.",
    )
    summary_table = _build_summary_table(aggregated or {}, labels)
    detail_table = _build_detail_table(projection, labels)
    if not summary_table and not detail_table:
        return ""
    h_summary = labels.get("robproj_summary", "Résumé par moteur")
    h_detail = labels.get(
        "robproj_detail", "Détail par couple (moteur × dégradation)",
    )
    parts = [
        '<section class="robproj-section" style="margin:1.5rem 0">',
        f'<h3 style="margin:0 0 .3rem 0">{_e(title)}</h3>',
        f'<div style="font-size:.85rem;opacity:.75;margin-bottom:.7rem">'
        f'{_e(note)}</div>',
    ]
    if summary_table:
        parts.append(
            f'<div style="font-weight:600;margin:.4rem 0 .3rem 0">'
            f'{_e(h_summary)}</div>'
        )
        parts.append(summary_table)
    if detail_table:
        parts.append(
            f'<div style="font-weight:600;margin:.6rem 0 .3rem 0">'
            f'{_e(h_detail)}</div>'
        )
        parts.append(detail_table)
    parts.append('</section>')
    return "".join(parts)


__all__ = ["build_robustness_projection_html"]
