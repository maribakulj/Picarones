"""Rendu HTML « Throughput effectif » — Sprint 91 (A.II.6).

Suite directe ``picarones/core/throughput.py``.  Pattern
identique aux autres rendus : server-side, pas de JS, anti-
injection systématique.

Vue
---
Tableau résumé moteur × {pages/h brut, pages/h **utilisable**,
% temps de correction (drag), n_pages, n_errors}.  La cellule
**pages/h utilisable** est colorée en gradient rouge (faible)
→ vert (élevé), normalisé sur le maximum observé.

Adaptive : ``""`` si ``aggregate_effective_throughput`` retourne
``None`` (aucun moteur exploitable).

Note d'intégration
------------------
Cette vue est un **module pur** — l'utilisateur compose :

.. code-block:: python

    from picarones.measurements.throughput import (
        aggregate_effective_throughput,
    )
    from picarones.report.throughput_render import (
        build_throughput_html,
    )

    per_engine = []
    for report in benchmark.engine_reports:
        n_errors = sum(
            int(round(dr.metrics.wer * dr.metrics.reference_length / 5))
            for dr in report.document_results
        )
        per_engine.append({
            "engine_name": report.engine_name,
            "n_pages": len(report.document_results),
            "duration_seconds": sum(
                dr.duration_seconds for dr in report.document_results
            ),
            "n_errors": n_errors,
        })
    agg = aggregate_effective_throughput(per_engine)
    html = build_throughput_html(agg, labels)
"""

from __future__ import annotations

from html import escape as _e
from typing import Optional

from picarones.report.render_helpers import color_traffic_light


def build_throughput_html(
    aggregated: Optional[dict],
    labels: Optional[dict[str, str]] = None,
) -> str:
    """Construit la vue HTML throughput effectif.

    Parameters
    ----------
    aggregated:
        Sortie de ``aggregate_effective_throughput``.  Si
        ``None`` ou liste vide, retourne ``""``.
    labels:
        Dict i18n.  Clés sous le préfixe ``throughput_*``.
    """
    if not aggregated:
        return ""
    rows = aggregated.get("engines") or []
    if not rows:
        return ""
    labels = labels or {}
    title = labels.get("throughput_title", "Throughput effectif")
    note = labels.get(
        "throughput_note",
        "Pages traitables par heure en intégrant le temps de "
        "correction humaine post-OCR. Discrimine entre un cloud "
        "rapide mais imprécis et un local lent mais fiable. "
        "Constante de correction : {time_per_error}s par erreur "
        "(défaut HTR-United, surchargeable).",
    )
    time_per_error = aggregated.get("time_per_error_seconds", 5.0)
    note = note.replace("{time_per_error}", f"{time_per_error:.0f}")
    h_engine = labels.get("throughput_engine", "Moteur")
    h_raw = labels.get("throughput_raw", "Pages/h brut")
    h_effective = labels.get(
        "throughput_effective", "Pages/h utilisable",
    )
    h_drag = labels.get("throughput_drag", "% correction")
    h_pages = labels.get("throughput_pages", "Pages")
    h_errors = labels.get("throughput_errors", "Erreurs")
    max_eff = max(
        (r.get("pages_per_hour_effective") or 0.0) for r in rows
    )

    parts = [
        '<section class="throughput-section" style="margin:1rem 0">',
        f'<h3 style="margin:0 0 .3rem 0">{_e(title)}</h3>',
        f'<div style="font-size:.85rem;opacity:.75;margin-bottom:.6rem">'
        f'{_e(note)}</div>',
        '<table style="border-collapse:collapse;width:100%;'
        'font-size:.9rem">',
        '<thead><tr>',
    ]
    for col in (h_engine, h_raw, h_effective, h_drag, h_pages, h_errors):
        parts.append(
            f'<th scope=\"col\" style="padding:.4rem .6rem;text-align:left;'
            f'border-bottom:1px solid #ccc;font-weight:600">'
            f'{_e(col)}</th>'
        )
    parts.append("</tr></thead><tbody>")
    for row in sorted(
        rows,
        key=lambda r: -(r.get("pages_per_hour_effective") or 0.0),
    ):
        engine = str(row.get("engine_name") or "?")
        raw = row.get("pages_per_hour_raw")
        eff = row.get("pages_per_hour_effective") or 0.0
        drag = row.get("drag_ratio") or 0.0
        n_pages = int(row.get("n_pages") or 0)
        n_errors = int(row.get("n_errors") or 0)
        eff_color = (
            color_traffic_light(eff, scale_max=max_eff)
            if max_eff > 0 else "#e0e0e0"
        )
        drag_color = color_traffic_light(drag, low_is_good=True)
        raw_str = (
            f"{raw:,.0f}" if isinstance(raw, (int, float)) else "—"
        )
        parts.append(
            f'<tr>'
            f'<td style="padding:.4rem .6rem">{_e(engine)}</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'font-family:monospace">{raw_str}</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'background:{eff_color};font-family:monospace;'
            f'font-weight:600">{eff:,.0f}</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'background:{drag_color};font-family:monospace">'
            f'{drag * 100:.1f}%</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'font-family:monospace">{n_pages}</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'font-family:monospace">{n_errors}</td>'
            f'</tr>'
        )
    parts.append("</tbody></table></section>")
    return "".join(parts)


__all__ = ["build_throughput_html"]
