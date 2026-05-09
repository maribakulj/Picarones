"""Rendu HTML « Recherchabilité fuzzy » — Sprint 86 (A.II.5a HTML).

Phase 5.C — module relocalisé depuis
``picarones.report.searchability_render`` vers
``picarones.reports.html.renderers.searchability``.  Le chemin
legacy reste disponible via un shim avec ``DeprecationWarning`` ;
suppression prévue en 2.0.

Suite directe ``picarones/core/searchability.py`` (Sprint 84) +
câblage runner (Sprint 86).

Pattern identique aux autres rendus (Sprints 41/43/62/67/72) :
**server-side**, pas de JavaScript, anti-injection systématique.

Vue
---
Tableau résumé : moteur × (rappel, n_searchable / n_gt_tokens,
docs).  Cellule rappel colorée par gradient rouge → vert.
Adaptative : ``""`` si aucun moteur n'a de
``aggregated_searchability``.
"""

from __future__ import annotations

from html import escape as _e
from typing import Optional

from picarones.reports._helpers.render_helpers import color_traffic_light


def build_searchability_summary_html(
    engines: list[dict],
    labels: Optional[dict[str, str]] = None,
) -> str:
    """Construit la table HTML de recherchabilité.

    Parameters
    ----------
    engines:
        Liste de dicts moteur ; chacun peut avoir
        ``aggregated_searchability``.
    labels:
        Dict i18n, clés ``search_*``.

    Returns
    -------
    str
        ``""`` si aucun moteur n'a de signal.
    """
    rows = [
        e for e in engines
        if isinstance(e.get("aggregated_searchability"), dict)
    ]
    if not rows:
        return ""
    labels = labels or {}
    title = labels.get("search_title", "Recherchabilité fuzzy")
    note = labels.get(
        "search_note",
        "Proportion de tokens GT retrouvés dans la sortie OCR à "
        "distance de Levenshtein ≤ 2 — proxy direct de la "
        "qualité pour la recherche plein-texte (Elastic, Solr).",
    )
    col_engine = labels.get("search_engine", "Moteur")
    col_recall = labels.get("search_recall", "Rappel")
    col_count = labels.get("search_count", "Tokens retrouvés / total")
    col_docs = labels.get("search_docs", "Docs")

    parts = [
        '<div class="searchability-section" style="margin:1rem 0">',
        f'<h3 style="margin:0 0 .3rem 0">{_e(title)}</h3>',
        f'<div style="font-size:.85rem;opacity:.75;margin-bottom:.5rem">'
        f'{_e(note)}</div>',
        '<table style="border-collapse:collapse;width:100%;'
        'font-size:.9rem">',
        '<thead><tr>',
    ]
    for col in (col_engine, col_recall, col_count, col_docs):
        parts.append(
            f'<th scope=\"col\" style="padding:.4rem .6rem;text-align:left;'
            f'border-bottom:1px solid #ccc;font-weight:600">'
            f'{_e(col)}</th>'
        )
    parts.append("</tr></thead><tbody>")
    for engine in rows:
        agg = engine["aggregated_searchability"]
        name = engine.get("name") or "?"
        recall = float(agg.get("recall") or 0.0)
        n_search = int(agg.get("n_searchable") or 0)
        n_total = int(agg.get("n_gt_tokens") or 0)
        n_docs = int(agg.get("n_docs") or 0)
        color = color_traffic_light(recall)
        parts.append(
            f'<tr>'
            f'<td style="padding:.4rem .6rem">{_e(str(name))}</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'background:{color};font-family:monospace;font-weight:600">'
            f'{recall * 100:.1f}%</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'font-family:monospace">{n_search} / {n_total}</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'font-family:monospace">{n_docs}</td>'
            f'</tr>'
        )
    parts.append("</tbody></table></div>")
    return "".join(parts)


__all__ = ["build_searchability_summary_html"]
