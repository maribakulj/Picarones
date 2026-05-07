"""Rendu HTML du recall sur tokens rares (Sprint 71, A.I.1).

Petit tableau récapitulatif moteur × {n_rare_tokens, n_recalled,
recall, n_docs}. Adaptive : retourne ``""`` si aucune donnée.

Critique pour l'indexation prosopographique : un OCR qui rate
systématiquement les noms propres rares produit un corpus
inutilisable pour la recherche, même avec un CER global respectable.
"""

from __future__ import annotations

from html import escape as _e
from typing import Optional

from picarones.reports_v2._helpers.render_helpers import color_traffic_light


def build_rare_token_recall_html(
    per_engine: Optional[dict[str, dict]],
    labels: Optional[dict[str, str]] = None,
) -> str:
    """Construit le tableau récapitulatif du recall sur tokens rares.

    Parameters
    ----------
    per_engine:
        Sortie de
        :func:`picarones.report.report_data.extra_metrics.compute_rare_token_recall_per_engine`.
        Dict ``{engine_name: {n_rare_tokens, n_recalled, recall, n_docs, max_freq}}``.
        Si ``None`` ou vide, retourne ``""``.
    labels:
        Dict i18n optionnel.
    """
    if not per_engine:
        return ""
    labels = labels or {}
    title = labels.get(
        "rare_token_title", "Recall sur tokens rares (hapax + dis legomena)",
    )
    note = labels.get(
        "rare_token_note",
        "Pour chaque moteur, fraction des tokens rares (apparaissant ≤ 2 "
        "fois dans la GT du corpus) effectivement transcrits. Critique "
        "pour l'indexation prosopographique — un OCR qui rate les noms "
        "propres rares rend le corpus inutilisable pour la recherche.",
    )
    h_engine = labels.get("rare_token_engine", "Moteur")
    h_recall = labels.get("rare_token_recall", "Recall")
    h_recalled = labels.get("rare_token_recalled", "Tokens recalled")
    h_total = labels.get("rare_token_total", "Tokens rares (corpus)")
    h_docs = labels.get("rare_token_docs", "Docs évalués")

    rows = [
        (engine, info)
        for engine, info in per_engine.items()
        if isinstance(info, dict)
    ]
    if not rows:
        return ""

    parts = [
        '<section class="rare-token-section" style="margin:1rem 0">',
        f'<h3 style="margin:0 0 .3rem 0">{_e(title)}</h3>',
        f'<div style="font-size:.85rem;opacity:.75;margin-bottom:.5rem">'
        f'{_e(note)}</div>',
        '<table style="border-collapse:collapse;width:100%;'
        'font-size:.9rem">',
        '<thead><tr>',
    ]
    for h in (h_engine, h_recall, h_recalled, h_total, h_docs):
        parts.append(
            f'<th scope="col" style="padding:.4rem .6rem;text-align:left;'
            f'border-bottom:1px solid #ccc;font-weight:600">{_e(h)}</th>'
        )
    parts.append('</tr></thead><tbody>')

    # Tri par recall décroissant (les meilleurs en haut, None en queue).
    sorted_rows = sorted(
        rows,
        key=lambda kv: -(kv[1].get("recall") or -1.0),
    )
    for engine, info in sorted_rows:
        recall = info.get("recall")
        n_recalled = int(info.get("n_recalled") or 0)
        n_total = int(info.get("n_rare_tokens") or 0)
        n_docs = int(info.get("n_docs") or 0)
        if isinstance(recall, (int, float)):
            recall_color = color_traffic_light(float(recall))
            recall_cell = (
                f'<td style="padding:.4rem .6rem;text-align:right;'
                f'background:{recall_color};font-family:monospace;'
                f'font-weight:600">{recall * 100:.1f} %</td>'
            )
        else:
            recall_cell = (
                '<td style="padding:.4rem .6rem;text-align:right;'
                'opacity:.4">—</td>'
            )
        parts.append(
            f'<tr>'
            f'<td style="padding:.4rem .6rem">{_e(str(engine))}</td>'
            f'{recall_cell}'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'font-family:monospace">{n_recalled}</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'font-family:monospace">{n_total}</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'font-family:monospace">{n_docs}</td>'
            f'</tr>'
        )
    parts.append('</tbody></table></section>')
    return "".join(parts)


__all__ = ["build_rare_token_recall_html"]
