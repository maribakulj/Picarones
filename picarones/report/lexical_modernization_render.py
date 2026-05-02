"""Rendu HTML de la vue « Modernisation lexicale » — Sprint 80.

A.I.7 du plan d'évolution 2026.

Suite directe ``picarones/core/lexical_modernization.py``.
Pattern identique aux autres rendus (Sprints 41/43/62/67/72/74/75/76/77) :
**server-side**, pas de JavaScript, anti-injection systématique.

Vue
---
Tableau trié par taux de modernisation décroissant : forme
historique GT → forme(s) modernisée(s), occurrences GT, %.
Couleur de cellule pour le %.
"""

from __future__ import annotations

from html import escape as _e
from typing import Optional

from picarones.measurements.lexical_modernization import top_modernized_tokens
from picarones.report.render_helpers import (
    GRADIENT_TARGET_ORANGE,
    color_single_gradient,
)


def _format_variants(variants: dict, max_show: int = 3) -> str:
    """Liste compacte des variants modernisés."""
    items = sorted(variants.items(), key=lambda kv: -kv[1])
    shown = items[:max_show]
    rest = len(items) - max_show
    parts = [
        f"{_e(form)} ({count})"
        for form, count in shown
    ]
    if rest > 0:
        parts.append(f"+{rest}")
    return ", ".join(parts)


def build_lexical_modernization_html(
    data: Optional[dict],
    labels: Optional[dict[str, str]] = None,
    *,
    top_n: int = 20,
    min_total: int = 1,
) -> str:
    """Construit la table HTML de modernisation lexicale.

    Retourne ``""`` si ``data is None`` ou si aucun token modernisé.
    """
    if not data:
        return ""
    rows = top_modernized_tokens(data, n=top_n, min_total=min_total)
    if not rows:
        return ""
    labels = labels or {}
    title = labels.get(
        "lexmod_title", "Modernisation lexicale (top tokens)",
    )
    note = labels.get(
        "lexmod_note",
        "Tokens GT que le moteur réécrit le plus souvent. "
        "Lecture : « maistre → maître modernisé dans 85 % des cas » "
        "indique de quoi corriger dans le prompt pour préserver "
        "l'orthographe historique.",
    )
    gt_label = labels.get("lexmod_gt_label", "Forme historique GT")
    hyp_label = labels.get("lexmod_hyp_label", "Variantes OCR")
    n_label = labels.get("lexmod_n_label", "n GT")
    rate_label = labels.get("lexmod_rate_label", "% modernisé")

    parts = [
        '<div class="lexmod" style="margin:1rem 0">',
        f'<div style="font-weight:600;margin-bottom:.4rem">{_e(title)}</div>',
        f'<div style="font-size:.85rem;opacity:.75;margin-bottom:.5rem">'
        f'{_e(note)}</div>',
        '<table style="border-collapse:collapse;width:100%;'
        'font-size:.85rem">',
        '<thead><tr>',
    ]
    for col in (gt_label, hyp_label, n_label, rate_label):
        parts.append(
            f'<th scope=\"col\" style="padding:.3rem .5rem;text-align:left;'
            f'border-bottom:1px solid #ccc;font-weight:600">'
            f'{_e(col)}</th>'
        )
    parts.append("</tr></thead><tbody>")
    for gt_token, slot in rows:
        rate = slot.get("rate_modernized", 0.0)
        n_total = slot.get("n_total", 0)
        variants_str = _format_variants(slot.get("variants") or {})
        rate_color = color_single_gradient(rate, end_rgb=GRADIENT_TARGET_ORANGE)
        parts.append(
            f'<tr>'
            f'<td style="padding:.3rem .5rem;font-family:monospace">'
            f'{_e(gt_token)}</td>'
            f'<td style="padding:.3rem .5rem;font-size:.85rem">'
            f'{variants_str}</td>'
            f'<td style="padding:.3rem .5rem;text-align:right;'
            f'font-family:monospace">{n_total}</td>'
            f'<td style="padding:.3rem .5rem;text-align:right;'
            f'background:{rate_color};font-family:monospace">'
            f'{rate * 100:.0f}%</td>'
            f'</tr>'
        )
    parts.append("</tbody></table></div>")
    return "".join(parts)


__all__ = [
    "build_lexical_modernization_html",
]
