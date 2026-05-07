"""Rendu HTML « Lisibilité (delta Flesch) » — Sprint 87 (A.II.2).

Phase 5.C — module relocalisé depuis
``picarones.report.readability_render`` vers
``picarones.reports_v2.html.renderers.readability``.  Le chemin
legacy reste disponible via un shim avec ``DeprecationWarning`` ;
suppression prévue en 2.0.

Suite directe ``picarones/core/readability.py`` (Sprint 52) +
câblage runner Sprint 87.

Pattern identique aux autres rendus : server-side, pas de JS,
anti-injection systématique.

Vue
---
Tableau résumé moteur × {delta_mean, delta_median, %
over-normalisés, % under-normalisés, n_docs}.  Cellule delta_mean
colorée par gradient :

- vert (delta ≈ 0) : OCR fidèle à la GT en complexité.
- orange (delta > 5) : over-normalisation (typique LLM).
- bleu (delta < -5) : dégradation OCR brutale.

Adaptative : ``""`` si aucun moteur n'a de
``aggregated_readability``.
"""

from __future__ import annotations

from html import escape as _e
from typing import Optional

from picarones.reports_v2._helpers.render_helpers import color_diverging


def _bg_for_flesch_delta(delta: float) -> str:
    """Vert au centre (delta ≈ 0), orange en sur-normalisation (delta > 0),
    bleu en sous-normalisation (delta < 0). Saturation à ±15 pts Flesch.
    """
    if abs(delta) <= 1.0:
        return "#a7f0a7"  # neutre vert clair, indistinguable du bruit
    return color_diverging(
        delta,
        max_abs=15.0,
        neutral_rgb=(167, 240, 167),
        positive_rgb=(220, 140, 60),
        negative_rgb=(90, 160, 210),
    )


def build_readability_summary_html(
    engines: list[dict],
    labels: Optional[dict[str, str]] = None,
) -> str:
    """Construit la table HTML lisibilité.

    Returns ``""`` si aucun moteur n'a de signal.
    """
    rows = [
        e for e in engines
        if isinstance(e.get("aggregated_readability"), dict)
    ]
    if not rows:
        return ""
    labels = labels or {}
    title = labels.get("readability_title", "Lisibilité (delta Flesch)")
    note = labels.get(
        "readability_note",
        "Différence de lisibilité Flesch entre la sortie OCR et la "
        "GT. Δ > +5 : over-normalisation (typique des LLM qui "
        "modernisent un texte ancien). Δ < -5 : dégradation "
        "brutale. Δ ≈ 0 : fidélité au registre linguistique.",
    )
    col_engine = labels.get("readability_engine", "Moteur")
    col_mean = labels.get("readability_delta_mean", "Δ moyen")
    col_median = labels.get("readability_delta_median", "Δ médian")
    col_over = labels.get(
        "readability_over_norm_rate", "% over-normalisé",
    )
    col_under = labels.get(
        "readability_under_norm_count", "Docs under-normalisés",
    )
    col_docs = labels.get("readability_docs", "Docs")

    parts = [
        '<div class="readability-section" style="margin:1rem 0">',
        f'<h3 style="margin:0 0 .3rem 0">{_e(title)}</h3>',
        f'<div style="font-size:.85rem;opacity:.75;margin-bottom:.5rem">'
        f'{_e(note)}</div>',
        '<table style="border-collapse:collapse;width:100%;'
        'font-size:.9rem">',
        '<thead><tr>',
    ]
    for col in (col_engine, col_mean, col_median, col_over,
                col_under, col_docs):
        parts.append(
            f'<th scope=\"col\" style="padding:.4rem .6rem;text-align:left;'
            f'border-bottom:1px solid #ccc;font-weight:600">'
            f'{_e(col)}</th>'
        )
    parts.append("</tr></thead><tbody>")
    for engine in rows:
        agg = engine["aggregated_readability"]
        name = engine.get("name") or "?"
        delta_mean = float(agg.get("delta_mean") or 0.0)
        delta_median = float(agg.get("delta_median") or 0.0)
        over_rate = float(agg.get("over_normalized_rate") or 0.0)
        n_under = int(agg.get("n_under_normalized") or 0)
        n_docs = int(agg.get("n_docs") or 0)
        color = _bg_for_flesch_delta(delta_mean)
        parts.append(
            f'<tr>'
            f'<td style="padding:.4rem .6rem">{_e(str(name))}</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'background:{color};font-family:monospace;font-weight:600">'
            f'{delta_mean:+.2f}</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'font-family:monospace">{delta_median:+.2f}</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'font-family:monospace">{over_rate * 100:.0f}%</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'font-family:monospace">{n_under}</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'font-family:monospace">{n_docs}</td>'
            f'</tr>'
        )
    parts.append("</tbody></table></div>")
    return "".join(parts)


__all__ = ["build_readability_summary_html"]
