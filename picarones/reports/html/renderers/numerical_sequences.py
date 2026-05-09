"""Rendu HTML « Précision sur séquences numériques » — Sprint 86.

Phase 5.C.batch7 — module relocalisé depuis
``picarones.report.numerical_sequences_render`` vers
``picarones.reports.html.renderers.numerical_sequences``.
Le chemin legacy reste disponible via un shim avec
``DeprecationWarning`` ; suppression prévue en 2.0.

Suite directe ``picarones/core/numerical_sequences.py``
(Sprint 85) + câblage runner Sprint 86.

Pattern identique aux autres rendus : server-side, pas de JS,
anti-injection systématique.

Vue
---
Tableau moteur × catégorie (year / roman / foliation / currency
/ regnal) × score strict ; une ligne par moteur, une cellule
colorée par cellule.  Une seconde ligne donne le score ``value``
(en plus petit).  Catégorie omise si **aucun** moteur n'a de
GT exploitable pour elle.

Adaptative : ``""`` si aucun moteur n'a de
``aggregated_numerical_sequences``.
"""

from __future__ import annotations

from html import escape as _e
from typing import Optional

from picarones.evaluation.metrics.numerical_sequences import CATEGORIES
from picarones.reports._helpers.render_helpers import color_traffic_light


def _category_columns_with_signal(rows: list[dict]) -> list[str]:
    """Ne garde que les catégories où ≥ 1 moteur a un n_total > 0."""
    visible: list[str] = []
    for cat in CATEGORIES:
        for r in rows:
            agg = r.get("aggregated_numerical_sequences") or {}
            cat_data = (agg.get("per_category") or {}).get(cat) or {}
            if (cat_data.get("n_total") or 0) > 0:
                visible.append(cat)
                break
    return visible


def build_numerical_sequences_html(
    engines: list[dict],
    labels: Optional[dict[str, str]] = None,
) -> str:
    """Construit la section HTML séquences numériques.

    Returns
    -------
    str
        ``""`` si aucun moteur n'a de signal.
    """
    rows = [
        e for e in engines
        if isinstance(e.get("aggregated_numerical_sequences"), dict)
    ]
    if not rows:
        return ""
    visible_cats = _category_columns_with_signal(rows)
    if not visible_cats:
        return ""
    labels = labels or {}
    title = labels.get(
        "numseq_title", "Précision sur séquences numériques",
    )
    note = labels.get(
        "numseq_note",
        "Score strict (forme préservée) — la valeur entre "
        "parenthèses est le score sur la valeur (XIV ↔ 14 "
        "accepté). Foliotation : recto/verso non interchangeables.",
    )
    col_engine = labels.get("numseq_engine", "Moteur")
    col_global = labels.get("numseq_global", "Global")
    cat_label = {
        "year": labels.get("numseq_cat_year", "Année"),
        "roman": labels.get("numseq_cat_roman", "Romain"),
        "foliation": labels.get("numseq_cat_foliation", "Foliation"),
        "currency": labels.get("numseq_cat_currency", "Montant"),
        "regnal": labels.get("numseq_cat_regnal", "Régnal"),
    }

    parts = [
        '<div class="numseq-section" style="margin:1rem 0">',
        f'<h3 style="margin:0 0 .3rem 0">{_e(title)}</h3>',
        f'<div style="font-size:.85rem;opacity:.75;margin-bottom:.5rem">'
        f'{_e(note)}</div>',
        '<table style="border-collapse:collapse;width:100%;'
        'font-size:.9rem">',
        '<thead><tr>',
        f'<th scope=\"col\" style="padding:.4rem .6rem;text-align:left;'
        f'border-bottom:1px solid #ccc;font-weight:600">'
        f'{_e(col_engine)}</th>',
        f'<th scope=\"col\" style="padding:.4rem .6rem;text-align:right;'
        f'border-bottom:1px solid #ccc;font-weight:600">'
        f'{_e(col_global)}</th>',
    ]
    for cat in visible_cats:
        parts.append(
            f'<th scope=\"col\" style="padding:.4rem .6rem;text-align:right;'
            f'border-bottom:1px solid #ccc;font-weight:600">'
            f'{_e(cat_label.get(cat, cat))}</th>'
        )
    parts.append("</tr></thead><tbody>")

    for engine in rows:
        agg = engine["aggregated_numerical_sequences"]
        name = engine.get("name") or "?"
        per_cat = agg.get("per_category") or {}
        global_strict = float(agg.get("global_strict_score") or 0.0)
        global_value = float(agg.get("global_value_score") or 0.0)
        n_total = int(agg.get("n_total") or 0)
        global_color = color_traffic_light(global_strict)
        parts.append(
            f'<tr>'
            f'<td style="padding:.4rem .6rem">{_e(str(name))}</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'background:{global_color};font-family:monospace;'
            f'font-weight:600">'
            f'{global_strict * 100:.1f}%'
            f'<span style="font-size:.75rem;font-weight:400;'
            f'opacity:.75"> ({global_value * 100:.0f}%, '
            f'n={n_total})</span></td>'
        )
        for cat in visible_cats:
            cat_data = per_cat.get(cat) or {}
            n = int(cat_data.get("n_total") or 0)
            if n == 0:
                parts.append(
                    '<td style="padding:.4rem .6rem;text-align:right;'
                    'opacity:.4">—</td>'
                )
                continue
            strict = float(cat_data.get("strict_score") or 0.0)
            value = float(cat_data.get("value_score") or 0.0)
            color = color_traffic_light(strict)
            parts.append(
                f'<td style="padding:.4rem .6rem;text-align:right;'
                f'background:{color};font-family:monospace">'
                f'{strict * 100:.0f}%'
                f'<span style="font-size:.75rem;opacity:.75"> '
                f'({value * 100:.0f}%, n={n})</span></td>'
            )
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


__all__ = ["build_numerical_sequences_html"]
