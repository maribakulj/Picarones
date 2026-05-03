"""Rendu HTML de la vue « Worst lines globale » — Sprint 72.

Suite directe de ``picarones/core/worst_lines.py`` (extraction
transversale).  Pattern identique aux Sprints 41/43/62/67 : rendu
**server-side**, pas de JavaScript, anti-injection systématique
via ``html.escape``.

Vue distincte du tableau gallery existant
-----------------------------------------
La galerie OCR (vue ``view_gallery.html``) liste les documents
les plus problématiques.  Cette vue va plus fin : elle liste les
**lignes individuelles** les plus problématiques, transversalement
à tous les documents et moteurs.  Complémentaire, pas redondante.
"""

from __future__ import annotations

from html import escape as _e
from typing import Optional

from picarones.measurements.worst_lines import WorstLineEntry
from picarones.core.diff_utils import compute_char_diff
from picarones.report.render_helpers import color_traffic_light


def _bg_for_cer(cer: float) -> str:
    """Beige clair sous le seuil catastrophique (0.30), gradient
    jaune → rouge au-delà.

    Le seuil dur à 0.30 préserve la sémantique « toléré jusqu'à 30 %
    pour un manuscrit difficile ». Au-delà, on entre en zone visible
    avec :func:`color_traffic_light` (low_is_good).
    """
    f = max(0.0, min(1.0, cer))
    if f < 0.3:
        return "#fff8dc"
    return color_traffic_light(f, low_is_good=True, scale_min=0.3, scale_max=1.0)


def _render_diff_inline(reference: str, hypothesis: str) -> str:
    """Rendu HTML inline d'un diff caractère par caractère.

    - ``equal``   → texte normal
    - ``delete``  → fond rouge clair, barré (manquait dans hyp)
    - ``insert``  → fond vert clair (ajouté par hyp)
    - ``replace`` → fond rouge clair barré + fond vert clair pour
      la nouvelle valeur (côte à côte)
    """
    if not reference and not hypothesis:
        return '<span style="opacity:.5">∅</span>'
    ops = compute_char_diff(reference or "", hypothesis or "")
    parts: list[str] = []
    for op in ops:
        kind = op["op"]
        if kind == "equal":
            parts.append(_e(op["text"]))
        elif kind == "delete":
            parts.append(
                f'<span style="background:#fdd;text-decoration:line-through">'
                f'{_e(op["text"])}</span>'
            )
        elif kind == "insert":
            parts.append(
                f'<span style="background:#dfd">{_e(op["text"])}</span>'
            )
        elif kind == "replace":
            parts.append(
                f'<span style="background:#fdd;text-decoration:line-through">'
                f'{_e(op["old"])}</span>'
                f'<span style="background:#dfd">{_e(op["new"])}</span>'
            )
    return "".join(parts)


def build_worst_lines_table_html(
    entries: list[WorstLineEntry],
    labels: Optional[dict[str, str]] = None,
) -> str:
    """Construit le tableau HTML des worst lines.

    Retourne ``""`` si la liste est vide.  Adaptive : si aucune
    entrée n'a de ``script_type``, la colonne strate est omise.
    """
    if not entries:
        return ""
    labels = labels or {}
    title = labels.get("worst_lines_title", "Lignes les plus problématiques")
    note = labels.get(
        "worst_lines_note",
        "Top-N lignes du corpus classées par CER ligne décroissant. "
        "Diff caractère par caractère : rouge barré = manquant dans "
        "l'OCR, vert = ajouté par l'OCR.",
    )
    rank_label = labels.get("worst_lines_rank_label", "Rang")
    cer_label = labels.get("worst_lines_cer_label", "CER")
    engine_label = labels.get("worst_lines_engine_label", "Moteur")
    doc_label = labels.get("worst_lines_doc_label", "Document")
    line_label = labels.get("worst_lines_line_label", "Ligne #")
    strata_label = labels.get("worst_lines_strata_label", "Strate")
    diff_label = labels.get("worst_lines_diff_label", "GT → OCR (diff)")

    has_strata = any(e.script_type for e in entries)

    parts = [
        '<div class="worst-lines" style="margin:1rem 0">',
        f'<div style="font-weight:600;margin-bottom:.4rem">{_e(title)}</div>',
        f'<div style="font-size:.8rem;opacity:.75;margin-bottom:.5rem">'
        f'{_e(note)}</div>',
        '<table style="border-collapse:collapse;width:100%;'
        'font-size:.85rem">',
        '<thead><tr>',
    ]
    cols = [rank_label, cer_label, engine_label, doc_label, line_label]
    if has_strata:
        cols.append(strata_label)
    cols.append(diff_label)
    for col in cols:
        parts.append(
            f'<th scope=\"col\" style="padding:.3rem .5rem;text-align:left;'
            f'border-bottom:1px solid #ccc;font-weight:600">'
            f'{_e(col)}</th>'
        )
    parts.append("</tr></thead><tbody>")
    for entry in entries:
        cer_color = _bg_for_cer(entry.cer)
        parts.append("<tr>")
        parts.append(
            f'<td style="padding:.3rem .5rem;text-align:right;'
            f'font-weight:600">{entry.rank}</td>'
        )
        parts.append(
            f'<td style="padding:.3rem .5rem;text-align:right;'
            f'background:{cer_color};font-family:monospace">'
            f'{entry.cer * 100:.1f}%</td>'
        )
        parts.append(
            f'<td style="padding:.3rem .5rem">{_e(entry.engine_name)}</td>'
        )
        parts.append(
            f'<td style="padding:.3rem .5rem;font-family:monospace;'
            f'font-size:.8rem">{_e(entry.doc_id)}</td>'
        )
        parts.append(
            f'<td style="padding:.3rem .5rem;text-align:right">'
            f'{entry.line_index}</td>'
        )
        if has_strata:
            parts.append(
                f'<td style="padding:.3rem .5rem;font-size:.8rem">'
                f'{_e(entry.script_type or "—")}</td>'
            )
        parts.append(
            f'<td style="padding:.3rem .5rem;font-family:monospace;'
            f'font-size:.85rem">'
            f'{_render_diff_inline(entry.gt_line, entry.hyp_line)}</td>'
        )
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


__all__ = [
    "build_worst_lines_table_html",
]
