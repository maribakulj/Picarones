"""Rendu HTML de la heatmap de co-occurrence taxonomique

A.I.4 chantier 1 du plan d'évolution 2026.

Suite directe ``picarones/core/taxonomy_cooccurrence.py``.  Pattern
identique aux autres rendus (Sprints 41/43/62/67/72/74) :
**server-side**, pas de JavaScript, anti-injection systématique.

Sortie typique
--------------
- ``build_taxonomy_cooccurrence_html(data, labels)`` produit un
  bloc complet : titre + note d'usage + heatmap SVG + table des
  paires les plus co-occurrentes.
- ``""`` retourné si ``data is None`` ou si la matrice est vide
  (rapport adaptatif).
"""

from __future__ import annotations

from html import escape as _e
from typing import Optional

from picarones.reports._helpers.render_helpers import (
    GRADIENT_TARGET_BLUE,
    build_grid_svg,
    color_single_gradient,
    text_color_for_bg,
)


def _build_jaccard_heatmap_svg(
    classes: list[str],
    matrix: dict[str, dict[str, float]],
    *,
    cell_size: int = 36,
    label_left: int = 130,
    label_top: int = 80,
) -> str:
    """Heatmap Jaccard de co-occurrence taxonomique.

    Délègue à :func:`build_grid_svg` ; reste un wrapper local qui
    encapsule les conventions spécifiques à la matrice symétrique
    (valeur affichée seulement si > 0,05, étiquettes rotées).
    """
    if not classes:
        return ""

    def cell_value(i: int, j: int) -> float:
        return matrix.get(classes[i], {}).get(classes[j], 0.0)

    return build_grid_svg(
        n_rows=len(classes),
        n_cols=len(classes),
        row_label_fn=lambda i: classes[i],
        col_label_fn=lambda j: classes[j],
        cell_color_fn=lambda i, j: color_single_gradient(
            cell_value(i, j), end_rgb=GRADIENT_TARGET_BLUE,
        ),
        cell_text_fn=lambda i, j: (
            f"{cell_value(i, j):.2f}" if cell_value(i, j) > 0.05 else None
        ),
        cell_text_color_fn=lambda i, j: text_color_for_bg(cell_value(i, j)),
        cell_w=cell_size,
        cell_h=cell_size,
        label_left=label_left,
        label_top=label_top,
        rotate_col_labels=True,
        aria_label="Heatmap Jaccard co-occurrence taxonomique",
    )


def _build_top_pairs_table(
    top_pairs: list,
    labels: dict,
) -> str:
    """Construit la table HTML des paires les plus co-occurrentes."""
    if not top_pairs:
        return ""
    pair_label = labels.get("taxocooc_pair_label", "Paire")
    jaccard_label = labels.get("taxocooc_jaccard_label", "Jaccard")

    parts = [
        '<table style="border-collapse:collapse;font-size:.85rem;'
        'margin-top:.5rem">',
        '<thead><tr>',
        f'<th scope=\"col\" style="padding:.3rem .5rem;text-align:left;'
        f'border-bottom:1px solid #ccc;font-weight:600">'
        f'{_e(pair_label)}</th>',
        f'<th scope=\"col\" style="padding:.3rem .5rem;text-align:right;'
        f'border-bottom:1px solid #ccc;font-weight:600">'
        f'{_e(jaccard_label)}</th>',
        '</tr></thead><tbody>',
    ]
    for ca, cb, j in top_pairs:
        parts.append(
            f'<tr>'
            f'<td style="padding:.2rem .5rem">'
            f'<code>{_e(ca)}</code> ↔ <code>{_e(cb)}</code></td>'
            f'<td style="padding:.2rem .5rem;text-align:right;'
            f'font-family:monospace;'
            f'background:{color_single_gradient(j, end_rgb=GRADIENT_TARGET_BLUE)};'
            f'color:{text_color_for_bg(j)}">{j:.2f}</td>'
            f'</tr>'
        )
    parts.append("</tbody></table>")
    return "".join(parts)


def build_taxonomy_cooccurrence_html(
    data: Optional[dict],
    labels: Optional[dict[str, str]] = None,
) -> str:
    """Construit le bloc HTML complet de co-occurrence taxonomique.

    Retourne ``""`` si ``data is None`` ou matrice vide.
    """
    if not data:
        return ""
    classes = data.get("classes") or []
    matrix = data.get("cooccurrence_matrix") or {}
    if not classes or not matrix:
        return ""
    labels = labels or {}
    title = labels.get(
        "taxocooc_title",
        "Co-occurrence des classes d'erreur",
    )
    note = labels.get(
        "taxocooc_note",
        "Indice de Jaccard au niveau document : 1,00 = ces deux classes "
        "apparaissent toujours ensemble ; 0,00 = jamais. Lecture par paires "
        "co-occurrentes ci-dessous.",
    )
    n_docs = data.get("n_documents", 0)
    n_docs_label_template = labels.get(
        "taxocooc_n_docs", "Calculé sur {n_docs} documents.",
    )
    n_docs_phrase = n_docs_label_template.format(n_docs=n_docs)

    svg = _build_jaccard_heatmap_svg(classes, matrix)
    top_table = _build_top_pairs_table(
        data.get("top_pairs") or [], labels,
    )

    parts = [
        '<div class="taxocooc" style="margin:1rem 0">',
        f'<div style="font-weight:600;margin-bottom:.4rem">{_e(title)}</div>',
        f'<div style="font-size:.85rem;opacity:.75;margin-bottom:.5rem">'
        f'{_e(note)}</div>',
        f'<div style="font-size:.8rem;opacity:.7;margin-bottom:.5rem">'
        f'{_e(n_docs_phrase)}</div>',
        svg,
        top_table,
        "</div>",
    ]
    return "".join(parts)


__all__ = [
    "build_taxonomy_cooccurrence_html",
]
