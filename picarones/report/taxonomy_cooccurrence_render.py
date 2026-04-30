"""Rendu HTML de la heatmap de co-occurrence taxonomique — Sprint 75.

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


def _color_for_jaccard(j: float) -> str:
    """Gradient blanc → bleu profond pour Jaccard ∈ [0, 1].

    Interpolation entre #ffffff (j=0) et #1e3a8a (j=1).
    """
    f = max(0.0, min(1.0, j))
    r = int(255 + (30 - 255) * f)
    g = int(255 + (58 - 255) * f)
    b = int(255 + (138 - 255) * f)
    return f"#{r:02x}{g:02x}{b:02x}"


def _text_color_for_bg(j: float) -> str:
    """Texte blanc si fond foncé, noir sinon (lisibilité)."""
    return "#fff" if j > 0.55 else "#222"


def _build_heatmap_svg(
    classes: list[str],
    matrix: dict[str, dict[str, float]],
    *,
    cell_size: int = 36,
    label_left: int = 130,
    label_top: int = 80,
) -> str:
    """Construit la heatmap SVG.

    Cellule = carré coloré ``_color_for_jaccard``, valeur Jaccard
    affichée en chiffres si > 0,05.  Étiquettes des classes en
    colonne (haut) et en ligne (gauche).
    """
    n = len(classes)
    if n == 0:
        return ""
    width = label_left + n * cell_size + 10
    height = label_top + n * cell_size + 10

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="Heatmap Jaccard co-occurrence taxonomique">',
    ]
    # Étiquettes de colonnes (rotées -45°)
    for j, cls in enumerate(classes):
        cx = label_left + j * cell_size + cell_size // 2
        cy = label_top - 6
        parts.append(
            f'<text x="{cx}" y="{cy}" '
            f'transform="rotate(-45 {cx} {cy})" '
            f'font-size="11" fill="#333" text-anchor="start">'
            f'{_e(cls)}</text>'
        )
    # Étiquettes de lignes
    for i, cls in enumerate(classes):
        rx = label_left - 6
        ry = label_top + i * cell_size + cell_size // 2 + 4
        parts.append(
            f'<text x="{rx}" y="{ry}" '
            f'font-size="11" fill="#333" text-anchor="end">'
            f'{_e(cls)}</text>'
        )
    # Cellules
    for i, ca in enumerate(classes):
        for j, cb in enumerate(classes):
            value = matrix.get(ca, {}).get(cb, 0.0)
            x = label_left + j * cell_size
            y = label_top + i * cell_size
            color = _color_for_jaccard(value)
            text_color = _text_color_for_bg(value)
            parts.append(
                f'<rect x="{x}" y="{y}" '
                f'width="{cell_size}" height="{cell_size}" '
                f'fill="{color}" stroke="#ddd" stroke-width="0.5"/>'
            )
            if value > 0.05:
                parts.append(
                    f'<text x="{x + cell_size // 2}" '
                    f'y="{y + cell_size // 2 + 4}" '
                    f'font-size="10" fill="{text_color}" '
                    f'text-anchor="middle">'
                    f'{value:.2f}</text>'
                )
    parts.append("</svg>")
    return "".join(parts)


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
        f'<th style="padding:.3rem .5rem;text-align:left;'
        f'border-bottom:1px solid #ccc;font-weight:600">'
        f'{_e(pair_label)}</th>',
        f'<th style="padding:.3rem .5rem;text-align:right;'
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
            f'font-family:monospace;background:{_color_for_jaccard(j)};'
            f'color:{_text_color_for_bg(j)}">{j:.2f}</td>'
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

    svg = _build_heatmap_svg(classes, matrix)
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
