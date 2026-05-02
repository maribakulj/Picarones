"""Rendu HTML de la heatmap class × position — Sprint 76.

A.I.4 chantier 2 du plan d'évolution 2026.

Suite directe ``picarones/core/taxonomy_intra_doc.py``.  Pattern
identique aux autres rendus (Sprints 41/43/62/67/72/74/75) :
**server-side**, pas de JavaScript, anti-injection systématique.

Sortie typique
--------------
Une grille N_classes × N_bins où chaque cellule indique la densité
d'erreurs de cette classe à cette position dans le document.
Lecture immédiate : « ligature_error concentré dans la première
tranche → erreur de marge ; visual_confusion uniformément réparti
→ erreur de scribe ».

Adaptive : si ``data is None`` ou si toutes les classes ont 0
erreur, retourne ``""``.
"""

from __future__ import annotations

from html import escape as _e
from typing import Optional

from picarones.report.render_helpers import (
    GRADIENT_TARGET_ORANGE,
    build_grid_svg,
    color_single_gradient,
    text_color_for_bg,
)


def _build_position_heatmap_svg(
    classes_with_errors: list[str],
    per_class: dict[str, list[int]],
    n_bins: int,
    *,
    cell_w: int = 36,
    cell_h: int = 26,
    label_left: int = 150,
    label_top: int = 30,
) -> str:
    """Heatmap class taxonomique × position (densité relative par classe).

    Délègue à :func:`build_grid_svg` ; reste un wrapper local qui
    encapsule la normalisation par classe (densité relative au max
    observé sur la ligne).
    """
    if not classes_with_errors:
        return ""

    # Pré-calcule densité et count par cellule pour éviter les boucles
    # imbriquées dans les callbacks.
    grid: list[list[tuple[int, float]]] = []
    for cls in classes_with_errors:
        counts = per_class.get(cls, [0] * n_bins)
        max_count = max(counts) if counts else 0
        row: list[tuple[int, float]] = []
        for j in range(n_bins):
            count = counts[j] if j < len(counts) else 0
            density = (count / max_count) if max_count > 0 else 0.0
            row.append((count, density))
        grid.append(row)

    return build_grid_svg(
        n_rows=len(classes_with_errors),
        n_cols=n_bins,
        row_label_fn=lambda i: classes_with_errors[i],
        col_label_fn=lambda j: str(j + 1),
        cell_color_fn=lambda i, j: color_single_gradient(
            grid[i][j][1], end_rgb=GRADIENT_TARGET_ORANGE,
        ),
        cell_text_fn=lambda i, j: (
            str(grid[i][j][0]) if grid[i][j][0] > 0 else None
        ),
        cell_text_color_fn=lambda i, j: text_color_for_bg(grid[i][j][1]),
        cell_w=cell_w,
        cell_h=cell_h,
        label_left=label_left,
        label_top=label_top,
        aria_label="Heatmap class taxonomique × position",
        x_axis_title="Position dans le document (1 = début)",
    )


def build_taxonomy_intra_doc_html(
    data: Optional[dict],
    labels: Optional[dict[str, str]] = None,
) -> str:
    """Construit le bloc HTML complet de la heatmap intra-document.

    Retourne ``""`` si ``data is None`` ou aucune erreur.
    """
    if not data:
        return ""
    n_bins = data.get("n_bins", 0)
    per_class = data.get("per_class") or {}
    total_errors = data.get("total_errors", 0)
    if total_errors == 0 or n_bins <= 0:
        return ""
    # Filtre : uniquement les classes ayant au moins une erreur
    classes_with_errors = [
        cls for cls, counts in per_class.items()
        if isinstance(counts, list) and sum(counts) > 0
    ]
    if not classes_with_errors:
        return ""

    labels = labels or {}
    title = labels.get(
        "intradoc_title",
        "Évolution intra-document des classes d'erreur",
    )
    note = labels.get(
        "intradoc_note",
        "Heatmap class × position : densité relative par classe "
        "(plus foncé = concentré). Une classe concentrée dans la "
        "première colonne suggère une erreur de marge ; "
        "une distribution uniforme suggère une erreur de scribe.",
    )
    n_words_gt = data.get("n_words_gt", 0)
    n_words_template = labels.get(
        "intradoc_n_words",
        "Calculé sur {n_words_gt} mots GT, répartis en {n_bins} tranches.",
    )
    n_words_phrase = n_words_template.format(
        n_words_gt=n_words_gt, n_bins=n_bins,
    )

    svg = _build_position_heatmap_svg(classes_with_errors, per_class, n_bins)

    parts = [
        '<div class="intradoc" style="margin:1rem 0">',
        f'<div style="font-weight:600;margin-bottom:.4rem">{_e(title)}</div>',
        f'<div style="font-size:.85rem;opacity:.75;margin-bottom:.5rem">'
        f'{_e(note)}</div>',
        f'<div style="font-size:.8rem;opacity:.7;margin-bottom:.5rem">'
        f'{_e(n_words_phrase)}</div>',
        svg,
        "</div>",
    ]
    return "".join(parts)


__all__ = [
    "build_taxonomy_intra_doc_html",
]
