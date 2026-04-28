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


def _color_for_density(density: float) -> str:
    """Gradient blanc → orange profond pour densité ∈ [0, 1].

    Interpolation entre #ffffff (0) et #c2410c (1).
    """
    f = max(0.0, min(1.0, density))
    r = int(255 + (194 - 255) * f)
    g = int(255 + (65 - 255) * f)
    b = int(255 + (12 - 255) * f)
    return f"#{r:02x}{g:02x}{b:02x}"


def _text_color_for_bg(density: float) -> str:
    return "#fff" if density > 0.55 else "#222"


def _build_heatmap_svg(
    classes_with_errors: list[str],
    per_class: dict[str, list[int]],
    n_bins: int,
    *,
    cell_w: int = 36,
    cell_h: int = 26,
    label_left: int = 150,
    label_top: int = 30,
) -> str:
    """Construit la heatmap SVG class × position."""
    n_rows = len(classes_with_errors)
    if n_rows == 0:
        return ""
    width = label_left + n_bins * cell_w + 10
    height = label_top + n_rows * cell_h + 30  # +30 pour étiquette X

    # Normalisation : pour chaque classe, densité relative au max
    # de cette classe (mise en évidence des positions concentrées).
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="Heatmap class taxonomique × position">',
    ]
    # Étiquettes des colonnes (positions)
    for j in range(n_bins):
        cx = label_left + j * cell_w + cell_w // 2
        cy = label_top - 6
        parts.append(
            f'<text x="{cx}" y="{cy}" '
            f'font-size="10" fill="#666" text-anchor="middle">'
            f'{j + 1}</text>'
        )
    # Cellules
    for i, cls in enumerate(classes_with_errors):
        # Étiquette de ligne (classe)
        rx = label_left - 6
        ry = label_top + i * cell_h + cell_h // 2 + 4
        parts.append(
            f'<text x="{rx}" y="{ry}" '
            f'font-size="11" fill="#333" text-anchor="end">'
            f'{_e(cls)}</text>'
        )
        counts = per_class.get(cls, [0] * n_bins)
        max_count = max(counts) if counts else 0
        for j in range(n_bins):
            x = label_left + j * cell_w
            y = label_top + i * cell_h
            count = counts[j] if j < len(counts) else 0
            density = (count / max_count) if max_count > 0 else 0.0
            color = _color_for_density(density)
            text_color = _text_color_for_bg(density)
            parts.append(
                f'<rect x="{x}" y="{y}" '
                f'width="{cell_w}" height="{cell_h}" '
                f'fill="{color}" stroke="#ddd" stroke-width="0.5"/>'
            )
            if count > 0:
                parts.append(
                    f'<text x="{x + cell_w // 2}" '
                    f'y="{y + cell_h // 2 + 4}" '
                    f'font-size="10" fill="{text_color}" '
                    f'text-anchor="middle">{count}</text>'
                )
    # Étiquette axe X en bas
    cx_axis = label_left + (n_bins * cell_w) // 2
    cy_axis = height - 6
    parts.append(
        f'<text x="{cx_axis}" y="{cy_axis}" '
        f'font-size="11" fill="#666" text-anchor="middle" '
        f'font-style="italic">'
        f'Position dans le document (1 = début)</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


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

    svg = _build_heatmap_svg(classes_with_errors, per_class, n_bins)

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
