"""Rendu HTML du diagramme miroir taxonomique — Sprint 77.

Phase 5.C — module relocalisé depuis
``picarones.report.taxonomy_comparison_render`` vers
``picarones.reports.html.renderers.taxonomy_comparison``.
Le chemin legacy reste disponible via un shim avec
``DeprecationWarning`` ; suppression prévue en 2.0.

A.I.4 chantier 3 du plan d'évolution 2026.

Suite directe ``picarones/core/taxonomy_comparison.py``.  Pattern
identique aux autres rendus (Sprints 41/43/62/67/72/74/75/76) :
**server-side**, pas de JavaScript, anti-injection systématique.

Diagramme miroir
----------------
Une ligne par classe taxonomique, divisée en deux barres
horizontales :

- À **gauche** : barre du moteur A (orientée vers la gauche, du
  centre vers le bord).
- À **droite** : barre du moteur B (orientée vers la droite).
- Couleur de la classe selon ``recoverability`` :

  - vert (#5fa860) : ``recoverable``
  - orange (#e0a050) : ``difficult``
  - rouge (#d8553b) : ``irrecoverable``

Lecture immédiate : un moteur dont les barres tirent vers la
**gauche** sur du vert (case_error, ligature_error) et un moteur
qui tire à droite sur du rouge (lacuna) — la décision éditoriale
est évidente même si les CER globaux sont identiques.
"""

from __future__ import annotations

from html import escape as _e
from typing import Optional


_RECOVERABILITY_COLORS = {
    "recoverable":   "#5fa860",
    "difficult":     "#e0a050",
    "irrecoverable": "#d8553b",
}


def _build_mirror_chart_svg(
    data: dict,
    *,
    bar_max_width: int = 200,
    row_height: int = 22,
    label_width: int = 140,
    margin_top: int = 50,
    margin_bottom: int = 20,
) -> str:
    """Construit le diagramme miroir SVG."""
    classes = data["classes"]
    prop_a = data["proportions_a"]
    prop_b = data["proportions_b"]
    recov = data["recoverability"]
    engine_a = data["engine_a"]
    engine_b = data["engine_b"]

    n_rows = len(classes)
    if n_rows == 0:
        return ""

    # Échelle : on normalise à la valeur max de toutes les
    # proportions (pour que la classe la plus présente atteigne
    # bar_max_width).
    max_prop = max(
        max(prop_a.values(), default=0.0),
        max(prop_b.values(), default=0.0),
    )
    if max_prop <= 0:
        max_prop = 1.0  # évite division par zéro (cas dégénéré)

    width = label_width + 2 * bar_max_width + 40
    height = margin_top + n_rows * row_height + margin_bottom
    center = width // 2

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="Diagramme miroir taxonomique">',
        # En-têtes des deux moteurs
        f'<text x="{center - bar_max_width // 2}" y="20" '
        f'font-size="13" font-weight="600" fill="#333" '
        f'text-anchor="middle">{_e(engine_a)}</text>',
        f'<text x="{center + bar_max_width // 2}" y="20" '
        f'font-size="13" font-weight="600" fill="#333" '
        f'text-anchor="middle">{_e(engine_b)}</text>',
        # Ligne centrale
        f'<line x1="{center}" y1="{margin_top - 4}" '
        f'x2="{center}" y2="{height - margin_bottom + 4}" '
        f'stroke="#999" stroke-width="1"/>',
    ]

    # Barres
    for i, cls in enumerate(classes):
        y = margin_top + i * row_height
        level = recov.get(cls, "difficult")
        color = _RECOVERABILITY_COLORS.get(level, "#888")
        # Étiquette de classe au centre
        parts.append(
            f'<text x="{center}" y="{y + row_height // 2 + 4}" '
            f'font-size="11" fill="#222" text-anchor="middle" '
            f'font-family="monospace">{_e(cls)}</text>'
        )
        # Barre A (gauche)
        a_width = (prop_a.get(cls, 0.0) / max_prop) * bar_max_width
        if a_width > 0:
            x_a = center - label_width // 2 - a_width
            parts.append(
                f'<rect x="{x_a:.1f}" y="{y + 3}" '
                f'width="{a_width:.1f}" height="{row_height - 6}" '
                f'fill="{color}" stroke="#666" stroke-width="0.5" '
                f'opacity="0.85"/>'
            )
            # Valeur en %
            parts.append(
                f'<text x="{x_a - 3:.1f}" y="{y + row_height // 2 + 4}" '
                f'font-size="10" fill="#444" text-anchor="end">'
                f'{prop_a.get(cls, 0.0) * 100:.1f}%</text>'
            )
        # Barre B (droite)
        b_width = (prop_b.get(cls, 0.0) / max_prop) * bar_max_width
        if b_width > 0:
            x_b = center + label_width // 2
            parts.append(
                f'<rect x="{x_b:.1f}" y="{y + 3}" '
                f'width="{b_width:.1f}" height="{row_height - 6}" '
                f'fill="{color}" stroke="#666" stroke-width="0.5" '
                f'opacity="0.85"/>'
            )
            parts.append(
                f'<text x="{x_b + b_width + 3:.1f}" '
                f'y="{y + row_height // 2 + 4}" '
                f'font-size="10" fill="#444" text-anchor="start">'
                f'{prop_b.get(cls, 0.0) * 100:.1f}%</text>'
            )
    parts.append("</svg>")
    return "".join(parts)


def _build_recoverability_summary_html(
    data: dict, labels: dict,
) -> str:
    """Encart résumé par catégorie de récupérabilité (3 lignes)."""
    totals = data.get("totals_by_recoverability") or {}
    if not totals:
        return ""
    label_recov = labels.get("taxocomp_recoverable", "Récupérable")
    label_diff = labels.get("taxocomp_difficult", "Difficile")
    label_irrec = labels.get("taxocomp_irrecoverable", "Irrécupérable")
    rows = [
        ("recoverable", label_recov),
        ("difficult", label_diff),
        ("irrecoverable", label_irrec),
    ]
    parts = [
        '<table style="border-collapse:collapse;font-size:.85rem;'
        'margin-top:.5rem">',
        '<thead><tr>',
        '<th scope=\"col\" style="padding:.2rem .5rem;text-align:left;'
        'border-bottom:1px solid #ccc">'
        f'{_e(labels.get("taxocomp_level_label", "Catégorie"))}</th>',
        '<th scope=\"col\" style="padding:.2rem .5rem;text-align:right;'
        'border-bottom:1px solid #ccc">'
        f'{_e(_e(data["engine_a"]))}</th>',
        '<th scope=\"col\" style="padding:.2rem .5rem;text-align:right;'
        'border-bottom:1px solid #ccc">'
        f'{_e(_e(data["engine_b"]))}</th>',
        '</tr></thead><tbody>',
    ]
    for level, label in rows:
        cell = totals.get(level, {"a": 0.0, "b": 0.0})
        color = _RECOVERABILITY_COLORS.get(level, "#888")
        parts.append(
            f'<tr>'
            f'<td style="padding:.2rem .5rem">'
            f'<span style="display:inline-block;width:10px;height:10px;'
            f'background:{color};margin-right:.4rem;border-radius:2px"></span>'
            f'{_e(label)}</td>'
            f'<td style="padding:.2rem .5rem;text-align:right;'
            f'font-family:monospace">{cell["a"] * 100:.1f}%</td>'
            f'<td style="padding:.2rem .5rem;text-align:right;'
            f'font-family:monospace">{cell["b"] * 100:.1f}%</td>'
            f'</tr>'
        )
    parts.append("</tbody></table>")
    return "".join(parts)


def build_taxonomy_comparison_html(
    data: Optional[dict],
    labels: Optional[dict[str, str]] = None,
) -> str:
    """Construit le bloc HTML de comparaison taxonomique entre 2 moteurs.

    Retourne ``""`` si ``data is None`` ou aucune classe.
    """
    if not data:
        return ""
    classes = data.get("classes") or []
    if not classes:
        return ""
    labels = labels or {}
    title_template = labels.get(
        "taxocomp_title", "Profil taxonomique : {engine_a} vs {engine_b}",
    )
    title = title_template.format(
        engine_a=data["engine_a"], engine_b=data["engine_b"],
    )
    note = labels.get(
        "taxocomp_note",
        "Diagramme miroir des proportions d'erreurs par classe. "
        "Couleur selon récupérabilité éditoriale (vert = corrigeable, "
        "rouge = irrécupérable). À CER global égal, un moteur dont les "
        "erreurs sont majoritairement vertes est préférable pour une "
        "édition critique.",
    )
    parts = [
        '<div class="taxocomp" style="margin:1rem 0">',
        f'<div style="font-weight:600;margin-bottom:.4rem">{_e(title)}</div>',
        f'<div style="font-size:.85rem;opacity:.75;margin-bottom:.5rem">'
        f'{_e(note)}</div>',
        _build_mirror_chart_svg(data),
        _build_recoverability_summary_html(data, labels),
        "</div>",
    ]
    return "".join(parts)


__all__ = [
    "build_taxonomy_comparison_html",
]
