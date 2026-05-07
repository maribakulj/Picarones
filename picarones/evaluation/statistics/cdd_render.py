"""Rendu SVG du Critical Difference Diagram (Sprint 17).

Visualisation canonique du résultat Friedman-Nemenyi (Demšar 2006) :
axe horizontal des rangs moyens + barres horizontales reliant les
moteurs statistiquement indiscernables au seuil α.

Module séparé du calcul (:mod:`friedman_nemenyi`) pour respecter la
distinction "computation vs presentation" : on peut imaginer un
rendu PNG, PDF, ou autre, sans toucher au calcul.
"""

from __future__ import annotations


def build_critical_difference_svg(
    nemenyi_result: dict,
    width: int = 780,
    row_height: int = 22,
) -> str:
    """Génère le SVG du Critical Difference Diagram (Demšar 2006).

    Le diagramme montre :
      * un axe horizontal des rangs moyens (1 à k),
      * chaque moteur positionné sur l'axe à son rang moyen,
      * des barres horizontales épaisses reliant les moteurs statistiquement
        indiscernables (distance ≤ CD),
      * la longueur de CD affichée au-dessus de l'axe en référence.

    Parameters
    ----------
    nemenyi_result:
        Résultat de ``nemenyi_posthoc``.
    width:
        Largeur totale du SVG en pixels.
    row_height:
        Hauteur de chaque ligne d'étiquette moteur (auto-adaptatif).

    Returns
    -------
    Chaîne contenant le SVG (balise racine ``<svg>…</svg>``).
    """
    k = nemenyi_result.get("n_engines", 0)
    if k < 2 or nemenyi_result.get("error"):
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="40" '
            'role="img" aria-label="Critical Difference Diagram indisponible">'
            '<text x="10" y="24" font-family="sans-serif" font-size="12" fill="#666">'
            'Critical Difference Diagram non calculable — données insuffisantes.'
            '</text></svg>'
        )

    engines_sorted: list[str] = list(nemenyi_result.get("engines_sorted", []))
    mean_ranks: dict[str, float] = dict(nemenyi_result.get("mean_ranks", {}))
    tied_groups: list[list[str]] = list(nemenyi_result.get("tied_groups", []))
    cd: float = float(nemenyi_result.get("critical_distance", 0.0))

    # Dimensions
    left_pad, right_pad = 40, 40
    top_pad = 50   # espace pour l'affichage CD
    axis_y = top_pad + 10
    bars_start_y = axis_y + 20  # première barre d'ex-aequo sous l'axe
    # Empiler une ligne par groupe + une ligne par étiquette
    label_rows = k  # chaque moteur a sa propre ligne de label
    bars_count = len(tied_groups)
    total_h = bars_start_y + bars_count * 10 + label_rows * row_height + 20

    axis_x0, axis_x1 = left_pad, width - right_pad
    axis_width = axis_x1 - axis_x0

    def x_for_rank(r: float) -> float:
        # Rang 1 à gauche, rang k à droite
        if k <= 1:
            return axis_x0
        return axis_x0 + (r - 1.0) / (k - 1.0) * axis_width

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" viewBox="0 0 {width} {total_h}" '
        f'role="img" aria-label="Critical Difference Diagram (Friedman-Nemenyi)" '
        f'font-family="system-ui, -apple-system, sans-serif">'
    )
    parts.append('<style>.cd-axis{stroke:#334155;stroke-width:1.5}.cd-tick{stroke:#334155;stroke-width:1}'
                 '.cd-label{fill:#0f172a;font-size:11px}'
                 '.cd-tie{stroke:#0f172a;stroke-width:4;stroke-linecap:round}'
                 '.cd-cd-bar{stroke:#dc2626;stroke-width:2}'
                 '.cd-cd-txt{fill:#dc2626;font-size:11px;font-weight:600}'
                 '.cd-name{fill:#0f172a;font-size:12px}'
                 '.cd-rank{fill:#64748b;font-size:10px}'
                 '</style>')

    # Barre CD de référence (en haut, à gauche de l'axe)
    if cd > 0 and k >= 2:
        cd_bar_x0 = axis_x0
        cd_bar_x1 = axis_x0 + (cd / max(1, k - 1)) * axis_width
        cd_y = top_pad - 20
        parts.append(f'<line class="cd-cd-bar" x1="{cd_bar_x0:.1f}" y1="{cd_y}" '
                     f'x2="{cd_bar_x1:.1f}" y2="{cd_y}"/>')
        parts.append(f'<line class="cd-cd-bar" x1="{cd_bar_x0:.1f}" y1="{cd_y - 4}" '
                     f'x2="{cd_bar_x0:.1f}" y2="{cd_y + 4}"/>')
        parts.append(f'<line class="cd-cd-bar" x1="{cd_bar_x1:.1f}" y1="{cd_y - 4}" '
                     f'x2="{cd_bar_x1:.1f}" y2="{cd_y + 4}"/>')
        parts.append(f'<text class="cd-cd-txt" x="{(cd_bar_x0 + cd_bar_x1)/2:.1f}" y="{cd_y - 8}" '
                     f'text-anchor="middle">CD = {cd:.3f}</text>')

    # Axe principal
    parts.append(f'<line class="cd-axis" x1="{axis_x0}" y1="{axis_y}" '
                 f'x2="{axis_x1}" y2="{axis_y}"/>')
    # Ticks entiers
    for r in range(1, k + 1):
        xt = x_for_rank(r)
        parts.append(f'<line class="cd-tick" x1="{xt:.1f}" y1="{axis_y - 5}" '
                     f'x2="{xt:.1f}" y2="{axis_y + 5}"/>')
        parts.append(f'<text class="cd-label" x="{xt:.1f}" y="{axis_y - 9}" '
                     f'text-anchor="middle">{r}</text>')

    # Barres reliant les groupes indiscernables
    for i, group in enumerate(tied_groups):
        if len(group) < 2:
            continue
        rs = [mean_ranks[n] for n in group]
        x0 = x_for_rank(min(rs))
        x1 = x_for_rank(max(rs))
        y_bar = bars_start_y + i * 10
        parts.append(f'<line class="cd-tie" x1="{x0 - 3:.1f}" y1="{y_bar}" '
                     f'x2="{x1 + 3:.1f}" y2="{y_bar}"/>')

    # Étiquettes des moteurs : la moitié la plus basse à gauche, l'autre à droite
    labels_y_base = bars_start_y + bars_count * 10 + 15
    half = (len(engines_sorted) + 1) // 2
    left_engines = engines_sorted[:half]
    right_engines = engines_sorted[half:]

    for idx, name in enumerate(left_engines):
        r = mean_ranks[name]
        x = x_for_rank(r)
        y_label = labels_y_base + idx * row_height
        # Ligne du moteur vers axe
        parts.append(f'<line class="cd-tick" x1="{x:.1f}" y1="{axis_y + 6}" '
                     f'x2="{x:.1f}" y2="{y_label - 4}"/>')
        parts.append(f'<line class="cd-tick" x1="{x:.1f}" y1="{y_label - 4}" '
                     f'x2="{axis_x0 - 4:.1f}" y2="{y_label - 4}"/>')
        parts.append(f'<text class="cd-name" x="{axis_x0 - 6:.1f}" y="{y_label}" '
                     f'text-anchor="end">{_svg_escape(name)} '
                     f'<tspan class="cd-rank">({r:.2f})</tspan></text>')

    for idx, name in enumerate(right_engines):
        r = mean_ranks[name]
        x = x_for_rank(r)
        y_label = labels_y_base + idx * row_height
        parts.append(f'<line class="cd-tick" x1="{x:.1f}" y1="{axis_y + 6}" '
                     f'x2="{x:.1f}" y2="{y_label - 4}"/>')
        parts.append(f'<line class="cd-tick" x1="{x:.1f}" y1="{y_label - 4}" '
                     f'x2="{axis_x1 + 4:.1f}" y2="{y_label - 4}"/>')
        parts.append(f'<text class="cd-name" x="{axis_x1 + 6:.1f}" y="{y_label}" '
                     f'text-anchor="start">{_svg_escape(name)} '
                     f'<tspan class="cd-rank">({r:.2f})</tspan></text>')

    parts.append('</svg>')
    return "".join(parts)


def _svg_escape(text: str) -> str:
    """Échappe un texte pour inclusion sûre dans un nœud SVG/XML."""
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&#39;"))


__all__ = ["build_critical_difference_svg"]
