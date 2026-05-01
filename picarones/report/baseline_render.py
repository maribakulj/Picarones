"""Rendu de l'encart « Ce corpus est-il habituel ? » — Sprint 74.

A.I.3 chantier 1 du plan d'évolution 2026.

Suite directe Sprint 73 (couche calcul + détecteur narratif).  Ce
sprint livre le rendu HTML qui place la difficulté du corpus
courant dans la distribution des corpus précédents stockés en
SQLite (Sprint 8) — un mini-boxplot horizontal en SVG avec un
point pour la position du corpus courant, accompagné d'une phrase
factuelle.

Pattern identique aux autres rendus (Sprints 41/43/62/67/72) :
**server-side**, pas de JavaScript, anti-injection systématique
via ``html.escape``.

Sortie typique
--------------
Un encart court (~80px de haut) à insérer en tête du rapport,
sous la synthèse factuelle :

    Difficulté observée 0,62 — au 88ᵉ percentile des 47 corpus
    précédents de votre institution. Ce corpus est plus difficile
    que la moyenne.

    [boxplot SVG horizontal avec point courant coloré]

Si moins de ``min_runs`` runs historiques ont une difficulté
enregistrée, ``compute_corpus_difficulty_percentile`` retourne
``None`` et le rendu retourne ``""`` (rapport adaptatif).
"""

from __future__ import annotations

import statistics
from html import escape as _e
from typing import Optional


def _quantiles(values: list[float]) -> tuple[float, float, float, float, float]:
    """Retourne (min, Q1, median, Q3, max)."""
    if not values:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    sorted_v = sorted(values)
    n = len(sorted_v)
    if n == 1:
        v = sorted_v[0]
        return (v, v, v, v, v)
    median = statistics.median(sorted_v)
    # Calcul des quartiles avec interpolation linéaire (méthode
    # « inclusive » : Q1 = médiane de la moitié inférieure
    # incluant la médiane si N impair).
    half = n // 2
    if n % 2 == 0:
        lower = sorted_v[:half]
        upper = sorted_v[half:]
    else:
        lower = sorted_v[: half + 1]
        upper = sorted_v[half:]
    q1 = statistics.median(lower)
    q3 = statistics.median(upper)
    return (sorted_v[0], q1, median, q3, sorted_v[-1])


def _build_difficulty_boxplot_svg(
    historical_values: list[float],
    current: float,
    *,
    width: int = 480,
    height: int = 80,
) -> str:
    """Construit un boxplot horizontal SVG avec point courant.

    Le SVG est autonome (pas de CSS externe) et utilise des
    coordonnées explicites — sûr à intégrer dans n'importe quel
    document HTML.
    """
    if not historical_values:
        return ""
    min_v, q1, median, q3, max_v = _quantiles(historical_values)
    # Borne du domaine : on inclut le point courant pour qu'il soit
    # visible même s'il dépasse les valeurs historiques.
    domain_min = min(min_v, current)
    domain_max = max(max_v, current)
    if domain_max == domain_min:
        # Cas dégénéré : tous les points superposés
        domain_min -= 0.01
        domain_max += 0.01

    margin_x = 30
    margin_y = 10
    plot_w = width - 2 * margin_x
    plot_h = height - 2 * margin_y - 14  # 14px pour le label
    cy = margin_y + plot_h // 2
    box_top = cy - plot_h // 4
    box_bottom = cy + plot_h // 4
    whisker_top = cy - plot_h // 6
    whisker_bottom = cy + plot_h // 6

    def x(v: float) -> float:
        return margin_x + (v - domain_min) / (domain_max - domain_min) * plot_w

    # Le point courant : couleur selon position
    if current < q1:
        point_color = "#3b87d8"  # bleu — plus facile que d'habitude
    elif current > q3:
        point_color = "#d8553b"  # rouge — plus difficile
    else:
        point_color = "#5fa860"  # vert — habituel

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="Distribution de difficulté historique">',
        # Ligne de moustache (min → max)
        f'<line x1="{x(min_v):.1f}" y1="{cy}" x2="{x(max_v):.1f}" '
        f'y2="{cy}" stroke="#999" stroke-width="1"/>',
        # Moustache verticale gauche (min)
        f'<line x1="{x(min_v):.1f}" y1="{whisker_top}" '
        f'x2="{x(min_v):.1f}" y2="{whisker_bottom}" '
        f'stroke="#999" stroke-width="1"/>',
        # Moustache verticale droite (max)
        f'<line x1="{x(max_v):.1f}" y1="{whisker_top}" '
        f'x2="{x(max_v):.1f}" y2="{whisker_bottom}" '
        f'stroke="#999" stroke-width="1"/>',
        # Boîte Q1 → Q3
        f'<rect x="{x(q1):.1f}" y="{box_top}" '
        f'width="{x(q3) - x(q1):.1f}" height="{box_bottom - box_top}" '
        f'fill="#e8e8e8" stroke="#666" stroke-width="1"/>',
        # Médiane
        f'<line x1="{x(median):.1f}" y1="{box_top}" '
        f'x2="{x(median):.1f}" y2="{box_bottom}" '
        f'stroke="#333" stroke-width="2"/>',
        # Point courant (cercle plus grand que les autres marqueurs)
        f'<circle cx="{x(current):.1f}" cy="{cy}" r="6" '
        f'fill="{point_color}" stroke="#000" stroke-width="1"/>',
        # Étiquettes min / max
        f'<text x="{x(min_v):.1f}" y="{height - 2}" '
        f'font-size="10" fill="#666" text-anchor="middle">'
        f'{min_v:.2f}</text>',
        f'<text x="{x(max_v):.1f}" y="{height - 2}" '
        f'font-size="10" fill="#666" text-anchor="middle">'
        f'{max_v:.2f}</text>',
        # Étiquette du point courant
        f'<text x="{x(current):.1f}" y="{margin_y + 8}" '
        f'font-size="11" fill="{point_color}" '
        f'text-anchor="middle" font-weight="600">'
        f'{current:.2f}</text>',
        "</svg>",
    ]
    return "".join(parts)


def build_corpus_difficulty_baseline_html(
    percentile_data: Optional[dict],
    historical_values: Optional[list[float]] = None,
    labels: Optional[dict[str, str]] = None,
) -> str:
    """Construit l'encart « Ce corpus est-il habituel ? ».

    Parameters
    ----------
    percentile_data:
        Sortie de
        ``picarones.measurements.baseline_comparison.compute_corpus_difficulty_percentile``.
        Si ``None``, retourne ``""`` (rapport adaptatif —
        historique trop court ou difficulté absente).
    historical_values:
        Liste des difficultés historiques pour le boxplot.  Si
        ``None`` ou vide, le boxplot est omis et seule la phrase
        factuelle apparaît.
    labels:
        Map i18n.

    Returns
    -------
    str
        HTML de l'encart, ou ``""`` si rien à afficher.
    """
    if not percentile_data:
        return ""
    labels = labels or {}
    title = labels.get(
        "baseline_corpus_title", "Ce corpus est-il habituel ?",
    )
    template_harder = labels.get(
        "baseline_corpus_harder",
        "Difficulté observée {current:.2f} — au {percentile:.0f}ᵉ "
        "percentile des {n_runs} corpus précédents de votre institution. "
        "Ce corpus est plus difficile que la moyenne.",
    )
    template_easier = labels.get(
        "baseline_corpus_easier",
        "Difficulté observée {current:.2f} — au {percentile:.0f}ᵉ "
        "percentile des {n_runs} corpus précédents. Ce corpus est "
        "plus facile que la moyenne.",
    )
    template_usual = labels.get(
        "baseline_corpus_usual",
        "Difficulté observée {current:.2f} — au {percentile:.0f}ᵉ "
        "percentile des {n_runs} corpus précédents. Ce corpus est "
        "dans la moyenne.",
    )

    current = float(percentile_data.get("current_difficulty", 0.0))
    percentile = float(percentile_data.get("percentile", 0.0))
    n_runs = int(percentile_data.get("n_runs", 0))
    if percentile_data.get("harder_than_usual"):
        phrase_template = template_harder
    elif percentile_data.get("easier_than_usual"):
        phrase_template = template_easier
    else:
        phrase_template = template_usual
    phrase = phrase_template.format(
        current=current, percentile=percentile, n_runs=n_runs,
    )

    svg = ""
    if historical_values:
        svg = _build_difficulty_boxplot_svg(
            list(historical_values), current,
        )

    parts = [
        '<div class="baseline-corpus" '
        'style="margin:1rem 0;padding:.75rem;'
        'background:var(--bg-secondary,#f7f7f7);border-radius:6px">',
        f'<div style="font-weight:600;margin-bottom:.4rem">{_e(title)}</div>',
        f'<div style="font-size:.9rem;margin-bottom:.5rem">{_e(phrase)}</div>',
    ]
    if svg:
        parts.append(svg)
    parts.append("</div>")
    return "".join(parts)


__all__ = [
    "build_corpus_difficulty_baseline_html",
]
