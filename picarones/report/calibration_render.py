"""Rendu HTML server-side de la section calibration (Sprint 43).

Suite directe des Sprints 39+42 : la couche de calcul (ECE, MCE,
reliability_diagram) et le câblage runner sont en place ; ce module
produit les blocs HTML qui rendent ces données visibles dans le
rapport.

- ``build_calibration_summary_html`` — tableau résumé par moteur :
  ECE, MCE, nombre de prédictions évaluées, accuracy moyenne,
  confidence moyenne.  Cellule ECE colorée par gradient vert (bien
  calibré) → rouge (mal calibré).
- ``build_reliability_diagram_svg`` — SVG d'un reliability diagram
  pour un moteur donné : barres d'accuracy par bin, ligne idéale
  (calibration parfaite) en diagonale, axes annotés.

Principe — cohérent avec le SVG du CDD (Sprint 18) et les renderers
Sprint 37/41 : strictement server-side, déterministe, pas de
JavaScript.  Si aucun moteur n'a de ``aggregated_calibration``, le
masquage adaptatif fait que les fonctions retournent ``""`` et la
section est silencieusement omise.

Anti-injection : tous les noms de moteurs et étiquettes sont passés à
``html.escape`` avant insertion.
"""

from __future__ import annotations

from html import escape as _e
from typing import Optional


def _color_for_ece(ece: float) -> str:
    """Gradient vert (ECE = 0, bien calibré) → rouge (ECE = 0.5+)."""
    f = max(0.0, min(1.0, ece * 2.0))  # ECE > 0.5 → rouge max
    if f <= 0.5:
        ratio = f / 0.5
        r = int(130 + (240 - 130) * ratio)
        g = int(200 + (220 - 200) * ratio)
        b = int(130 + (130 - 130) * ratio)
    else:
        ratio = (f - 0.5) / 0.5
        r = int(240 + (220 - 240) * ratio)
        g = int(220 + (100 - 220) * ratio)
        b = int(130 + (100 - 130) * ratio)
    return f"#{r:02x}{g:02x}{b:02x}"


def _engines_with_calibration(engines_summary: list[dict]) -> list[dict]:
    return [e for e in engines_summary if e.get("aggregated_calibration")]


def build_calibration_summary_html(
    engines_summary: list[dict],
    labels: Optional[dict[str, str]] = None,
) -> str:
    """Tableau résumé : ECE/MCE/N par moteur."""
    relevant = _engines_with_calibration(engines_summary)
    if not relevant:
        return ""

    labels = labels or {}
    caption = labels.get(
        "calibration_summary_caption",
        "Calibration des moteurs (ECE, MCE)",
    )
    engine_label = labels.get("calibration_engine_label", "Moteur")
    ece_label = labels.get("calibration_ece_label", "ECE")
    mce_label = labels.get("calibration_mce_label", "MCE")
    n_label = labels.get("calibration_n_label", "Prédictions")
    acc_label = labels.get("calibration_acc_label", "Précision moyenne")
    conf_label = labels.get("calibration_conf_label", "Confiance moyenne")
    docs_label = labels.get("calibration_docs_label", "Docs évalués")

    parts: list[str] = []
    parts.append('<div class="calibration-summary">')
    parts.append(
        f'<div class="calibration-summary-caption" '
        f'style="font-weight:600;margin-bottom:.4rem">{_e(caption)}</div>'
    )
    parts.append(
        '<table class="calibration-summary-table" '
        'style="border-collapse:collapse;font-size:.85rem;width:100%">'
    )
    parts.append("<thead><tr>")
    for hdr in (engine_label, ece_label, mce_label,
                acc_label, conf_label, n_label, docs_label):
        parts.append(
            f'<th style="padding:.3rem .5rem;text-align:left;'
            f'border-bottom:1px solid var(--border);font-weight:600">'
            f'{_e(hdr)}</th>'
        )
    parts.append("</tr></thead><tbody>")
    for engine in relevant:
        agg = engine["aggregated_calibration"]
        ece = float(agg.get("ece") or 0.0)
        mce = float(agg.get("mce") or 0.0)
        n_pred = int(agg.get("n_predictions") or 0)
        acc = float(agg.get("overall_accuracy") or 0.0)
        conf = float(agg.get("overall_confidence") or 0.0)
        doc_count = int(agg.get("doc_count") or 0)
        bg = _color_for_ece(ece)
        parts.append("<tr>")
        parts.append(
            f'<td style="padding:.3rem .5rem;font-weight:600">'
            f'{_e(engine.get("name", ""))}</td>'
        )
        parts.append(
            f'<td style="padding:.3rem .5rem;background:{bg};'
            f'font-variant-numeric:tabular-nums">{ece * 100:.2f} %</td>'
        )
        parts.append(
            f'<td style="padding:.3rem .5rem;font-variant-numeric:tabular-nums">'
            f'{mce * 100:.2f} %</td>'
        )
        parts.append(
            f'<td style="padding:.3rem .5rem;font-variant-numeric:tabular-nums">'
            f'{acc * 100:.1f} %</td>'
        )
        parts.append(
            f'<td style="padding:.3rem .5rem;font-variant-numeric:tabular-nums">'
            f'{conf * 100:.1f} %</td>'
        )
        parts.append(
            f'<td style="padding:.3rem .5rem;font-variant-numeric:tabular-nums">'
            f'{n_pred:,}</td>'.replace(",", " ")
        )
        parts.append(
            f'<td style="padding:.3rem .5rem;font-variant-numeric:tabular-nums">'
            f'{doc_count}</td>'
        )
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


# ──────────────────────────────────────────────────────────────────────────
# SVG reliability diagram
# ──────────────────────────────────────────────────────────────────────────


# Géométrie SVG (en unités viewBox)
_SVG_W = 240
_SVG_H = 240
_PAD_LEFT = 38
_PAD_RIGHT = 8
_PAD_TOP = 8
_PAD_BOTTOM = 30


def _svg_x(value: float) -> float:
    """Mappe une confidence ∈ [0, 1] sur l'axe x du SVG."""
    return _PAD_LEFT + value * (_SVG_W - _PAD_LEFT - _PAD_RIGHT)


def _svg_y(value: float) -> float:
    """Mappe une accuracy ∈ [0, 1] sur l'axe y (inversé : 0 en bas)."""
    return _SVG_H - _PAD_BOTTOM - value * (_SVG_H - _PAD_TOP - _PAD_BOTTOM)


def build_reliability_diagram_svg(
    aggregated_calibration: Optional[dict],
    labels: Optional[dict[str, str]] = None,
    *,
    engine_name: str = "",
) -> str:
    """Construit un SVG du reliability diagram pour un moteur.

    Conventions
    -----------
    - Axe x : confidence moyenne par bin ∈ [0, 1]
    - Axe y : accuracy par bin ∈ [0, 1]
    - Diagonale en pointillé : calibration parfaite (référence)
    - Barre par bin : largeur = bin_width, hauteur = accuracy
    - Cercle par bin : (avg_confidence, accuracy) — position réelle

    Returns
    -------
    str
        SVG complet, ou ``""`` si pas de bin non vide.
    """
    if not aggregated_calibration:
        return ""
    bins = aggregated_calibration.get("bins") or []
    non_empty = [b for b in bins if (b.get("count") or 0) > 0]
    if not non_empty:
        return ""

    labels = labels or {}
    title = labels.get(
        "reliability_diagram_title", "Diagramme de fiabilité",
    )
    conf_axis = labels.get("reliability_x_axis", "Confiance")
    acc_axis = labels.get("reliability_y_axis", "Précision")

    parts: list[str] = []
    parts.append(
        f'<svg viewBox="0 0 {_SVG_W} {_SVG_H}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-label="{_e(title)} — {_e(engine_name)}" '
        f'style="width:100%;max-width:300px;height:auto;'
        f'background:#fff;border:1px solid var(--border)">'
    )

    # Axes
    parts.append(
        f'<line x1="{_PAD_LEFT}" y1="{_SVG_H - _PAD_BOTTOM}" '
        f'x2="{_SVG_W - _PAD_RIGHT}" y2="{_SVG_H - _PAD_BOTTOM}" '
        f'stroke="#333" stroke-width="1"/>'
    )
    parts.append(
        f'<line x1="{_PAD_LEFT}" y1="{_PAD_TOP}" '
        f'x2="{_PAD_LEFT}" y2="{_SVG_H - _PAD_BOTTOM}" '
        f'stroke="#333" stroke-width="1"/>'
    )

    # Diagonale (calibration parfaite)
    parts.append(
        f'<line x1="{_svg_x(0)}" y1="{_svg_y(0)}" '
        f'x2="{_svg_x(1)}" y2="{_svg_y(1)}" '
        f'stroke="#888" stroke-width="1" stroke-dasharray="4 3"/>'
    )

    # Barres par bin
    for b in bins:
        n = int(b.get("count") or 0)
        if n == 0:
            continue
        bin_low = float(b.get("bin_low", 0.0))
        bin_high = float(b.get("bin_high", 1.0))
        accuracy = float(b.get("accuracy") or 0.0)
        x = _svg_x(bin_low)
        w = _svg_x(bin_high) - x
        y = _svg_y(accuracy)
        h = _svg_y(0) - y
        # Barre semi-transparente pour ne pas masquer la diagonale
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" '
            f'width="{w:.2f}" height="{h:.2f}" '
            f'fill="#4b8bbe" fill-opacity="0.35" '
            f'stroke="#4b8bbe" stroke-width="0.5"/>'
        )

    # Points (avg_confidence, accuracy) reliés par une ligne
    points = []
    for b in bins:
        n = int(b.get("count") or 0)
        if n == 0:
            continue
        avg_conf = float(b.get("avg_confidence") or 0.0)
        accuracy = float(b.get("accuracy") or 0.0)
        points.append((_svg_x(avg_conf), _svg_y(accuracy)))
    if len(points) >= 2:
        path_d = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        parts.append(
            f'<path d="{path_d}" fill="none" '
            f'stroke="#c8553d" stroke-width="1.5"/>'
        )
    for x, y in points:
        parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.5" '
            f'fill="#c8553d"/>'
        )

    # Étiquettes d'axes
    parts.append(
        f'<text x="{(_PAD_LEFT + _SVG_W - _PAD_RIGHT) / 2}" '
        f'y="{_SVG_H - 6}" text-anchor="middle" '
        f'font-size="10" fill="#444">{_e(conf_axis)}</text>'
    )
    parts.append(
        f'<text x="10" y="{(_PAD_TOP + _SVG_H - _PAD_BOTTOM) / 2}" '
        f'text-anchor="middle" font-size="10" fill="#444" '
        f'transform="rotate(-90 10 {(_PAD_TOP + _SVG_H - _PAD_BOTTOM) / 2})">'
        f'{_e(acc_axis)}</text>'
    )

    # Graduations 0 / 0.5 / 1 sur les deux axes
    for v in (0.0, 0.5, 1.0):
        parts.append(
            f'<text x="{_svg_x(v):.1f}" y="{_SVG_H - _PAD_BOTTOM + 12}" '
            f'text-anchor="middle" font-size="9" fill="#666">{v:.1f}</text>'
        )
        parts.append(
            f'<text x="{_PAD_LEFT - 6}" y="{_svg_y(v) + 3:.1f}" '
            f'text-anchor="end" font-size="9" fill="#666">{v:.1f}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def build_reliability_diagrams_grid_html(
    engines_summary: list[dict],
    labels: Optional[dict[str, str]] = None,
) -> str:
    """Construit une grille de reliability diagrams (un par moteur).

    Layout : grid auto-fit, chaque cellule a son SVG + le nom du moteur
    en titre.  Vide si aucun moteur n'a d'``aggregated_calibration``.
    """
    relevant = _engines_with_calibration(engines_summary)
    if not relevant:
        return ""

    parts: list[str] = []
    parts.append(
        '<div class="reliability-diagrams-grid" '
        'style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));'
        'gap:1rem;margin-top:.6rem">'
    )
    for engine in relevant:
        name = engine.get("name", "")
        svg = build_reliability_diagram_svg(
            engine["aggregated_calibration"],
            labels=labels,
            engine_name=name,
        )
        if not svg:
            continue
        parts.append('<div class="reliability-diagram-card">')
        parts.append(
            f'<div style="font-weight:600;font-size:.85rem;'
            f'margin-bottom:.3rem;text-align:center">{_e(name)}</div>'
        )
        parts.append(svg)
        parts.append("</div>")
    parts.append("</div>")
    return "".join(parts)


__all__ = [
    "build_calibration_summary_html",
    "build_reliability_diagram_svg",
    "build_reliability_diagrams_grid_html",
]
