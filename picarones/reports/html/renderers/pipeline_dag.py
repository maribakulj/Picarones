"""Visualisation DAG d'un pipeline composé — Sprint 95 (B.4).

Phase 5.C — module relocalisé depuis
``picarones.report.pipeline_dag_render`` vers
``picarones.reports.html.renderers.pipeline_dag``.  Le chemin
legacy reste disponible via un shim avec ``DeprecationWarning`` ;
suppression prévue en 2.0.

Sprint 95 — B.4 du plan d'évolution 2026.

Outil d'inspection, pas de construction
---------------------------------------
Le YAML reste source de vérité.  Cette vue **affiche** le
graphe orienté de la pipeline pour permettre l'inspection et
le debug d'un benchmark d'axe B (Sprint 63+) — elle ne
construit rien, ne supporte pas le drag-and-drop, n'exporte
aucun JSON modifiable.

Pattern identique aux autres rendus : SVG **server-side**,
pas de JS, anti-injection systématique.

Vue
---
Layout horizontal de gauche à droite :

- Chaque **nœud** est un rectangle annoté du nom du module et
  de ses types d'entrée/sortie.
- Chaque **arête** porte une étiquette : type d'artefact +
  métrique principale + valeur, avec un code couleur
  vert/jaune/rouge selon le seuil sur la valeur.

Adaptive : ``""`` si moins d'un nœud.

Note d'intégration
------------------
Module pur — l'utilisateur compose les structures simples
``nodes`` et ``edges`` depuis sa ``PipelineSpec`` (Sprint 63)
et son ``PipelineBenchmarkResult`` (Sprint 64) :

.. code-block:: python

    from picarones.reports.html.renderers.pipeline_dag import build_pipeline_dag_html

    nodes = [
        {"name": s.name, "input_types": [t.value for t in s.module.input_types],
         "output_types": [t.value for t in s.module.output_types]}
        for s in spec.steps
    ]
    edges = []
    for prev, curr in zip(spec.steps, spec.steps[1:]):
        agg = bench.aggregate_for_step(curr.name)
        for art_type, metrics in (agg.junction_metrics or {}).items():
            for metric_name, value in metrics.items():
                edges.append({
                    "from": prev.name, "to": curr.name,
                    "artifact_type": art_type, "metric_name": metric_name,
                    "metric_value": value.get("mean"),
                })
    html = build_pipeline_dag_html(nodes, edges, labels)
"""

from __future__ import annotations

from html import escape as _e
from typing import Optional


# Seuils par défaut sur les métriques d'erreur (CER-like, lower is better).
_DEFAULT_THRESHOLDS = (0.05, 0.15)  # vert ≤ 0.05, jaune ≤ 0.15, rouge > 0.15


def _classify_metric(
    value: Optional[float],
    thresholds: tuple[float, float],
    higher_is_better: bool,
) -> str:
    """Retourne ``"green"``, ``"yellow"``, ``"red"`` ou ``"none"``."""
    if value is None:
        return "none"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "none"
    low, high = thresholds
    if higher_is_better:
        # Inversion : haut = bon
        if v >= 1.0 - low:
            return "green"
        if v >= 1.0 - high:
            return "yellow"
        return "red"
    if v <= low:
        return "green"
    if v <= high:
        return "yellow"
    return "red"


# Sprint A7 (m-5) — palette Okabe-Ito daltonien-friendly importée
# depuis le module canonique ``picarones.report.colors``. Avant
# A7, les hex étaient hardcodés (rouge/vert classiques, problème
# pour la deutéranopie) ; maintenant cohérent avec _cer_color et
# difficulty_color.
from picarones.reports._helpers.colors import COLOR_GREEN, COLOR_RED, COLOR_YELLOW

_QUALITY_COLORS = {
    "green":  COLOR_GREEN,    # Okabe-Ito blue (substitut sémantique « bon »)
    "yellow": COLOR_YELLOW,   # Okabe-Ito yellow
    "red":    COLOR_RED,      # Okabe-Ito vermillion (substitut sémantique « mauvais »)
    "none":   "#6b7280",
}


def _format_value(value: Optional[float]) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if abs(v) < 1.0:
        return f"{v * 100:.1f}%"
    return f"{v:.2f}"


def build_pipeline_dag_html(
    nodes: Optional[list[dict]],
    labels: Optional[dict[str, str]] = None,
    edges: Optional[list[dict]] = None,
    *,
    thresholds: tuple[float, float] = _DEFAULT_THRESHOLDS,
    higher_is_better: bool = False,
) -> str:
    """Construit la vue HTML « Pipeline DAG ».

    Parameters
    ----------
    nodes:
        Liste de dicts ``{"name", "input_types"?, "output_types"?}``
        dans l'ordre topologique.  Si vide ou ``None``, retourne
        ``""``.
    labels:
        Dict i18n.  Clés sous le préfixe ``dag_*``.
    edges:
        Liste de dicts ``{"from", "to", "artifact_type"?,
        "metric_name"?, "metric_value"?}``.  Optionnel —
        auto-déduit séquentiel sinon.
    thresholds:
        ``(seuil_vert, seuil_jaune)`` sur la valeur de métrique.
        Défaut ``(0.05, 0.15)`` — convention CER.
    higher_is_better:
        Si ``True``, la sémantique est inversée (1 = meilleur).
    """
    nodes = list(nodes or [])
    if not nodes:
        return ""
    edges = list(edges or [])
    labels = labels or {}
    title = labels.get("dag_title", "Pipeline DAG")
    note = labels.get(
        "dag_note",
        "Graphe orienté du pipeline composé. Chaque arête porte "
        "le type d'artefact transmis et la métrique calculée à "
        "la jonction. Code couleur vert/orange/rouge selon le "
        "seuil. Outil d'inspection — le YAML reste source de "
        "vérité.",
    )
    # Layout horizontal régulier
    n = len(nodes)
    box_width = 160
    box_height = 70
    h_gap = 110          # espace horizontal entre nœuds
    margin = 30
    svg_width = margin * 2 + n * box_width + (n - 1) * h_gap
    svg_height = box_height + margin * 2 + 60  # +60 pour étiquettes arêtes
    centre_y = margin + box_height / 2 + 30  # offset pour étiquette de tête

    # Index des nœuds par name pour récupérer la position
    node_x: dict[str, float] = {}
    parts: list[str] = [
        '<section class="dag-section" style="margin:1rem 0">',
        f'<h3 style="margin:0 0 .3rem 0">{_e(title)}</h3>',
        f'<div style="font-size:.85rem;opacity:.75;margin-bottom:.5rem">'
        f'{_e(note)}</div>',
        f'<svg viewBox="0 0 {svg_width} {svg_height}" '
        f'role="img" aria-label="{_e(title)}" '
        'xmlns="http://www.w3.org/2000/svg" '
        'style="max-width:100%;height:auto;'
        'font-family:system-ui,sans-serif;font-size:12px">',
        # Définition d'une flèche
        '<defs>'
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M0,0 L10,5 L0,10 z" fill="#374151"/>'
        '</marker>'
        '</defs>',
    ]

    # Étape 1 : nœuds
    for i, node in enumerate(nodes):
        name = str(node.get("name") or f"step_{i}")
        x = margin + i * (box_width + h_gap)
        y = margin + 30
        node_x[name] = x + box_width
        in_types = ", ".join(node.get("input_types") or [])
        out_types = ", ".join(node.get("output_types") or [])
        parts.append(
            f'<rect x="{x}" y="{y}" width="{box_width}" '
            f'height="{box_height}" rx="6" fill="#f3f4f6" '
            f'stroke="#374151" stroke-width="1.5"/>'
        )
        parts.append(
            f'<text x="{x + box_width / 2}" y="{y + 22}" '
            f'text-anchor="middle" font-weight="600" '
            f'fill="#111827">{_e(name)}</text>'
        )
        if in_types:
            parts.append(
                f'<text x="{x + box_width / 2}" y="{y + 40}" '
                f'text-anchor="middle" fill="#4b5563" '
                f'font-size="10">in: {_e(in_types)}</text>'
            )
        if out_types:
            parts.append(
                f'<text x="{x + box_width / 2}" y="{y + 56}" '
                f'text-anchor="middle" fill="#4b5563" '
                f'font-size="10">out: {_e(out_types)}</text>'
            )

    # Étape 2 : arêtes (mappées sur paires séquentielles si pas de
    # "from"/"to" explicites — voir nodes par défaut)
    auto_edges: list[dict] = []
    if not edges:
        for prev, curr in zip(nodes, nodes[1:]):
            auto_edges.append({
                "from": prev.get("name"),
                "to": curr.get("name"),
            })
    else:
        auto_edges = edges

    for edge in auto_edges:
        src = str(edge.get("from") or "")
        dst = str(edge.get("to") or "")
        if not src or not dst:
            continue
        # Position : du bord droit du src au bord gauche du dst
        # Heuristique : on prend la position du nœud src dans la
        # liste pour calculer x1, et celle de dst pour x2.
        try:
            i_src = next(
                i for i, n_ in enumerate(nodes)
                if n_.get("name") == src
            )
            i_dst = next(
                i for i, n_ in enumerate(nodes)
                if n_.get("name") == dst
            )
        except StopIteration:
            continue
        x1 = margin + i_src * (box_width + h_gap) + box_width
        x2 = margin + i_dst * (box_width + h_gap)
        y = centre_y
        # Classe la métrique pour le code couleur
        value = edge.get("metric_value")
        try:
            value_f = float(value) if value is not None else None
        except (TypeError, ValueError):
            value_f = None
        cls = _classify_metric(value_f, thresholds, higher_is_better)
        color = _QUALITY_COLORS[cls]
        # Trace la flèche
        parts.append(
            f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" '
            f'stroke="{color}" stroke-width="2" '
            f'marker-end="url(#arrow)"/>'
        )
        # Étiquette : type + métrique : valeur
        artifact_type = edge.get("artifact_type") or ""
        metric_name = edge.get("metric_name") or ""
        value_str = _format_value(value_f)
        label_lines: list[str] = []
        if artifact_type:
            label_lines.append(str(artifact_type))
        if metric_name:
            label_lines.append(f"{metric_name}: {value_str}")
        if label_lines:
            label_x = (x1 + x2) / 2
            for k, line in enumerate(label_lines):
                parts.append(
                    f'<text x="{label_x}" y="{y - 8 - k * 12}" '
                    f'text-anchor="middle" fill="{color}" '
                    f'font-size="10" font-weight="600">'
                    f'{_e(line)}</text>'
                )
    parts.append("</svg>")

    # Légende
    h_legend = labels.get("dag_legend", "Lecture")
    legend_green = labels.get("dag_legend_green", "qualité élevée")
    legend_yellow = labels.get("dag_legend_yellow", "qualité moyenne")
    legend_red = labels.get("dag_legend_red", "qualité faible")
    parts.append(
        '<div style="font-size:.8rem;opacity:.75;margin-top:.4rem">'
        f'<strong>{_e(h_legend)} :</strong> '
        f'<span style="color:{_QUALITY_COLORS["green"]};'
        f'font-weight:600">●</span> {_e(legend_green)} '
        f'(≤ {thresholds[0] * 100:.0f}%) '
        f'<span style="color:{_QUALITY_COLORS["yellow"]};'
        f'font-weight:600">●</span> {_e(legend_yellow)} '
        f'(≤ {thresholds[1] * 100:.0f}%) '
        f'<span style="color:{_QUALITY_COLORS["red"]};'
        f'font-weight:600">●</span> {_e(legend_red)}'
        '</div>'
    )
    parts.append("</section>")
    return "".join(parts)


__all__ = ["build_pipeline_dag_html"]
