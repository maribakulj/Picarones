"""Rendu HTML serveur-side de la section inter-moteurs (Sprint 37).

Suite des Sprints 35-36 : la couche de calcul (`inter_engine.py`) et le
câblage runner+narratif sont en place.  Ce module produit les blocs HTML
qui remontent ces données dans le rapport :

- ``build_divergence_matrix_html`` — table HTML colorée (heatmap CSS
  inline) qui montre la divergence taxonomique entre paires de moteurs.
- ``build_oracle_gap_html`` — encart factuel avec oracle_recall,
  best_single_recall, gap absolu/relatif, top paires divergentes.

Principe — cohérent avec le CDD du Sprint 18 : rendu serveur-side, pas
de JavaScript, pas de dépendance Chart.js, déterministe.  Si
``inter_engine_analysis`` est ``None`` ou incomplet, les fonctions
retournent une chaîne vide — le template Jinja2 masque la section
silencieusement (principe du rapport adaptatif).
"""

from __future__ import annotations

from html import escape as _e
from typing import Optional

from picarones.reports_v2._helpers.render_helpers import (
    GRADIENT_TARGET_RED,
    color_single_gradient,
)


def build_divergence_matrix_html(
    inter_engine_analysis: Optional[dict],
    labels: Optional[dict[str, str]] = None,
) -> str:
    """Construit une table HTML colorée représentant la matrice de divergence
    taxonomique inter-moteurs.

    Parameters
    ----------
    inter_engine_analysis:
        Dict produit par ``compute_inter_engine_analysis`` (Sprint 36).
        ``None`` ou sans ``taxonomy_divergence`` → chaîne vide.
    labels:
        Dict d'étiquettes i18n.  Clés utilisées :
        ``"divergence_caption"``, ``"divergence_metric_label"``,
        ``"divergence_max_pair_label"``, ``"divergence_diagonal_label"``.
        Fallback FR si manquantes.

    Returns
    -------
    str
        HTML complet (``<div class="...">…</div>``) ou ``""`` si pas de
        données disponibles.
    """
    if not inter_engine_analysis:
        return ""
    div = inter_engine_analysis.get("taxonomy_divergence")
    if not div or not div.get("matrix"):
        return ""

    matrix: dict[str, dict[str, float]] = div["matrix"]
    metric: str = str(div.get("metric") or "js")
    engines = sorted(matrix.keys())
    if len(engines) < 2:
        return ""

    labels = labels or {}
    caption = labels.get(
        "divergence_caption",
        "Divergence taxonomique entre moteurs",
    )
    metric_label = labels.get(
        "divergence_metric_label",
        "Métrique",
    )
    diag_label = labels.get(
        "divergence_diagonal_label",
        "(identité)",
    )

    # vmax = max hors diagonale (la diagonale est toujours 0)
    off_diag_values = [
        matrix[a][b] for a in engines for b in engines if a != b
    ]
    vmax = max(off_diag_values) if off_diag_values else 0.0

    parts: list[str] = []
    parts.append('<div class="inter-engine-divergence">')
    parts.append(
        f'<div class="inter-engine-caption" style="font-weight:600;margin-bottom:.4rem">'
        f'{_e(caption)} <span style="font-weight:400;color:var(--text-muted)">'
        f'({_e(metric_label)} : {_e(metric)})</span></div>'
    )
    parts.append('<table class="divergence-matrix" style="border-collapse:collapse;font-size:.8rem">')
    # En-tête
    parts.append("<thead><tr><th scope=\"col\"></th>")
    for b in engines:
        parts.append(
            f'<th scope=\"col\" style="padding:.3rem .5rem;text-align:center;'
            f'border-bottom:1px solid var(--border)">{_e(b)}</th>'
        )
    parts.append("</tr></thead>")
    # Lignes
    parts.append("<tbody>")
    for a in engines:
        parts.append("<tr>")
        parts.append(
            f'<th scope=\"col\" style="padding:.3rem .5rem;text-align:right;'
            f'border-right:1px solid var(--border);font-weight:600">{_e(a)}</th>'
        )
        for b in engines:
            v = float(matrix[a].get(b, 0.0))
            if a == b:
                parts.append(
                    f'<td style="padding:.3rem .5rem;text-align:center;'
                    f'background:#f4f4f4;color:var(--text-muted);'
                    f'font-style:italic">{_e(diag_label)}</td>'
                )
            else:
                bg = (
                    color_single_gradient(v, end_rgb=GRADIENT_TARGET_RED, max_value=vmax)
                    if vmax > 0 else "#ffffff"
                )
                # Texte sombre toujours lisible (pas de seuil fort sur le rouge clair).
                parts.append(
                    f'<td style="padding:.3rem .5rem;text-align:center;'
                    f'background:{bg};color:#222;font-variant-numeric:tabular-nums">'
                    f'{v:.3f}</td>'
                )
        parts.append("</tr>")
    parts.append("</tbody></table>")

    # Ligne « paire la plus divergente », si disponible et > 0
    max_pair = div.get("max_pair")
    if (
        max_pair
        and len(max_pair) >= 3
        and isinstance(max_pair[2], (int, float))
        and float(max_pair[2]) > 0
    ):
        max_pair_label = labels.get(
            "divergence_max_pair_label",
            "Paire la plus divergente",
        )
        parts.append(
            f'<div style="margin-top:.5rem;font-size:.8rem;color:var(--text-muted)">'
            f'{_e(max_pair_label)} : '
            f'<strong>{_e(str(max_pair[0]))}</strong> ↔ '
            f'<strong>{_e(str(max_pair[1]))}</strong> '
            f'({_e(metric)} = {float(max_pair[2]):.3f})</div>'
        )

    parts.append("</div>")
    return "".join(parts)


def build_oracle_gap_html(
    inter_engine_analysis: Optional[dict],
    labels: Optional[dict[str, str]] = None,
) -> str:
    """Construit l'encart factuel sur la complémentarité (oracle gap).

    Parameters
    ----------
    inter_engine_analysis:
        Dict produit par ``compute_inter_engine_analysis``.  ``None`` ou
        sans ``complementarity`` → chaîne vide.
    labels:
        Dict d'étiquettes i18n.  Clés :
        ``"oracle_caption"``, ``"oracle_best_engine"``,
        ``"oracle_best_recall"``, ``"oracle_recall"``, ``"oracle_gap"``,
        ``"oracle_doc_count"``, ``"oracle_explanation"``.

    Returns
    -------
    str
        HTML ``<div>...</div>`` ou ``""`` si pas de données.
    """
    if not inter_engine_analysis:
        return ""
    comp = inter_engine_analysis.get("complementarity")
    if not comp:
        return ""

    labels = labels or {}
    caption = labels.get(
        "oracle_caption",
        "Complémentarité — gain potentiel d'un voting majoritaire",
    )
    best_engine_label = labels.get("oracle_best_engine", "Meilleur moteur seul")
    best_recall_label = labels.get("oracle_best_recall", "Tokens préservés")
    oracle_label = labels.get("oracle_recall", "Oracle (au moins un moteur)")
    gap_label = labels.get("oracle_gap", "Gain potentiel d'un ensemble")
    doc_count_label = labels.get("oracle_doc_count", "Documents évalués")
    explanation = labels.get(
        "oracle_explanation",
        "L'oracle est la borne supérieure du recall token-level atteignable "
        "par un voting majoritaire entre les moteurs (proxy bag-of-words).",
    )

    best_engine = str(comp.get("best_engine") or "")
    best_recall_pct = round(float(comp.get("best_single_recall") or 0.0) * 100, 1)
    oracle_recall_pct = round(float(comp.get("oracle_recall") or 0.0) * 100, 1)
    absolute_gap_pct = round(float(comp.get("absolute_gap") or 0.0) * 100, 1)
    relative_gap_pct = round(float(comp.get("relative_gap") or 0.0) * 100, 1)
    doc_count = int(comp.get("doc_count") or 0)

    parts: list[str] = []
    parts.append('<div class="inter-engine-oracle">')
    parts.append(
        f'<div class="inter-engine-caption" style="font-weight:600;margin-bottom:.4rem">'
        f'{_e(caption)}</div>'
    )
    parts.append('<dl style="display:grid;grid-template-columns:auto 1fr;gap:.25rem .8rem;font-size:.85rem;margin:0">')

    def _row(label: str, value: str) -> str:
        return (
            f'<dt style="color:var(--text-muted)">{_e(label)}</dt>'
            f'<dd style="margin:0;font-variant-numeric:tabular-nums">{_e(value)}</dd>'
        )

    parts.append(_row(best_engine_label, best_engine or "—"))
    parts.append(_row(best_recall_label, f"{best_recall_pct} %"))
    parts.append(_row(oracle_label, f"{oracle_recall_pct} %"))
    parts.append(
        _row(
            gap_label,
            f"+{absolute_gap_pct} pts ({relative_gap_pct} % "
            + labels.get("oracle_recoverable", "récupérable")
            + ")",
        )
    )
    parts.append(_row(doc_count_label, str(doc_count)))
    parts.append("</dl>")
    parts.append(
        f'<div style="margin-top:.5rem;font-size:.72rem;color:var(--text-muted)">'
        f'{_e(explanation)}</div>'
    )
    parts.append("</div>")
    return "".join(parts)


__all__ = [
    "build_divergence_matrix_html",
    "build_oracle_gap_html",
]
