"""Rendu HTML « Profil d'image du corpus » — Sprint 93 (A.II.7).

Phase 5.C — module relocalisé depuis
``picarones.report.image_predictive_render`` vers
``picarones.reports_v2.html.renderers.image_predictive``.  Le chemin
legacy reste disponible via un shim avec ``DeprecationWarning`` ;
suppression prévue en 2.0.

Suite directe ``picarones/core/image_predictive.py``.  Pattern
identique aux autres rendus : server-side, pas de JS, anti-
injection systématique.

Vue
---
Deux blocs dans une section unique :

1. **Complexité paléographique** : moyenne, médiane, min, max,
   écart-type sur l'ensemble du corpus.
2. **Homogénéité du corpus** : score combiné + détail par
   feature (mean, stdev, contribution normalisée).

Adaptive : ``""`` si pas de données.

Note d'intégration
------------------
Module pur — l'utilisateur compose :

.. code-block:: python

    from picarones.measurements.image_predictive import aggregate_corpus_predictive
    from picarones.reports_v2.html.renderers.image_predictive import (
        build_image_predictive_html,
    )

    qualities = [doc.image_quality.as_dict() for doc in benchmark.docs]
    agg = aggregate_corpus_predictive(qualities)
    html = build_image_predictive_html(agg, labels)
"""

from __future__ import annotations

from html import escape as _e
from typing import Optional

from picarones.reports_v2._helpers.render_helpers import color_traffic_light


_FEATURE_LABEL_KEYS = {
    "noise_level": "imgpred_feat_noise",
    "sharpness_score": "imgpred_feat_sharpness",
    "contrast_score": "imgpred_feat_contrast",
    "rotation_degrees": "imgpred_feat_rotation",
}


def _render_complexity_block(
    aggregated: dict, labels: dict[str, str],
) -> str:
    h_complex = labels.get(
        "imgpred_complexity", "Complexité paléographique",
    )
    h_mean = labels.get("imgpred_mean", "Moyenne")
    h_median = labels.get("imgpred_median", "Médiane")
    h_min = labels.get("imgpred_min", "Min")
    h_max = labels.get("imgpred_max", "Max")
    h_stdev = labels.get("imgpred_stdev", "Écart-type")
    h_docs = labels.get("imgpred_docs", "Docs")
    mean = float(aggregated.get("complexity_mean") or 0.0)
    median = float(aggregated.get("complexity_median") or 0.0)
    mn = float(aggregated.get("complexity_min") or 0.0)
    mx = float(aggregated.get("complexity_max") or 0.0)
    sd = float(aggregated.get("complexity_stdev") or 0.0)
    n_docs = int(aggregated.get("n_docs") or 0)
    color_mean = color_traffic_light(mean, low_is_good=True)
    return (
        f'<div style="font-weight:600;margin:.4rem 0 .3rem 0">'
        f'{_e(h_complex)}</div>'
        '<table style="border-collapse:collapse;width:100%;'
        'font-size:.9rem;margin-bottom:.8rem">'
        f'<thead><tr>'
        f'<th scope=\"col\" style="padding:.4rem .6rem;text-align:right;'
        f'border-bottom:1px solid #ccc;font-weight:600">{_e(h_mean)}</th>'
        f'<th scope=\"col\" style="padding:.4rem .6rem;text-align:right;'
        f'border-bottom:1px solid #ccc;font-weight:600">{_e(h_median)}</th>'
        f'<th scope=\"col\" style="padding:.4rem .6rem;text-align:right;'
        f'border-bottom:1px solid #ccc;font-weight:600">{_e(h_min)}</th>'
        f'<th scope=\"col\" style="padding:.4rem .6rem;text-align:right;'
        f'border-bottom:1px solid #ccc;font-weight:600">{_e(h_max)}</th>'
        f'<th scope=\"col\" style="padding:.4rem .6rem;text-align:right;'
        f'border-bottom:1px solid #ccc;font-weight:600">{_e(h_stdev)}</th>'
        f'<th scope=\"col\" style="padding:.4rem .6rem;text-align:right;'
        f'border-bottom:1px solid #ccc;font-weight:600">{_e(h_docs)}</th>'
        f'</tr></thead>'
        f'<tbody><tr>'
        f'<td style="padding:.4rem .6rem;text-align:right;'
        f'background:{color_mean};font-family:monospace;font-weight:600">'
        f'{mean:.3f}</td>'
        f'<td style="padding:.4rem .6rem;text-align:right;'
        f'font-family:monospace">{median:.3f}</td>'
        f'<td style="padding:.4rem .6rem;text-align:right;'
        f'font-family:monospace">{mn:.3f}</td>'
        f'<td style="padding:.4rem .6rem;text-align:right;'
        f'font-family:monospace">{mx:.3f}</td>'
        f'<td style="padding:.4rem .6rem;text-align:right;'
        f'font-family:monospace">{sd:.3f}</td>'
        f'<td style="padding:.4rem .6rem;text-align:right;'
        f'font-family:monospace">{n_docs}</td>'
        f'</tr></tbody></table>'
    )


def _render_homogeneity_block(
    homogeneity: dict, labels: dict[str, str],
) -> str:
    h_homo = labels.get(
        "imgpred_homogeneity", "Homogénéité du corpus",
    )
    h_feat = labels.get("imgpred_feature", "Feature")
    h_mean = labels.get("imgpred_feat_mean", "Moyenne")
    h_stdev = labels.get("imgpred_feat_stdev", "Écart-type")
    h_norm = labels.get(
        "imgpred_feat_norm", "Contribution normalisée",
    )
    score = float(homogeneity.get("score") or 0.0)
    color = color_traffic_light(score, low_is_good=True)
    parts = [
        f'<div style="font-weight:600;margin:.4rem 0 .3rem 0">'
        f'{_e(h_homo)} : '
        f'<span style="background:{color};padding:.1rem .4rem;'
        f'border-radius:.3rem;font-family:monospace">{score:.3f}</span>'
        f'</div>',
        '<table style="border-collapse:collapse;width:100%;'
        'font-size:.9rem">',
        '<thead><tr>',
    ]
    for col in (h_feat, h_mean, h_stdev, h_norm):
        parts.append(
            f'<th scope=\"col\" style="padding:.4rem .6rem;text-align:left;'
            f'border-bottom:1px solid #ccc;font-weight:600">'
            f'{_e(col)}</th>'
        )
    parts.append("</tr></thead><tbody>")
    per_feat = homogeneity.get("per_feature") or {}
    for key, label_key in _FEATURE_LABEL_KEYS.items():
        if key not in per_feat:
            continue
        slot = per_feat[key]
        feat_label = labels.get(label_key, key)
        feat_mean = float(slot.get("mean") or 0.0)
        feat_stdev = float(slot.get("stdev") or 0.0)
        feat_norm = float(slot.get("normalised") or 0.0)
        norm_color = color_traffic_light(feat_norm, low_is_good=True)
        parts.append(
            f'<tr>'
            f'<td style="padding:.4rem .6rem">{_e(feat_label)}</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'font-family:monospace">{feat_mean:.3f}</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'font-family:monospace">{feat_stdev:.3f}</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'background:{norm_color};font-family:monospace">'
            f'{feat_norm:.3f}</td>'
            f'</tr>'
        )
    parts.append("</tbody></table>")
    return "".join(parts)


def build_image_predictive_html(
    aggregated: Optional[dict],
    labels: Optional[dict[str, str]] = None,
) -> str:
    """Construit la vue HTML « Profil d'image du corpus ».

    Parameters
    ----------
    aggregated:
        Sortie de ``aggregate_corpus_predictive``.  Si ``None``
        ou ``n_docs == 0``, retourne ``""``.
    labels:
        Dict i18n.  Clés sous le préfixe ``imgpred_*``.
    """
    if not aggregated:
        return ""
    if not aggregated.get("n_docs"):
        return ""
    labels = labels or {}
    title = labels.get(
        "imgpred_title", "Profil d'image du corpus",
    )
    note = labels.get(
        "imgpred_note",
        "Score de complexité paléographique combinant bruit, "
        "flou, faible contraste et rotation. Le score "
        "d'homogénéité signale si la moyenne globale est fiable "
        "(corpus uniforme) ou trompeuse (corpus hétérogène — "
        "voir alors la vue stratifiée).",
    )
    parts = [
        '<section class="imgpred-section" style="margin:1rem 0">',
        f'<h3 style="margin:0 0 .3rem 0">{_e(title)}</h3>',
        f'<div style="font-size:.85rem;opacity:.75;margin-bottom:.6rem">'
        f'{_e(note)}</div>',
    ]
    parts.append(_render_complexity_block(aggregated, labels))
    homo = aggregated.get("homogeneity")
    if isinstance(homo, dict):
        parts.append(_render_homogeneity_block(homo, labels))
    parts.append("</section>")
    return "".join(parts)


__all__ = ["build_image_predictive_html"]
