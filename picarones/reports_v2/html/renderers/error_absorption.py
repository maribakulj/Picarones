"""Rendu HTML « Absorption d'erreur » — Sprint 94 (B.3).

Phase 5.C — module relocalisé depuis
``picarones.report.error_absorption_render`` vers
``picarones.reports_v2.html.renderers.error_absorption``.  Le chemin
legacy reste disponible via un shim avec ``DeprecationWarning`` ;
suppression prévue en 2.0.

Suite directe ``picarones/core/error_absorption.py``.  Pattern
identique aux autres rendus : server-side, pas de JS, anti-
injection systématique.

Vue
---
Tableau résumé des jonctions du pipeline ; chaque ligne décrit
un module post-correction et présente :

- erreurs en entrée vs en sortie ;
- nb corrigées (gradient vert), nb introduites (gradient rouge) ;
- taux de correction (gradient vert), taux d'introduction
  (gradient rouge) ;
- amélioration nette (n_corrected - n_introduced) — coloré.
- éventuellement un échantillon de tokens corrigés/introduits.

Adaptive : ``""`` si la liste est vide.

Note d'intégration
------------------
Module pur — la liste ``junctions`` est composée par
l'utilisateur depuis son benchmark de pipeline composée :

.. code-block:: python

    from picarones.evaluation.metrics.error_absorption import (
        compute_error_absorption, aggregate_error_absorption,
    )
    from picarones.reports_v2.html.renderers.error_absorption import (
        build_error_absorption_html,
    )

    junctions = []
    for step in pipeline.steps_with_text_output:
        per_doc = [
            compute_error_absorption(doc.gt_text, doc.before_text,
                                     doc.after_text)
            for doc in benchmark.docs
        ]
        agg = aggregate_error_absorption(per_doc)
        if agg is not None:
            agg["junction_name"] = step.name
            junctions.append(agg)
    html = build_error_absorption_html(junctions, labels)
"""

from __future__ import annotations

from html import escape as _e
from typing import Optional

from picarones.reports_v2._helpers.render_helpers import color_diverging, color_traffic_light


# Palette « net improvement » : vert clair au centre, vert profond
# si favorable (net > 0), rouge si défavorable (net < 0).  Centrée
# sur le vert clair car un delta nul est déjà « pas de régression ».
_NET_NEUTRAL_RGB = (167, 240, 167)
_NET_POSITIVE_RGB = (90, 200, 90)
_NET_NEGATIVE_RGB = (220, 50, 50)


def build_error_absorption_html(
    junctions: Optional[list],
    labels: Optional[dict[str, str]] = None,
    *,
    sample_max: int = 8,
) -> str:
    """Construit la vue HTML « Absorption d'erreur ».

    Parameters
    ----------
    junctions:
        Liste de dicts (un par jonction de pipeline), enrichis
        d'un ``junction_name``.  Si vide ou ``None``, retourne
        ``""``.
    labels:
        Dict i18n.  Clés sous le préfixe ``absorption_*``.
    sample_max:
        Nombre maximal de tokens corrigés/introduits affichés
        en cellule échantillon.
    """
    if not junctions:
        return ""
    rows = [
        j for j in junctions
        if isinstance(j, dict) and j.get("junction_name")
    ]
    if not rows:
        return ""
    labels = labels or {}
    title = labels.get(
        "absorption_title", "Absorption d'erreur par jonction",
    )
    note = labels.get(
        "absorption_note",
        "À chaque jonction du pipeline, deux flux sont mesurés "
        "indépendamment : combien d'erreurs sont corrigées et "
        "combien sont introduites. Une jonction qui corrige "
        "beaucoup mais introduit aussi beaucoup absorbe les "
        "différences amont au lieu de les améliorer.",
    )
    h_junction = labels.get("absorption_junction", "Jonction")
    h_errors_before = labels.get("absorption_errors_before", "Erreurs avant")
    h_errors_after = labels.get("absorption_errors_after", "Erreurs après")
    h_corrected = labels.get("absorption_corrected", "Corrigées")
    h_introduced = labels.get("absorption_introduced", "Introduites")
    h_corr_rate = labels.get("absorption_corr_rate", "% corrigées")
    h_intro_rate = labels.get("absorption_intro_rate", "% introduites")
    h_net = labels.get("absorption_net", "Amélioration nette")
    h_sample = labels.get("absorption_sample", "Échantillon (intro)")

    # Saturation pour le gradient « net »
    max_abs_net = max(
        (abs(int(r.get("net_improvement") or 0)) for r in rows), default=1,
    ) or 1

    parts = [
        '<section class="absorption-section" style="margin:1rem 0">',
        f'<h3 style="margin:0 0 .3rem 0">{_e(title)}</h3>',
        f'<div style="font-size:.85rem;opacity:.75;margin-bottom:.6rem">'
        f'{_e(note)}</div>',
        '<table style="border-collapse:collapse;width:100%;'
        'font-size:.9rem">',
        '<thead><tr>',
    ]
    for col in (h_junction, h_errors_before, h_errors_after,
                h_corrected, h_introduced, h_corr_rate,
                h_intro_rate, h_net, h_sample):
        parts.append(
            f'<th scope=\"col\" style="padding:.4rem .6rem;text-align:left;'
            f'border-bottom:1px solid #ccc;font-weight:600">'
            f'{_e(col)}</th>'
        )
    parts.append("</tr></thead><tbody>")
    for entry in rows:
        name = str(entry.get("junction_name") or "?")
        n_eb = int(entry.get("n_errors_before") or 0)
        n_ea = int(entry.get("n_errors_after") or 0)
        n_corr = int(entry.get("n_corrected") or 0)
        n_intro = int(entry.get("n_introduced") or 0)
        net = int(entry.get("net_improvement") or 0)
        corr_rate = entry.get("correction_rate")
        intro_rate = entry.get("introduction_rate")
        if isinstance(corr_rate, (int, float)):
            corr_rate_str = f"{corr_rate * 100:.1f}%"
            corr_color = color_traffic_light(float(corr_rate))
            corr_cell = (
                f'<td style="padding:.4rem .6rem;text-align:right;'
                f'background:{corr_color};font-family:monospace;'
                f'font-weight:600">{corr_rate_str}</td>'
            )
        else:
            corr_cell = (
                '<td style="padding:.4rem .6rem;text-align:right;'
                'opacity:.4">—</td>'
            )
        if isinstance(intro_rate, (int, float)):
            intro_rate_str = f"{intro_rate * 100:.1f}%"
            intro_color = color_traffic_light(float(intro_rate), low_is_good=True)
            intro_cell = (
                f'<td style="padding:.4rem .6rem;text-align:right;'
                f'background:{intro_color};font-family:monospace;'
                f'font-weight:600">{intro_rate_str}</td>'
            )
        else:
            intro_cell = (
                '<td style="padding:.4rem .6rem;text-align:right;'
                'opacity:.4">—</td>'
            )
        net_color = color_diverging(
            float(net),
            max_abs=float(max_abs_net) if max_abs_net else 1.0,
            neutral_rgb=_NET_NEUTRAL_RGB,
            positive_rgb=_NET_POSITIVE_RGB,
            negative_rgb=_NET_NEGATIVE_RGB,
        )
        intro_sample = entry.get("introduced_tokens_sample") or []
        sample_cell_text = ", ".join(
            _e(str(t)) for t in intro_sample[:sample_max]
        ) or "—"
        if len(intro_sample) > sample_max:
            sample_cell_text += " …"
        parts.append(
            f'<tr>'
            f'<td style="padding:.4rem .6rem">{_e(name)}</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'font-family:monospace">{n_eb}</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'font-family:monospace">{n_ea}</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'font-family:monospace">{n_corr}</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'font-family:monospace">{n_intro}</td>'
            f'{corr_cell}'
            f'{intro_cell}'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'background:{net_color};font-family:monospace;'
            f'font-weight:600">{net:+d}</td>'
            f'<td style="padding:.4rem .6rem;font-family:monospace;'
            f'font-size:.8rem">{sample_cell_text}</td>'
            f'</tr>'
        )
    parts.append("</tbody></table></section>")
    return "".join(parts)


__all__ = ["build_error_absorption_html"]
