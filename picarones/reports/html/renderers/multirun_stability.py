"""Rendu HTML « Stabilité multi-runs » — Sprint 90 (A.II.4).

Suite directe ``picarones/core/reliability.compute_multirun_stability``
(Sprint 83).  Pattern identique aux autres rendus : server-side,
pas de JS, anti-injection systématique.

Note d'intégration
------------------
La stabilité multi-runs n'est pas calculée automatiquement par
le runner — l'utilisateur doit relancer son moteur LLM/VLM
plusieurs fois (option ``--repeats N`` du runner reportée à un
sprint dédié) et appeler ``compute_multirun_stability`` lui-
même.  Cette vue est donc un **module de rendu pur** que
l'utilisateur compose :

.. code-block:: python

    from picarones.evaluation.metrics.reliability import compute_multirun_stability
    from picarones.reports.html.renderers.multirun_stability import (
        build_multirun_stability_html,
    )

    stability = []
    for engine_name, runs in per_engine_runs.items():
        s = compute_multirun_stability(runs, reference=ref)
        if s is not None:
            s["engine_name"] = engine_name
            stability.append(s)
    html = build_multirun_stability_html(stability, labels)

Vue
---
Tableau moteur × {n_runs, CER moyen ± écart-type, CV (%),
% paires identiques, n outputs distincts}.  Cellule CV colorée
par gradient vert (stable) → rouge (instable, CV > 20 %).

Adaptive : ``""`` si la liste est vide ou que tous les
``cer_cv`` sont ``None``.
"""

from __future__ import annotations

from html import escape as _e
from typing import Optional

from picarones.reports._helpers.render_helpers import color_traffic_light


def build_multirun_stability_html(
    stability: Optional[list],
    labels: Optional[dict[str, str]] = None,
) -> str:
    """Construit la vue HTML de stabilité multi-runs.

    Parameters
    ----------
    stability:
        Liste de dicts (un par moteur) issus de
        ``compute_multirun_stability`` enrichis d'un
        ``engine_name``.  Si vide ou ``None``, retourne ``""``.
    labels:
        Dict i18n.  Clés sous le préfixe ``stability_*``.
    """
    if not stability:
        return ""
    rows = [s for s in stability if isinstance(s, dict) and s.get("engine_name")]
    if not rows:
        return ""
    labels = labels or {}
    title = labels.get("stability_title", "Stabilité multi-runs")
    note = labels.get(
        "stability_note",
        "Quand un moteur LLM/VLM est non déterministe, la "
        "variance entre runs successifs sur les mêmes documents "
        "est un proxy de la fiabilité scientifique. Un CV élevé "
        "ou un faible taux de runs identiques discrédite "
        "l'interprétation du CER moyen.",
    )
    h_engine = labels.get("stability_engine", "Moteur")
    h_n_runs = labels.get("stability_n_runs", "Runs")
    h_cer = labels.get("stability_cer", "CER moyen ± σ")
    h_cv = labels.get("stability_cv", "CV (%)")
    h_identical = labels.get("stability_identical", "% runs identiques")
    h_distinct = labels.get("stability_distinct", "Sorties distinctes")

    parts = [
        '<section class="stability-section" style="margin:1rem 0">',
        f'<h3 style="margin:0 0 .3rem 0">{_e(title)}</h3>',
        f'<div style="font-size:.85rem;opacity:.75;margin-bottom:.6rem">'
        f'{_e(note)}</div>',
        '<table style="border-collapse:collapse;width:100%;'
        'font-size:.9rem">',
        '<thead><tr>',
    ]
    for col in (h_engine, h_n_runs, h_cer, h_cv, h_identical, h_distinct):
        parts.append(
            f'<th scope=\"col\" style="padding:.4rem .6rem;text-align:left;'
            f'border-bottom:1px solid #ccc;font-weight:600">'
            f'{_e(col)}</th>'
        )
    parts.append("</tr></thead><tbody>")
    for stab in rows:
        engine = str(stab.get("engine_name") or "?")
        n_runs = int(stab.get("n_runs") or 0)
        cer_mean = stab.get("cer_mean")
        cer_stdev = stab.get("cer_stdev")
        cer_cv = stab.get("cer_cv")
        identical = stab.get("identical_run_rate")
        n_distinct = stab.get("n_distinct_outputs")
        if isinstance(cer_mean, (int, float)) and isinstance(cer_stdev, (int, float)):
            cer_str = f"{cer_mean * 100:.2f}% ± {cer_stdev * 100:.2f}%"
        elif isinstance(cer_mean, (int, float)):
            cer_str = f"{cer_mean * 100:.2f}%"
        else:
            cer_str = "—"
        if isinstance(cer_cv, (int, float)):
            cv_color = color_traffic_light(float(cer_cv), low_is_good=True, scale_max=0.25)
            cv_cell = (
                f'<td style="padding:.4rem .6rem;text-align:right;'
                f'background:{cv_color};font-family:monospace;'
                f'font-weight:600">{float(cer_cv) * 100:.1f}</td>'
            )
        else:
            cv_cell = (
                '<td style="padding:.4rem .6rem;text-align:right;'
                'opacity:.4">—</td>'
            )
        identical_str = (
            f"{float(identical) * 100:.1f}"
            if isinstance(identical, (int, float)) else "—"
        )
        distinct_str = str(n_distinct) if isinstance(n_distinct, int) else "—"
        parts.append(
            f'<tr>'
            f'<td style="padding:.4rem .6rem">{_e(engine)}</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'font-family:monospace">{n_runs}</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'font-family:monospace">{cer_str}</td>'
            f'{cv_cell}'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'font-family:monospace">{identical_str}</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'font-family:monospace">{distinct_str}</td>'
            f'</tr>'
        )
    parts.append("</tbody></table></section>")
    return "".join(parts)


__all__ = ["build_multirun_stability_html"]
