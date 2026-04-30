"""Rendu HTML « Comparaison contrôlée » — Sprint 96 (B.5).

Suite directe ``picarones/core/incremental_comparison.py``.
Pattern identique aux autres rendus : server-side, pas de JS,
anti-injection systématique.

Vue
---
Tableau ANOVA-like : pour chaque valeur du slot variant, mean
± stdev, rang moyen, n_observations.  Mean colorée en
gradient vert (meilleur) → rouge (pire), normalisée sur la
plage des moyennes observées.

Adaptive : ``""`` si ``analysis`` est ``None``.

Note d'intégration
------------------
Module pur — l'utilisateur compose :

.. code-block:: python

    from picarones.core.incremental_comparison import (
        PipelineRun, compare_isolated_effect,
    )
    from picarones.report.incremental_comparison_render import (
        build_incremental_comparison_html,
    )

    runs = [
        PipelineRun(name=p.name,
                    slots={"ocr": p.ocr, "llm": p.llm},
                    score=p.cer_mean)
        for p in benchmark.pipelines
    ]
    analysis = compare_isolated_effect(runs, "llm")
    html = build_incremental_comparison_html(analysis, labels)
"""

from __future__ import annotations

from html import escape as _e
from typing import Optional


def _color_for_score(
    score: float, low: float, high: float, higher_is_better: bool,
) -> str:
    """Vert (meilleur) → orange → rouge (pire)."""
    if high == low:
        return "#a7f0a7"
    rel = (score - low) / (high - low)
    if higher_is_better:
        rel = 1.0 - rel
    rel = max(0.0, min(1.0, rel))
    if rel < 0.5:
        t = rel / 0.5
        r = int(167 + (235 - 167) * t)
        g = int(240 + (180 - 240) * t)
        b = int(167 + (60 - 167) * t)
    else:
        t = (rel - 0.5) / 0.5
        r = int(235 + (220 - 235) * t)
        g = int(180 + (50 - 180) * t)
        b = int(60 + (50 - 60) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _format_score(value: Optional[float]) -> str:
    if value is None:
        return "—"
    if abs(value) < 1.0:
        return f"{value * 100:.2f}%"
    return f"{value:.3f}"


def build_incremental_comparison_html(
    analysis: Optional[dict],
    labels: Optional[dict[str, str]] = None,
) -> str:
    """Construit la vue HTML « Comparaison contrôlée ».

    Parameters
    ----------
    analysis:
        Sortie de ``compare_isolated_effect``.  ``None`` ou
        ``per_value`` vide → retourne ``""``.
    labels:
        Dict i18n.  Clés sous le préfixe ``incr_*``.
    """
    if not analysis:
        return ""
    per_value = analysis.get("per_value") or {}
    if not per_value:
        return ""
    labels = labels or {}
    title = labels.get(
        "incr_title", "Comparaison contrôlée par slot",
    )
    note = labels.get(
        "incr_note",
        "Effet isolé du module variant sur les pipelines en "
        "contrôlant les autres slots. Pour chaque valeur du "
        "slot, moyenne ± écart-type, rang moyen sur les groupes "
        "fixes, et nombre d'observations. Type design "
        "d'expérience pour des comparaisons honnêtes.",
    )
    slot_label = labels.get("incr_slot_label", "Slot variant")
    h_value = labels.get("incr_value", "Valeur")
    h_mean = labels.get("incr_mean", "Score moyen")
    h_stdev = labels.get("incr_stdev", "± σ")
    h_rank = labels.get("incr_rank", "Rang moyen")
    h_n_obs = labels.get("incr_n_obs", "Observations")
    h_groups = labels.get("incr_groups", "Groupes fixes")
    higher_is_better = bool(analysis.get("higher_is_better", False))

    # Plage de moyennes pour le code couleur
    means = [
        d["mean"] for d in per_value.values() if d.get("mean") is not None
    ]
    low = min(means) if means else 0.0
    high = max(means) if means else 0.0

    varying_slot = str(analysis.get("varying_slot") or "?")
    n_groups = int(analysis.get("n_groups") or 0)
    n_runs = int(analysis.get("n_runs") or 0)
    best = analysis.get("best_value")
    worst = analysis.get("worst_value")

    parts = [
        '<section class="incr-section" style="margin:1rem 0">',
        f'<h3 style="margin:0 0 .3rem 0">{_e(title)}</h3>',
        f'<div style="font-size:.85rem;opacity:.75;margin-bottom:.5rem">'
        f'{_e(note)}</div>',
        f'<div style="font-size:.85rem;margin-bottom:.5rem">'
        f'<strong>{_e(slot_label)} :</strong> '
        f'<code>{_e(varying_slot)}</code> &nbsp; '
        f'<span style="opacity:.75">'
        f'{n_runs} runs, {n_groups} {_e(h_groups.lower())}'
        f'</span></div>',
        '<table style="border-collapse:collapse;width:100%;'
        'font-size:.9rem">',
        '<thead><tr>',
    ]
    for col in (h_value, h_mean, h_stdev, h_rank, h_n_obs):
        parts.append(
            f'<th style="padding:.4rem .6rem;text-align:left;'
            f'border-bottom:1px solid #ccc;font-weight:600">'
            f'{_e(col)}</th>'
        )
    parts.append("</tr></thead><tbody>")

    # Tri par rang moyen ascendant
    rows = sorted(
        per_value.items(),
        key=lambda kv: (kv[1].get("mean_rank") or float("inf")),
    )
    for value, d in rows:
        mean = d.get("mean")
        stdev = d.get("stdev")
        rank = d.get("mean_rank")
        n_obs = int(d.get("n_observations") or 0)
        if isinstance(mean, (int, float)):
            color = _color_for_score(
                float(mean), low, high, higher_is_better,
            )
            mean_cell = (
                f'<td style="padding:.4rem .6rem;text-align:right;'
                f'background:{color};font-family:monospace;'
                f'font-weight:600">{_format_score(mean)}</td>'
            )
        else:
            mean_cell = (
                '<td style="padding:.4rem .6rem;text-align:right;'
                'opacity:.4">—</td>'
            )
        stdev_str = (
            f"± {_format_score(stdev)}"
            if isinstance(stdev, (int, float)) else "—"
        )
        rank_str = f"{rank:.2f}" if isinstance(rank, (int, float)) else "—"
        marker = ""
        if value == best:
            marker = ' <span style="color:#16a34a">★</span>'
        elif value == worst:
            marker = ' <span style="color:#dc2626">▼</span>'
        parts.append(
            f'<tr>'
            f'<td style="padding:.4rem .6rem;font-family:monospace">'
            f'{_e(str(value))}{marker}</td>'
            f'{mean_cell}'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'font-family:monospace">{stdev_str}</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'font-family:monospace">{rank_str}</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'font-family:monospace">{n_obs}</td>'
            f'</tr>'
        )
    parts.append("</tbody></table></section>")
    return "".join(parts)


__all__ = ["build_incremental_comparison_html"]
