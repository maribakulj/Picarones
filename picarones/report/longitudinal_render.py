"""Rendu HTML « Évolution dans le temps » — Sprint 92 (A.II.9).

Suite directe ``picarones/core/longitudinal.py``.  Pattern
identique aux autres rendus : server-side, pas de JS, anti-
injection systématique.

Vue
---
Tableau résumé moteur × {n_runs, premier CER, dernier CER,
variation cumulée colorée, pente annualisée, R², point de
rupture si détecté}.

Adaptive : ``""`` si la liste est vide.

Note d'intégration
------------------
Module pur — l'utilisateur compose :

.. code-block:: python

    from picarones.measurements.history import BenchmarkHistory
    from picarones.measurements.longitudinal import compute_corpus_longitudinal
    from picarones.report.longitudinal_render import build_longitudinal_html

    hist = BenchmarkHistory(db_path)
    entries = hist.list_entries()
    trends = compute_corpus_longitudinal(entries, corpus_name)
    html = build_longitudinal_html(trends, labels)
"""

from __future__ import annotations

from html import escape as _e
from typing import Optional


def _color_for_delta(delta_pct: float) -> str:
    """Vert (≈0) → orange → rouge (≥ +5 pts CER) ;
    vert → bleu (≤ -5 pts CER, amélioration)."""
    if abs(delta_pct) < 1.0:
        return "#a7f0a7"
    f = max(-1.0, min(1.0, delta_pct / 5.0))
    if f >= 0:
        # vert → orange profond → rouge profond
        if f < 0.5:
            t = f / 0.5
            r = int(167 + (235 - 167) * t)
            g = int(240 + (180 - 240) * t)
            b = int(167 + (60 - 167) * t)
        else:
            t = (f - 0.5) / 0.5
            r = int(235 + (220 - 235) * t)
            g = int(180 + (50 - 180) * t)
            b = int(60 + (50 - 60) * t)
    else:
        # vert → bleu (amélioration)
        f = -f
        r = int(167 + (90 - 167) * f)
        g = int(240 + (160 - 240) * f)
        b = int(167 + (210 - 167) * f)
    return f"#{r:02x}{g:02x}{b:02x}"


def build_longitudinal_html(
    trends: Optional[list],
    labels: Optional[dict[str, str]] = None,
) -> str:
    """Construit la vue HTML longitudinale.

    Parameters
    ----------
    trends:
        Sortie de ``compute_corpus_longitudinal`` (liste de
        dicts).  Si ``None`` ou vide, retourne ``""``.
    labels:
        Dict i18n.  Clés sous le préfixe ``longitudinal_*``.
    """
    if not trends:
        return ""
    rows = [t for t in trends if isinstance(t, dict) and t.get("engine_name")]
    if not rows:
        return ""
    labels = labels or {}
    title = labels.get(
        "longitudinal_title", "Évolution dans le temps",
    )
    note = labels.get(
        "longitudinal_note",
        "Tendance et points de rupture sur l'historique SQLite "
        "des runs précédents. Une variation positive signale "
        "une dégradation cumulée — utile pour relier une "
        "régression à un changement de pipeline ou de modèle.",
    )
    h_engine = labels.get("longitudinal_engine", "Moteur")
    h_n_runs = labels.get("longitudinal_n_runs", "Runs")
    h_first = labels.get("longitudinal_first", "Premier CER")
    h_last = labels.get("longitudinal_last", "Dernier CER")
    h_delta = labels.get("longitudinal_delta", "Δ cumulé (pts)")
    h_slope = labels.get("longitudinal_slope", "Pente annuelle (pts/an)")
    h_r2 = labels.get("longitudinal_r2", "R²")
    h_change = labels.get("longitudinal_change", "Rupture")

    parts = [
        '<section class="longitudinal-section" style="margin:1rem 0">',
        f'<h3 style="margin:0 0 .3rem 0">{_e(title)}</h3>',
        f'<div style="font-size:.85rem;opacity:.75;margin-bottom:.6rem">'
        f'{_e(note)}</div>',
        '<table style="border-collapse:collapse;width:100%;'
        'font-size:.9rem">',
        '<thead><tr>',
    ]
    for col in (h_engine, h_n_runs, h_first, h_last, h_delta,
                h_slope, h_r2, h_change):
        parts.append(
            f'<th style="padding:.4rem .6rem;text-align:left;'
            f'border-bottom:1px solid #ccc;font-weight:600">'
            f'{_e(col)}</th>'
        )
    parts.append("</tr></thead><tbody>")
    for entry in sorted(
        rows,
        key=lambda r: -float(r.get("absolute_delta") or 0.0),
    ):
        engine = str(entry.get("engine_name") or "?")
        n_runs = int(entry.get("n_runs") or 0)
        first_cer = float(entry.get("first_cer") or 0.0)
        last_cer = float(entry.get("last_cer") or 0.0)
        delta_pct = float(entry.get("absolute_delta_pct") or 0.0)
        delta_color = _color_for_delta(delta_pct)
        trend = entry.get("trend") or {}
        slope = trend.get("slope")
        r2 = trend.get("r_squared")
        slope_str = (
            f"{float(slope) * 365 * 100:+.2f}"
            if isinstance(slope, (int, float)) else "—"
        )
        r2_str = (
            f"{float(r2):.2f}"
            if isinstance(r2, (int, float)) else "—"
        )
        cp = entry.get("change_point")
        if isinstance(cp, dict) and cp.get("timestamp"):
            cp_delta = float(cp.get("delta") or 0.0)
            cp_str = (
                f'{_e(str(cp["timestamp"]))} '
                f'<span style="opacity:.75">'
                f'({cp_delta * 100:+.2f} pts)</span>'
            )
        else:
            cp_str = "—"
        parts.append(
            f'<tr>'
            f'<td style="padding:.4rem .6rem">{_e(engine)}</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'font-family:monospace">{n_runs}</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'font-family:monospace">{first_cer * 100:.2f}%</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'font-family:monospace">{last_cer * 100:.2f}%</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'background:{delta_color};font-family:monospace;'
            f'font-weight:600">{delta_pct:+.2f}</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'font-family:monospace">{slope_str}</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'font-family:monospace">{r2_str}</td>'
            f'<td style="padding:.4rem .6rem">{cp_str}</td>'
            f'</tr>'
        )
    parts.append("</tbody></table></section>")
    return "".join(parts)


__all__ = ["build_longitudinal_html"]
