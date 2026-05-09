"""Rendu HTML du coût marginal inter-moteurs (Sprint 91, A.II.6).

Phase 5.C — module relocalisé depuis
``picarones.report.marginal_cost_render`` vers
``picarones.reports.html.renderers.marginal_cost``.  Le chemin
legacy reste disponible via un shim avec ``DeprecationWarning`` ;
suppression prévue en 2.0.

Tableau récapitulatif des paires (A → B) avec le coût additionnel
par erreur évitée. Adaptive : retourne ``""`` si moins de 2 moteurs
ou si aucune paire n'a de données coût/erreur exploitables.

Permet à un archiviste de voir : *« passer de Tesseract à GPT-4o
coûte X € de plus par erreur évitée — est-ce justifié pour mon
budget ? »*
"""

from __future__ import annotations

from html import escape as _e
from typing import Optional


def build_marginal_cost_html(
    matrix: Optional[list[dict]],
    labels: Optional[dict[str, str]] = None,
) -> str:
    """Construit le tableau du coût marginal inter-moteurs.

    Parameters
    ----------
    matrix:
        Sortie de
        :func:`picarones.report.report_data.extra_metrics.compute_marginal_cost_section`.
        Liste de dicts triée par coût marginal croissant. Si ``None``
        ou vide, retourne ``""``.
    labels:
        Dict i18n optionnel.
    """
    if not matrix:
        return ""
    labels = labels or {}
    title = labels.get(
        "marginal_cost_title",
        "Coût marginal inter-moteurs (€ par erreur évitée)",
    )
    note = labels.get(
        "marginal_cost_note",
        "Pour chaque paire de moteurs (A → B), coût additionnel par "
        "erreur évitée en passant de A à B. Valeur basse = changement "
        "rentable. ‘Dominé’ = B est moins cher ET plus précis. Estimation "
        "des erreurs basée sur ``cer × 1000`` (proxy par 1000 pages).",
    )
    h_from = labels.get("marginal_cost_from", "Depuis")
    h_to = labels.get("marginal_cost_to", "Vers")
    h_avoided = labels.get("marginal_cost_avoided", "Erreurs évitées")
    h_delta = labels.get("marginal_cost_delta", "Coût Δ (€)")
    h_per_err = labels.get("marginal_cost_per_err", "€ / erreur évitée")
    h_dominated = labels.get("marginal_cost_dominated", "Dominé ?")

    parts = [
        '<section class="marginal-cost-section" style="margin:1rem 0">',
        f'<h3 style="margin:0 0 .3rem 0">{_e(title)}</h3>',
        f'<div style="font-size:.85rem;opacity:.75;margin-bottom:.5rem">'
        f'{_e(note)}</div>',
        '<table style="border-collapse:collapse;width:100%;'
        'font-size:.9rem">',
        '<thead><tr>',
    ]
    for h in (h_from, h_to, h_avoided, h_delta, h_per_err, h_dominated):
        parts.append(
            f'<th scope="col" style="padding:.4rem .6rem;text-align:left;'
            f'border-bottom:1px solid #ccc;font-weight:600">{_e(h)}</th>'
        )
    parts.append('</tr></thead><tbody>')

    for row in matrix:
        engine_a = row.get("engine_a") or row.get("from") or "?"
        engine_b = row.get("engine_b") or row.get("to") or "?"
        n_avoided = row.get("n_errors_avoided")
        cost_delta = row.get("cost_delta")
        cost_per_err = row.get("cost_per_avoided_error")
        dominated = row.get("dominated", False)

        n_avoided_cell = (
            f"{int(n_avoided)}" if isinstance(n_avoided, (int, float)) else "—"
        )
        cost_delta_cell = (
            f"{cost_delta:+.2f}" if isinstance(cost_delta, (int, float)) else "—"
        )
        if isinstance(cost_per_err, (int, float)):
            cost_per_err_cell = f"{cost_per_err:.2f}"
        else:
            cost_per_err_cell = "—"
        dominated_cell = (
            '<span style="color:#16a34a;font-weight:600">✓ B dominé par A</span>'
            if dominated else "—"
        )

        parts.append(
            f'<tr>'
            f'<td style="padding:.4rem .6rem">{_e(str(engine_a))}</td>'
            f'<td style="padding:.4rem .6rem">{_e(str(engine_b))}</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'font-family:monospace">{n_avoided_cell}</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'font-family:monospace">{cost_delta_cell}</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'font-family:monospace;font-weight:600">{cost_per_err_cell}</td>'
            f'<td style="padding:.4rem .6rem">{dominated_cell}</td>'
            f'</tr>'
        )
    parts.append('</tbody></table></section>')
    return "".join(parts)


__all__ = ["build_marginal_cost_html"]
