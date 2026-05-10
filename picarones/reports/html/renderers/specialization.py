"""Rendu HTML « Spécialisation inter-moteurs » — Sprint 89
(A.II.8b).

Suite directe ``picarones/core/specialization.py``.  Vue
**factuelle** sans recommandation : on liste les paires de
moteurs les plus spécialisées, le chercheur arbitre.

Pattern identique aux autres rendus : server-side, pas de JS,
anti-injection systématique.
"""

from __future__ import annotations

from html import escape as _e
from typing import Optional

from picarones.evaluation.metrics.specialization import (
    compute_specialization_matrix,
    top_specialized_pairs,
)
from picarones.reports._helpers.render_helpers import color_single_gradient

#: Bleu profond cible — préservé de l'ancien `_color_for_score` local.
_SPECIALIZATION_BLUE = (50, 110, 180)


def _category_label(cat: str, labels: dict[str, str]) -> str:
    return labels.get(f"specialization_cat_{cat}", cat)


def build_specialization_html(
    taxonomies: Optional[dict[str, dict[str, float]]],
    labels: Optional[dict[str, str]] = None,
    *,
    top_n: int = 5,
) -> str:
    """Construit la vue HTML de spécialisation inter-moteurs.

    Parameters
    ----------
    taxonomies:
        Map ``{engine: {error_class: count}}``.  Si ``None`` ou
        moins de 2 moteurs, retourne ``""``.
    labels:
        Dict i18n.  Clés sous le préfixe ``specialization_*``.
    top_n:
        Nombre de paires à afficher (défaut 5).
    """
    if not taxonomies or len(taxonomies) < 2:
        return ""
    matrix_data = compute_specialization_matrix(taxonomies)
    if not matrix_data:
        return ""
    pairs = top_specialized_pairs(matrix_data, n=top_n)
    if not pairs:
        return ""
    labels = labels or {}
    title = labels.get(
        "specialization_title", "Spécialisation inter-moteurs",
    )
    note = labels.get(
        "specialization_note",
        "Score de divergence Jensen-Shannon entre les profils "
        "taxonomiques de chaque paire de moteurs (0 = profils "
        "identiques, 1 = totalement disjoints). Une paire très "
        "spécialisée signale des erreurs de natures différentes "
        "— c'est au chercheur d'en tirer parti, pas à l'outil "
        "de prescrire un ensemble.",
    )
    h_a = labels.get("specialization_engine_a", "Moteur A")
    h_b = labels.get("specialization_engine_b", "Moteur B")
    h_score = labels.get("specialization_score", "Score")
    h_cat = labels.get("specialization_category", "Lecture")

    parts = [
        '<section class="specialization-section" '
        'style="margin:1rem 0">',
        f'<h3 style="margin:0 0 .3rem 0">{_e(title)}</h3>',
        f'<div style="font-size:.85rem;opacity:.75;margin-bottom:.6rem">'
        f'{_e(note)}</div>',
        '<table style="border-collapse:collapse;width:100%;'
        'font-size:.9rem">',
        '<thead><tr>',
    ]
    for col in (h_a, h_b, h_score, h_cat):
        parts.append(
            f'<th scope=\"col\" style="padding:.4rem .6rem;text-align:left;'
            f'border-bottom:1px solid #ccc;font-weight:600">'
            f'{_e(col)}</th>'
        )
    parts.append("</tr></thead><tbody>")
    for pair in pairs:
        score = float(pair.get("score") or 0.0)
        cat = pair.get("category") or "?"
        color = color_single_gradient(score, end_rgb=_SPECIALIZATION_BLUE)
        parts.append(
            f'<tr>'
            f'<td style="padding:.4rem .6rem">'
            f'{_e(str(pair.get("engine_a", "?")))}</td>'
            f'<td style="padding:.4rem .6rem">'
            f'{_e(str(pair.get("engine_b", "?")))}</td>'
            f'<td style="padding:.4rem .6rem;text-align:right;'
            f'background:{color};font-family:monospace;font-weight:600">'
            f'{score:.3f}</td>'
            f'<td style="padding:.4rem .6rem">'
            f'{_e(_category_label(cat, labels))}</td>'
            f'</tr>'
        )
    parts.append("</tbody></table></section>")
    return "".join(parts)


__all__ = ["build_specialization_html"]
