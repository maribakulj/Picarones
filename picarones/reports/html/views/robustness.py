"""Vue robustesse — chantier 3 post-Sprint 97.

Branche le renderer :func:`picarones.reports.html.renderers.robustness_projection`
(Sprint 88) au workflow ``picarones robustness`` (CLI Sprint 8).

Cette vue ne s'inclut pas dans le rapport classique : la robustesse
synthétique exige une étape de calcul lourde (re-OCR sur des
versions dégradées de chaque image) qui sort du flux standard.
Le module est exposé pour que l'orchestrateur ``robustness_cmd``
de la CLI puisse composer un mini-rapport HTML autonome.

Sources de données
------------------
- ``opts["projection"]`` : sortie de
  :func:`picarones.evaluation.metrics.robustness_projection.project_robustness_on_corpus`.
- ``opts["aggregated"]`` : sortie de
  :func:`picarones.evaluation.metrics.robustness_projection.aggregate_projection_per_engine`.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def build_robustness_view_html(
    report_data: Optional[dict] = None,
    labels: Optional[dict[str, str]] = None,
    *,
    projection: Optional[dict] = None,
    aggregated: Optional[dict] = None,
) -> str:
    """Compose la vue robustesse.

    Parameters
    ----------
    report_data:
        Inutilisé (la robustesse a son propre flux).  Présent pour
        homogénéité avec les autres vues du chantier 3.
    labels:
        Dict i18n complet.
    projection:
        Sortie de
        :func:`picarones.evaluation.metrics.robustness_projection.project_robustness_on_corpus`.
    aggregated:
        Sortie de
        :func:`picarones.evaluation.metrics.robustness_projection.aggregate_projection_per_engine`.
        Si ``None`` mais ``projection`` fourni, recalculé.

    Returns
    -------
    str
        HTML de la vue ou ``""`` si pas de projection fournie.
    """
    if projection is None:
        return ""
    labels = labels or {}
    blocks: list[tuple[str, str]] = []

    try:
        from picarones.reports.html.renderers.robustness_projection import (
            build_robustness_projection_html,
        )
        html = build_robustness_projection_html(
            projection, aggregated=aggregated, labels=labels,
        )
        if html:
            blocks.append((
                labels.get(
                    "robust_view_title", "Déficit projeté de robustesse",
                ),
                html,
            ))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[robustness_view.projection] dégradé : %s", exc,
        )

    if not blocks:
        return ""

    from picarones.reports.html.views.economics import _render_view_shell

    return _render_view_shell(
        view_title=labels.get(
            "robust_view_title", "Robustesse projetée sur le corpus",
        ),
        view_note=labels.get(
            "robust_view_note",
            "Projection des courbes de dégradation synthétique "
            "(bruit, flou, rotation) sur les caractéristiques d'image "
            "réelles du corpus. Permet d'estimer le déficit attendu "
            "sans relancer un OCR coûteux par dégradation.",
        ),
        blocks=blocks,
    )


__all__ = ["build_robustness_view_html"]
