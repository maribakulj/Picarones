"""Helpers de rendu pour le score de difficulté intrinsèque.

Sprint A3 (item B-2 de l'audit institutional-readiness-2026-05) :
``difficulty_color`` vivait précédemment dans
``picarones/measurements/difficulty.py`` et y violait la règle
Cercle 2 → Cercle 3 par un import paresseux de
``picarones.reports._helpers.colors``. La fonction est désormais placée à sa
juste place — Cercle 3, à côté de la palette qu'elle consomme — et
``measurements/difficulty.py`` ne contient plus que de la logique
purement numérique.

Le module pur ``picarones.evaluation.metrics.difficulty`` reste utilisable
sans dépendance vers ``picarones.report``.
"""

from __future__ import annotations

from picarones.reports._helpers.colors import (
    COLOR_GREEN,
    COLOR_ORANGE,
    COLOR_RED,
    COLOR_YELLOW,
)


def difficulty_color(score: float) -> str:
    """Retourne une couleur CSS pour un score de difficulté ∈ [0, 1].

    Convention :

    - score < 0.25  → vert      (« facile »)
    - score < 0.50  → jaune     (« modéré »)
    - score < 0.75  → orange    (« difficile »)
    - score ≥ 0.75  → rouge     (« très difficile »)

    Le label texte correspondant est produit par
    :func:`picarones.evaluation.metrics.difficulty.difficulty_label`.
    """
    if score < 0.25:
        return COLOR_GREEN
    if score < 0.50:
        return COLOR_YELLOW
    if score < 0.75:
        return COLOR_ORANGE
    return COLOR_RED
