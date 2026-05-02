"""Helpers numériques internes au sous-package report_data.

Petites fonctions utilitaires partagées par tous les builders de
sections (engines, documents, statistics, scatter, pareto). Ne pas
importer depuis l'extérieur du sous-package — ces helpers sont
spécifiques aux conventions du dict JSON consommé par le template.
"""

from __future__ import annotations

from typing import Optional


def safe_round(v: Optional[float], decimals: int = 4) -> float:
    """Arrondit un float optionnel ; ``None`` devient ``0.0``."""
    return round(v or 0.0, decimals)


def percent_string(v: Optional[float], decimals: int = 2) -> str:
    """Formate un ratio ∈ [0, 1] en chaîne pourcentage : ``0.4723 → "47.23 %"``.

    ``None`` → ``"—"``. Conservé pour rétrocompat avec d'éventuels
    callers externes (Sprint 7 historique).
    """
    if v is None:
        return "—"
    return f"{v * 100:.{decimals}f} %"


__all__ = ["safe_round", "percent_string"]
