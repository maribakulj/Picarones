"""Projection de coût en volume cible — Sprint 79 (A.I.6).

A.I.6 du plan d'évolution 2026.

Pourquoi ce module
------------------
La vue Pareto (Sprint 20) trace CER vs coût mais le coût est par
unité (1 000 pages).  Pour décider business-side, il faut projeter
ce coût sur le **volume cible** que l'utilisateur prévoit de
traiter — payer 50 € de plus sur 50 pages est trivial, sur
5 millions ça change tout.

Sortie typique
--------------
*« Pour vos 80 000 pages BMS — Tesseract = 3 €, Pero = 0 € (local
amorti), Mistral OCR = 280 €, GPT-4o post-correction = 600 €. »*

Aucun seuil arbitraire imposé : le module fournit les chiffres,
le chercheur arbitre selon son budget.

Dépendance
----------
S'appuie sur ``picarones.evaluation.metrics.pricing`` (Sprint 20) qui expose
``EngineCost.cost_per_1k_pages_eur`` et
``co2_per_1k_pages_g``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from picarones.evaluation.metrics.pricing import EngineCost

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProjectedCost:
    """Coût total projeté d'un moteur pour un volume cible."""
    engine_key: str
    target_pages: int
    cost_total_eur: Optional[float]
    co2_total_g: Optional[float]
    cost_per_1k_pages_eur: Optional[float]
    co2_per_1k_pages_g: Optional[float]
    type: str  # "local" / "cloud_api" / "unknown"

    def as_dict(self) -> dict:
        return {
            "engine_key": self.engine_key,
            "target_pages": self.target_pages,
            "cost_total_eur": self.cost_total_eur,
            "co2_total_g": self.co2_total_g,
            "cost_per_1k_pages_eur": self.cost_per_1k_pages_eur,
            "co2_per_1k_pages_g": self.co2_per_1k_pages_g,
            "type": self.type,
        }


def project_cost_total(
    engine_cost: EngineCost, target_pages: int,
) -> Optional[float]:
    """Coût total projeté en euros pour ``target_pages`` pages.

    Retourne ``None`` si ``cost_per_1k_pages_eur`` est ``None``
    (données insuffisantes) ou si ``target_pages`` est négatif.
    """
    if target_pages < 0:
        return None
    if engine_cost.cost_per_1k_pages_eur is None:
        return None
    return engine_cost.cost_per_1k_pages_eur * target_pages / 1000.0


def project_co2_total(
    engine_cost: EngineCost, target_pages: int,
) -> Optional[float]:
    """Empreinte CO₂ totale en grammes pour ``target_pages`` pages."""
    if target_pages < 0:
        return None
    if engine_cost.co2_per_1k_pages_g is None:
        return None
    return engine_cost.co2_per_1k_pages_g * target_pages / 1000.0


def project_engine(
    engine_cost: EngineCost, target_pages: int,
) -> ProjectedCost:
    """Retourne le ``ProjectedCost`` complet pour un moteur."""
    return ProjectedCost(
        engine_key=engine_cost.engine_key,
        target_pages=int(target_pages),
        cost_total_eur=project_cost_total(engine_cost, target_pages),
        co2_total_g=project_co2_total(engine_cost, target_pages),
        cost_per_1k_pages_eur=engine_cost.cost_per_1k_pages_eur,
        co2_per_1k_pages_g=engine_cost.co2_per_1k_pages_g,
        type=engine_cost.type,
    )


def project_all_engines(
    engine_costs: dict[str, EngineCost],
    target_pages: int,
) -> dict[str, ProjectedCost]:
    """Projette les coûts de plusieurs moteurs sur le volume cible.

    Retourne un dict ``{engine_name: ProjectedCost}`` avec entrée
    pour chaque moteur, y compris ceux sans données de coût (où
    ``cost_total_eur`` sera ``None``).
    """
    if target_pages < 0:
        raise ValueError("target_pages doit être ≥ 0")
    return {
        name: project_engine(cost, target_pages)
        for name, cost in engine_costs.items()
    }


def cost_gap_table(
    projections: dict[str, ProjectedCost],
    baseline_engine: str,
) -> dict[str, dict[str, Optional[float]]]:
    """Pour chaque moteur, écart de coût total vs baseline.

    Retourne ``{engine: {"total": float, "delta_abs": float,
    "delta_rel": float}}`` où :

    - ``delta_abs`` = ``cost - cost_baseline`` (None si l'un des
      deux est None)
    - ``delta_rel`` = ``delta_abs / cost_baseline`` (None si
      baseline = 0 ou None)

    Lève ``KeyError`` si la baseline est inconnue.
    """
    if baseline_engine not in projections:
        raise KeyError(
            f"baseline {baseline_engine!r} absente des projections",
        )
    baseline_total = projections[baseline_engine].cost_total_eur
    out: dict[str, dict[str, Optional[float]]] = {}
    for name, proj in projections.items():
        total = proj.cost_total_eur
        if total is None or baseline_total is None:
            delta_abs: Optional[float] = None
            delta_rel: Optional[float] = None
        else:
            delta_abs = total - baseline_total
            if baseline_total != 0:
                delta_rel = delta_abs / baseline_total
            else:
                delta_rel = None
        out[name] = {
            "total": total,
            "delta_abs": delta_abs,
            "delta_rel": delta_rel,
        }
    return out


__all__ = [
    "ProjectedCost",
    "project_cost_total",
    "project_co2_total",
    "project_engine",
    "project_all_engines",
    "cost_gap_table",
]
