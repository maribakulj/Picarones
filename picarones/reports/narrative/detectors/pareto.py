"""Détecteurs narratifs orientés *coût/performance Pareto* (chantier 5).

2 détecteurs déplacés depuis ``narrative/detectors.py`` :

- :func:`detect_pareto_alternative` (Sprint 19) — alternative coût/qualité
- :func:`detect_cost_outlier`       (Sprint 19) — moteur dont le coût est aberrant
"""

from __future__ import annotations

import statistics as _stats
from typing import Optional

from picarones.domain.facts import Fact, FactImportance, FactType
from picarones.reports.narrative.registry import register_detector



@register_detector(
    FactType.PARETO_ALTERNATIVE,
    priority=90,
    importance=FactImportance.HIGH,
)
def detect_pareto_alternative(benchmark_data: dict) -> list[Fact]:
    """Moteur Pareto-dominant différent du leader CER.

    Lit ``benchmark_data["pareto"]["cost"]`` (Sprint 19) et émet un Fact si
    la frontière contient un moteur autre que le leader CER, pour souligner
    l'existence d'un compromis coût/qualité intéressant.
    """
    pareto = (benchmark_data.get("pareto") or {}).get("cost") or {}
    front = pareto.get("front") or []
    points = pareto.get("points") or []
    if len(front) < 2:
        return []

    ranking = benchmark_data.get("ranking") or []
    if not ranking:
        return []
    leader = ranking[0].get("engine")

    # Le moteur le moins cher sur le front (hors leader)
    alt: Optional[dict] = None
    for p in points:
        if p.get("engine") == leader:
            continue
        if p.get("engine") not in front:
            continue
        if alt is None or float(p.get("cost") or 0.0) < float(alt.get("cost") or 0.0):
            alt = p
    if alt is None:
        return []

    leader_point = next((p for p in points if p.get("engine") == leader), None)
    if leader_point is None:
        return []

    alt_cer = float(alt.get("cer") or 0.0)
    alt_cost = float(alt.get("cost") or 0.0)
    leader_cer = float(leader_point.get("cer") or 0.0)
    leader_cost = float(leader_point.get("cost") or 0.0)
    if alt_cost >= leader_cost or alt_cost <= 0:
        return []  # pas réellement moins cher — pas intéressant à remonter

    return [Fact(
        type=FactType.PARETO_ALTERNATIVE,
        importance=FactImportance.HIGH,
        payload={
            "engine": alt["engine"],
            "leader": leader,
            "cer": round(alt_cer, 4),
            "cer_pct": round(alt_cer * 100, 2),
            "cost": round(alt_cost, 2),
            "leader_cer": round(leader_cer, 4),
            "leader_cer_pct": round(leader_cer * 100, 2),
            "leader_cost": round(leader_cost, 2),
            "cost_saving_ratio": round(leader_cost / alt_cost, 1) if alt_cost > 0 else None,
            "delta_cer_pct": round((alt_cer - leader_cer) * 100, 2),
            # Unité du coût — propagée pour traçabilité (le template ne
            # hardcode plus "1000 pages").
            "cost_unit_pages": 1000,
        },
        engines_involved=(alt["engine"],),
    )]


@register_detector(
    FactType.COST_OUTLIER,
    priority=110,
    importance=FactImportance.MEDIUM,
)
def detect_cost_outlier(benchmark_data: dict) -> list[Fact]:
    """Moteur dont le coût est très disproportionné par rapport à son apport.

    Flag un moteur dont le coût ≥ 5× la médiane ET qui n'est pas sur le
    front Pareto (donc dominé par moins cher OU meilleur CER).
    """
    pareto = (benchmark_data.get("pareto") or {}).get("cost") or {}
    points = pareto.get("points") or []
    front = set(pareto.get("front") or [])
    if len(points) < 3:
        return []

    costs = [float(p["cost"]) for p in points if p.get("cost") is not None]
    if not costs:
        return []
    median_cost = _stats.median(costs)
    if median_cost <= 0:
        return []

    facts: list[Fact] = []
    for p in points:
        c = float(p.get("cost") or 0.0)
        if c < 5.0 * median_cost:
            continue
        if p["engine"] in front:
            continue  # sur le front → coût justifié par une qualité unique
        facts.append(Fact(
            type=FactType.COST_OUTLIER,
            importance=FactImportance.MEDIUM,
            payload={
                "engine": p["engine"],
                "cost": round(c, 2),
                "median_cost": round(median_cost, 2),
                "ratio_to_median": round(c / median_cost, 1),
                "cer_pct": round(float(p.get("cer") or 0.0) * 100, 2),
                "cost_unit_pages": 1000,
            },
            engines_involved=(p["engine"],),
        ))
    return facts


# ---------------------------------------------------------------------------
# Sprint A8 (item m-14) — détecteur PRICING_STALENESS_WARNING
# ---------------------------------------------------------------------------


@register_detector(
    FactType.PRICING_STALENESS_WARNING,
    # Priorité 200 — en queue (informationnel, pas bloquant pour le ranking).
    priority=200,
    importance=FactImportance.MEDIUM,
)
def detect_pricing_staleness(benchmark_data: dict) -> list[Fact]:
    """Émet un Fact si la table de pricing a dépassé sa date
    ``valid_until``.

    Lit ``benchmark_data["snapshots"]["pricing"]["meta"]["valid_until"]``
    (rempli par ``snapshot_all`` Sprint 27) et compare à la date du
    jour. Si la clé est absente ou la date mal formée, le détecteur
    reste silencieux — il ne casse pas un benchmark qui n'utilise pas
    le snapshot Pareto."""
    snapshots = benchmark_data.get("snapshots") or {}
    pricing = snapshots.get("pricing") or {}
    meta = pricing.get("meta") or {}
    valid_until_str = meta.get("valid_until")
    if not valid_until_str:
        return []

    from datetime import date, datetime

    try:
        valid_until = datetime.strptime(valid_until_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return []

    today = date.today()
    if today <= valid_until:
        return []  # pricing encore valide

    days_overdue = (today - valid_until).days
    return [Fact(
        type=FactType.PRICING_STALENESS_WARNING,
        importance=FactImportance.MEDIUM,
        payload={
            "valid_until": valid_until_str,
            "days_overdue": days_overdue,
            "today": today.isoformat(),
        },
        engines_involved=(),
    )]
