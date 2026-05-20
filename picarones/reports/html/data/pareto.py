"""Front Pareto coût/qualité

Construit trois fronts Pareto avec des axes alternatifs :

- ``cost`` — CER vs coût € / 1000 pages.
- ``speed`` — CER vs durée moyenne par page.
- ``co2`` — CER vs empreinte carbone (g CO₂ / 1000 pages, expérimental).

API
---
Deux fonctions séparées pour rendre le contrat explicite :

1. :func:`attach_engine_costs` — **mute en place** ``engines_summary``
   en y ajoutant ``mean_duration_seconds`` et ``cost`` (extraits du
   benchmark et de la table de pricing). Le nom dit clairement qu'il
   y a mutation.
2. :func:`build_pareto_section` — **fonction pure**, lit les coûts
   déjà attachés à ``engines_summary``. Retourne le dict ``pareto``
   prêt pour le template.

L'orchestrateur (``__init__.py``) appelle les deux dans l'ordre.
Cette séparation rend possible :

- Tester :func:`build_pareto_section` indépendamment avec un
  ``engines_summary`` pré-fabriqué.
- Réutiliser les coûts attachés sans recalculer Pareto.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from picarones.evaluation.metrics.pricing import (
    build_costs_for_benchmark,
    load_pricing_database,
)
from picarones.evaluation.statistics import compute_pareto_front

if TYPE_CHECKING:
    from picarones.evaluation.benchmark_result import BenchmarkResult


def attach_engine_costs(
    engines_summary: list[dict], benchmark: "BenchmarkResult",
) -> None:
    """Annote chaque entrée de ``engines_summary`` avec son coût.

    **Mute en place** : ajoute deux champs à chaque dict moteur :

    - ``mean_duration_seconds`` (float ou ``None`` si pas de durée).
    - ``cost`` : dict de la forme ``{cost_per_1k_pages_eur: ...,
      co2_per_1k_pages_g: ..., ...}`` ou ``None`` si pricing
      indisponible.

    Doit être appelée AVANT :func:`build_pareto_section`, qui lit
    ces deux champs.
    """
    durations_by_engine: dict[str, float] = {}
    for report in benchmark.engine_reports:
        durs = [
            dr.duration_seconds
            for dr in report.document_results
            if dr.duration_seconds is not None
        ]
        if durs:
            durations_by_engine[report.engine_name] = sum(durs) / len(durs)

    costs_by_engine = build_costs_for_benchmark(
        engines_summary, durations_by_engine,
    )
    for entry in engines_summary:
        name = entry["name"]
        entry["mean_duration_seconds"] = (
            round(durations_by_engine.get(name, 0.0), 4)
            if name in durations_by_engine else None
        )
        entry["cost"] = costs_by_engine.get(name)


def build_pareto_section(engines_summary: list[dict]) -> dict:
    """Construit le bloc ``pareto`` du dict de rapport.

    **Fonction pure** : ne mute rien. Lit ``mean_duration_seconds``
    et ``cost`` qui doivent avoir été attachés en amont par
    :func:`attach_engine_costs`. Si ces champs sont absents, le
    moteur est silencieusement omis du front (cohérent avec un
    moteur qui n'a pas de prix connu).

    Retour
    ------
    dict
        Trois fronts Pareto (``cost``, ``speed``, ``co2``) plus
        ``pricing_meta`` (table de pricing utilisée).
    """
    pricing_defaults, _ = load_pricing_database()

    pareto_points = []
    for entry in engines_summary:
        cer = entry.get("cer")
        cost = (entry.get("cost") or {}).get("cost_per_1k_pages_eur")
        if cer is None or cost is None:
            continue
        pareto_points.append({"engine": entry["name"], "cer": cer, "cost": cost})
    pareto_front_engines = compute_pareto_front(
        pareto_points, objectives=("cer", "cost"),
    )

    pareto_speed_points = []
    for entry in engines_summary:
        cer = entry.get("cer")
        dur = entry.get("mean_duration_seconds")
        if cer is None or dur is None:
            continue
        pareto_speed_points.append({"engine": entry["name"], "cer": cer, "dur": dur})
    pareto_front_speed = compute_pareto_front(
        pareto_speed_points, objectives=("cer", "dur"),
    )

    pareto_co2_points = []
    for entry in engines_summary:
        cer = entry.get("cer")
        co2 = (entry.get("cost") or {}).get("co2_per_1k_pages_g")
        if cer is None or co2 is None:
            continue
        pareto_co2_points.append({"engine": entry["name"], "cer": cer, "co2": co2})
    pareto_front_co2 = compute_pareto_front(
        pareto_co2_points, objectives=("cer", "co2"),
    )

    return {
        "cost": {
            "points": pareto_points,
            "front": pareto_front_engines,
            "axis_label": "Coût (€ / 1000 pages)",
        },
        "speed": {
            "points": pareto_speed_points,
            "front": pareto_front_speed,
            "axis_label": "Temps moyen (s / page)",
        },
        "co2": {
            "points": pareto_co2_points,
            "front": pareto_front_co2,
            "axis_label": (
                "Empreinte carbone (g CO₂ / 1000 pages, expérimental)"
            ),
        },
        "pricing_meta": {
            "last_updated": pricing_defaults.last_updated,
            "currency": pricing_defaults.currency,
            "hourly_rate_local_cpu_eur": pricing_defaults.hourly_rate_local_cpu_eur,
            "hourly_rate_local_gpu_eur": pricing_defaults.hourly_rate_local_gpu_eur,
            "grid_intensity_local": pricing_defaults.grid_intensity_local,
            "grid_intensity_cloud": pricing_defaults.grid_intensity_cloud,
        },
    }


__all__ = ["attach_engine_costs", "build_pareto_section"]
