"""Front Pareto coût/qualité (Sprint 19).

Construit trois fronts Pareto avec des axes alternatifs :

- ``cost`` — CER vs coût € / 1000 pages.
- ``speed`` — CER vs durée moyenne par page.
- ``co2`` — CER vs empreinte carbone (g CO₂ / 1000 pages, expérimental).

**Effet de bord** : :func:`build_pareto_section` enrichit en place
le ``engines_summary`` reçu en argument avec les champs
``mean_duration_seconds`` et ``cost`` (coût par 1000 pages + détail
de pricing). Cette responsabilité partagée est documentée dans le
module ``__init__.py`` du sous-package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from picarones.measurements.pricing import (
    build_costs_for_benchmark,
    load_pricing_database,
)
from picarones.measurements.statistics import compute_pareto_front

if TYPE_CHECKING:
    from picarones.core.results import BenchmarkResult


def build_pareto_section(
    engines_summary: list[dict], benchmark: "BenchmarkResult",
) -> dict:
    """Construit le bloc ``pareto`` du dict de rapport.

    Annote en place chaque entrée de ``engines_summary`` avec
    ``mean_duration_seconds`` et ``cost``.
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

    pricing_defaults, _ = load_pricing_database()
    costs_by_engine = build_costs_for_benchmark(
        engines_summary, durations_by_engine,
    )
    # Annoter en place chaque résumé moteur avec son coût et sa durée.
    for entry in engines_summary:
        name = entry["name"]
        entry["mean_duration_seconds"] = (
            round(durations_by_engine.get(name, 0.0), 4)
            if name in durations_by_engine else None
        )
        entry["cost"] = costs_by_engine.get(name)

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


__all__ = ["build_pareto_section"]
