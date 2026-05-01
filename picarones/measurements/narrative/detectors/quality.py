"""Détecteurs narratifs liés à la *qualité texte / fiabilité* (chantier 5).

4 détecteurs déplacés depuis ``narrative/detectors.py`` :

- :func:`detect_error_profile_outlier`  (Sprint 4)
- :func:`detect_llm_hallucination_flag` (Sprint 4)
- :func:`detect_robustness_fragile`     (Sprint 4)
- :func:`detect_confidence_warning`     (Sprint 4)
"""

from __future__ import annotations

import statistics as _stats

from picarones.measurements.narrative.facts import Fact, FactImportance, FactType
from picarones.measurements.narrative.registry import register_detector

from picarones.measurements.narrative.detectors._helpers import (
    _engines_summary,
)


@register_detector(
    FactType.ERROR_PROFILE_OUTLIER,
    priority=60,
    importance=FactImportance.MEDIUM,
)
def detect_error_profile_outlier(benchmark_data: dict) -> list[Fact]:
    """Moteur au profil taxonomique atypique.

    Émet un Fact si, pour un moteur et une classe d'erreur, la part relative
    est au moins 2× plus élevée que la médiane des autres moteurs (et > 15 %
    du total pour éviter les strates marginales).
    """
    engines = _engines_summary(benchmark_data)
    # {engine: {class_name: proportion}}
    profiles: dict[str, dict[str, float]] = {}
    for e in engines:
        tax = e.get("aggregated_taxonomy") or {}
        distribution = tax.get("distribution") or tax.get("proportions") or {}
        if not distribution:
            continue
        profiles[e["name"]] = {k: float(v) for k, v in distribution.items()}
    if len(profiles) < 2:
        return []

    # Collecter toutes les classes rencontrées
    all_classes: set[str] = set()
    for p in profiles.values():
        all_classes.update(p.keys())

    facts: list[Fact] = []
    for cls in all_classes:
        values = [(name, p.get(cls, 0.0)) for name, p in profiles.items()]
        props = [v for _, v in values]
        if not props:
            continue
        median_prop = _stats.median(props)
        for name, v in values:
            if v < 0.15:  # trop marginal pour être notable
                continue
            if median_prop <= 0:
                continue
            if v >= 2.0 * median_prop:
                facts.append(Fact(
                    type=FactType.ERROR_PROFILE_OUTLIER,
                    importance=FactImportance.HIGH,
                    payload={
                        "engine": name,
                        "error_class": cls,
                        "proportion": round(v, 4),
                        "proportion_pct": round(v * 100, 1),
                        "median_proportion": round(median_prop, 4),
                        "median_proportion_pct": round(median_prop * 100, 1),
                        "ratio_to_median": round(v / median_prop, 2) if median_prop else None,
                    },
                    engines_involved=(name,),
                ))
    return facts


@register_detector(
    FactType.LLM_HALLUCINATION_FLAG,
    priority=70,
    importance=FactImportance.HIGH,
)
def detect_llm_hallucination_flag(benchmark_data: dict) -> list[Fact]:
    """LLM/VLM au taux d'hallucination notablement élevé.

    Déclenché si ``hallucinating_doc_rate`` > 30 % OU ``anchor_score_mean`` < 0,6
    pour un moteur dont le champ ``is_pipeline`` ou ``is_vlm`` est ``True``.
    """
    facts: list[Fact] = []
    for e in _engines_summary(benchmark_data):
        agg = e.get("aggregated_hallucination") or {}
        if not agg:
            continue
        rate = agg.get("hallucinating_doc_rate")
        anchor = agg.get("anchor_score_mean")
        length_ratio = agg.get("length_ratio_mean")
        # Signal seulement si c'est un pipeline LLM ou un VLM
        is_llm = bool(e.get("is_pipeline")) or bool(e.get("is_vlm"))
        if not is_llm:
            continue

        flagged = False
        reasons = []
        if rate is not None and float(rate) > 0.30:
            flagged = True
            reasons.append("taux de documents hallucinés")
        if anchor is not None and float(anchor) < 0.60:
            flagged = True
            reasons.append("ancrage faible")
        if length_ratio is not None and float(length_ratio) > 1.30:
            flagged = True
            reasons.append("sortie anormalement longue")
        if not flagged:
            continue

        facts.append(Fact(
            type=FactType.LLM_HALLUCINATION_FLAG,
            importance=FactImportance.HIGH,
            payload={
                "engine": e["name"],
                "hallucinating_rate": round(float(rate or 0.0), 4),
                "hallucinating_rate_pct": round(float(rate or 0.0) * 100, 1),
                "anchor_score": round(float(anchor), 3) if anchor is not None else None,
                "length_ratio": round(float(length_ratio), 3) if length_ratio is not None else None,
                "reasons": reasons,
                "reasons_list": ", ".join(reasons),
            },
            engines_involved=(e["name"],),
        ))
    return facts


@register_detector(
    FactType.ROBUSTNESS_FRAGILE,
    priority=80,
    importance=FactImportance.MEDIUM,
)
def detect_robustness_fragile(benchmark_data: dict) -> list[Fact]:
    """Moteur qui dégrade fortement au-dessus d'un seuil de bruit/flou.

    Activé si les données de robustesse sont embarquées dans
    ``benchmark_data["robustness"]`` (hors scope du benchmark classique,
    produit par ``picarones robustness`` et injecté optionnellement).
    """
    robustness = benchmark_data.get("robustness")
    if not robustness:
        return []

    facts: list[Fact] = []
    curves = robustness.get("curves") or robustness.get("engines") or []
    # Structure attendue : [{engine, degradation_type, points: [{level, cer}]}]
    # Flag : CER à niveau max > 3× CER au niveau min.
    for entry in curves:
        engine = entry.get("engine")
        dtype = entry.get("degradation_type")
        points = entry.get("points") or []
        if not engine or not points or len(points) < 2:
            continue
        try:
            sorted_pts = sorted(points, key=lambda p: float(p["level"]))
        except (KeyError, TypeError, ValueError):
            continue
        first, last = sorted_pts[0], sorted_pts[-1]
        c0 = float(first.get("cer") or 0.0)
        c1 = float(last.get("cer") or 0.0)
        if c0 <= 0.01:  # éviter division par quasi-zéro
            continue
        if c1 >= 3.0 * c0 and c1 > 0.15:
            facts.append(Fact(
                type=FactType.ROBUSTNESS_FRAGILE,
                importance=FactImportance.HIGH,
                payload={
                    "engine": engine,
                    "degradation": dtype,
                    "cer_baseline": round(c0, 4),
                    "cer_baseline_pct": round(c0 * 100, 1),
                    "cer_degraded": round(c1, 4),
                    "cer_degraded_pct": round(c1 * 100, 1),
                    "ratio": round(c1 / c0, 1),
                    "level_max": float(last.get("level") or 0),
                },
                engines_involved=(engine,),
            ))
    return facts


@register_detector(
    FactType.CONFIDENCE_WARNING,
    priority=120,
    importance=FactImportance.MEDIUM,
)
def detect_confidence_warning(benchmark_data: dict) -> list[Fact]:
    """Intervalle de confiance large → classement peu fiable.

    Déclenché si, pour le leader ou le runner-up, la largeur de l'IC 95 %
    est plus du triple de l'écart |leader − runner-up| OU > 5 points de CER.
    """
    stats = benchmark_data.get("statistics", {}) or {}
    cis = stats.get("bootstrap_cis") or []
    if len(cis) < 2:
        return []

    ranking = benchmark_data.get("ranking") or []
    valid = [r for r in ranking if r.get("mean_cer") is not None]
    if len(valid) < 2:
        return []

    by_name = {c["engine"]: c for c in cis if "engine" in c}
    leader = valid[0]["engine"]
    runner_up = valid[1]["engine"]
    leader_ci = by_name.get(leader)
    runner_ci = by_name.get(runner_up)
    if not leader_ci or not runner_ci:
        return []

    gap = abs(float(valid[0]["mean_cer"]) - float(valid[1]["mean_cer"]))
    facts: list[Fact] = []
    for engine_name, ci in ((leader, leader_ci), (runner_up, runner_ci)):
        lo = float(ci.get("ci_lower") or 0.0)
        hi = float(ci.get("ci_upper") or 0.0)
        width = hi - lo
        wide_vs_gap = gap > 0 and width > 3.0 * gap
        wide_absolute = width > 0.05
        if wide_vs_gap or wide_absolute:
            facts.append(Fact(
                type=FactType.CONFIDENCE_WARNING,
                importance=FactImportance.MEDIUM,
                payload={
                    "engine": engine_name,
                    "ci_lower": round(lo, 4),
                    "ci_upper": round(hi, 4),
                    "ci_width": round(width, 4),
                    "ci_width_pct": round(width * 100, 2),
                    "mean_cer": round(float(ci.get("mean") or 0.0), 4),
                    "mean_cer_pct": round(float(ci.get("mean") or 0.0) * 100, 2),
                    "gap_to_runner_up_pct": round(gap * 100, 2),
                    # Niveau de confiance des bornes — propagé pour traçabilité
                    # anti-hallucination (le template ne hardcode plus "95 %").
                    "confidence_level": 95,
                },
                engines_involved=(engine_name,),
            ))
            break  # un seul avertissement suffit
    return facts
