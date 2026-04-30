"""Détecteurs narratifs orientés *classement* (chantier 5).

5 détecteurs déplacés depuis ``narrative/detectors.py`` :

- :func:`detect_global_leader_cer`     (Sprint 4)
- :func:`detect_statistical_tie`       (Sprint 18)
- :func:`detect_significant_gap`       (Sprint 4)
- :func:`detect_speed_winner`          (Sprint 4)
- :func:`detect_median_mean_gap_warning` (Sprint 44)

Comportement et signature inchangés. Tous restent enregistrés
automatiquement via ``@register_detector`` à l'import.
"""

from __future__ import annotations

import statistics as _stats
from typing import Optional

from picarones.core.narrative.facts import Fact, FactImportance, FactType
from picarones.core.narrative.registry import register_detector

from picarones.core.narrative.detectors._helpers import (
    _engine_by_name,
    _engines_summary,
    _n_docs,
)


@register_detector(
    FactType.GLOBAL_LEADER_CER,
    priority=10,
    importance=FactImportance.CRITICAL,
)
def detect_global_leader_cer(benchmark_data: dict) -> list[Fact]:
    """Moteur avec le CER moyen le plus bas sur l'ensemble du corpus.

    Émet un Fact CRITICAL si au moins 2 moteurs sont comparés, en attachant
    aussi le 2ᵉ pour permettre à l'arbitre de fusionner avec ``significant_gap``.
    """
    ranking = benchmark_data.get("ranking") or []
    # Éliminer les entrées sans CER calculé
    valid = [r for r in ranking if r.get("mean_cer") is not None]
    if len(valid) < 1:
        return []

    leader = valid[0]
    runner_up = valid[1] if len(valid) >= 2 else None

    payload = {
        "engine": leader["engine"],
        "cer": float(leader["mean_cer"]),
        "cer_pct": round(float(leader["mean_cer"]) * 100, 2),
        "n_engines": len(valid),
        "n_docs": _n_docs(benchmark_data),
    }
    if runner_up is not None:
        payload["runner_up"] = runner_up["engine"]
        payload["runner_up_cer"] = float(runner_up["mean_cer"])
        payload["runner_up_cer_pct"] = round(float(runner_up["mean_cer"]) * 100, 2)

    return [Fact(
        type=FactType.GLOBAL_LEADER_CER,
        importance=FactImportance.CRITICAL,
        payload=payload,
        engines_involved=(leader["engine"],),
    )]


@register_detector(
    FactType.STATISTICAL_TIE,
    priority=20,
    importance=FactImportance.CRITICAL,
)
def detect_statistical_tie(benchmark_data: dict) -> list[Fact]:
    """Groupes de moteurs statistiquement indiscernables (Nemenyi)."""
    nemenyi = benchmark_data.get("statistics", {}).get("nemenyi", {})
    if not nemenyi or nemenyi.get("error"):
        return []

    tied_groups = nemenyi.get("tied_groups", [])
    mean_ranks = nemenyi.get("mean_ranks", {})
    cd = nemenyi.get("critical_distance", 0.0)
    alpha = nemenyi.get("alpha", 0.05)
    n_blocks = nemenyi.get("n_blocks", 0)

    facts: list[Fact] = []
    for group in tied_groups:
        if len(group) < 2:
            continue
        is_leader_tie = min(mean_ranks.get(n, 999) for n in group) == min(
            mean_ranks.values(), default=0
        )
        importance = FactImportance.CRITICAL if is_leader_tie else FactImportance.HIGH

        facts.append(Fact(
            type=FactType.STATISTICAL_TIE,
            importance=importance,
            payload={
                "engines": list(group),
                "engines_list": ", ".join(group),
                "mean_ranks": {n: mean_ranks.get(n) for n in group},
                "critical_distance": round(cd, 3),
                "alpha": alpha,
                "n_blocks": n_blocks,
                "includes_leader": is_leader_tie,
                "n_tied": len(group),
            },
            engines_involved=tuple(group),
        ))
    return facts


@register_detector(
    FactType.SIGNIFICANT_GAP,
    priority=30,
    importance=FactImportance.HIGH,
)
def detect_significant_gap(benchmark_data: dict) -> list[Fact]:
    """Écart statistiquement significatif entre le 1ᵉʳ et le 2ᵉ du classement.

    Lit la matrice de Wilcoxon pairwise et vérifie si la paire (leader,
    runner-up) y apparaît avec ``significant = True``.
    """
    ranking = benchmark_data.get("ranking") or []
    valid = [r for r in ranking if r.get("mean_cer") is not None]
    if len(valid) < 2:
        return []

    leader = valid[0]["engine"]
    runner_up = valid[1]["engine"]

    pairwise = benchmark_data.get("statistics", {}).get("pairwise_wilcoxon") or []
    match = None
    for p in pairwise:
        names = {p.get("engine_a"), p.get("engine_b")}
        if names == {leader, runner_up}:
            match = p
            break
    if match is None:
        return []

    if not match.get("significant"):
        return []  # pas d'écart significatif — rien à signaler ici

    delta_cer = abs(float(valid[0]["mean_cer"]) - float(valid[1]["mean_cer"]))
    return [Fact(
        type=FactType.SIGNIFICANT_GAP,
        importance=FactImportance.CRITICAL,
        payload={
            "leader": leader,
            "runner_up": runner_up,
            "p_value": float(match.get("p_value", 0.0)),
            "delta_cer": round(delta_cer, 4),
            "delta_cer_pct": round(delta_cer * 100, 2),
            "n_pairs": int(match.get("n_pairs", 0)),
        },
        engines_involved=(leader, runner_up),
    )]


@register_detector(
    FactType.SPEED_WINNER,
    priority=100,
    importance=FactImportance.MEDIUM,
)
def detect_speed_winner(benchmark_data: dict) -> list[Fact]:
    """Moteur significativement plus rapide pour une qualité comparable.

    Déclenché si un moteur est au moins 3× plus rapide que la médiane ET que
    son CER n'est pas significativement pire (dans le même groupe Nemenyi que
    le leader OU CER ≤ 1,1 × CER du leader).
    """
    durations = _mean_duration_per_engine(benchmark_data)
    if len(durations) < 2:
        return []

    values = list(durations.values())
    median_dur = _stats.median(values)
    if median_dur <= 0:
        return []

    ranking = benchmark_data.get("ranking") or []
    valid = [r for r in ranking if r.get("mean_cer") is not None]
    if not valid:
        return []
    leader_cer = float(valid[0]["mean_cer"])
    quality_ceiling = max(0.01, leader_cer * 1.10)

    tied_groups = benchmark_data.get("statistics", {}).get("nemenyi", {}).get("tied_groups") or []
    leader_group: set[str] = set()
    for g in tied_groups:
        if valid[0]["engine"] in g:
            leader_group = set(g)
            break

    facts: list[Fact] = []
    candidates = sorted(durations.items(), key=lambda kv: kv[1])
    for engine, dur in candidates:
        if dur * 3.0 > median_dur:
            break  # les suivants sont encore plus lents
        summary = _engine_by_name(benchmark_data, engine) or {}
        engine_cer = summary.get("cer")
        if engine_cer is None:
            continue
        acceptable_quality = (
            engine in leader_group or float(engine_cer) <= quality_ceiling
        )
        if not acceptable_quality:
            continue
        facts.append(Fact(
            type=FactType.SPEED_WINNER,
            importance=FactImportance.MEDIUM,
            payload={
                "engine": engine,
                "mean_duration": round(dur, 3),
                "median_duration": round(median_dur, 3),
                "speedup": round(median_dur / dur, 1) if dur > 0 else None,
                "cer": round(float(engine_cer), 4),
                "cer_pct": round(float(engine_cer) * 100, 2),
            },
            engines_involved=(engine,),
        ))
    return facts[:1]  # seulement le plus rapide — éviter le bruit


@register_detector(
    FactType.MEDIAN_MEAN_GAP_WARNING,
    priority=140,
    importance=FactImportance.MEDIUM,
)
def detect_median_mean_gap_warning(benchmark_data: dict) -> list[Fact]:
    """Avertit quand le ratio ``|moyenne - médiane| / médiane`` du leader
    dépasse 30 %, ce qui indique une distribution fortement asymétrique
    où la moyenne masque les performances réelles.

    Sprint 44 — A.I.2 du plan d'évolution. Cohérent avec le passage du
    tri par défaut sur la médiane : si la moyenne du leader diverge
    fortement de la médiane, l'utilisateur doit le savoir pour
    interpréter correctement les chiffres.
    """
    ranking = benchmark_data.get("ranking") or []
    valid = [
        r for r in ranking
        if r.get("median_cer") is not None
        and r.get("mean_cer") is not None
    ]
    if not valid:
        return []

    leader = valid[0]
    median_cer = float(leader["median_cer"])
    mean_cer = float(leader["mean_cer"])

    if median_cer <= 0:
        # Médiane nulle (corpus très facile pour ce moteur) — l'écart
        # relatif n'est pas calculable de manière utile, on s'abstient.
        return []

    relative_gap = abs(mean_cer - median_cer) / median_cer
    if relative_gap < 0.30:
        return []

    importance = (
        FactImportance.HIGH if relative_gap >= 1.0 else FactImportance.MEDIUM
    )

    return [Fact(
        type=FactType.MEDIAN_MEAN_GAP_WARNING,
        importance=importance,
        payload={
            "engine": leader["engine"],
            "median_cer_pct": round(median_cer * 100, 2),
            "mean_cer_pct": round(mean_cer * 100, 2),
            "relative_gap_pct": round(relative_gap * 100, 1),
            "n_docs": int(leader.get("documents") or 0),
        },
        engines_involved=(leader["engine"],),
    )]
