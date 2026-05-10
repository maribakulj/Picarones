"""Détecteurs narratifs liés à la *stratification corpus* (chantier 5).

3 détecteurs + 1 helper (``_stratum_cer_by_engine``) déplacés depuis
``narrative/detectors.py`` :

- :func:`detect_stratum_winner`             (Sprint 4)
- :func:`detect_stratum_collapse`           (Sprint 4)
- :func:`detect_stratification_recommended` (Sprint 45)
"""

from __future__ import annotations


from picarones.domain.facts import Fact, FactImportance, FactType
from picarones.reports.narrative.registry import register_detector

from picarones.reports.narrative.detectors._helpers import (
    _engine_by_name,
)


def _stratum_cer_by_engine(benchmark_data: dict) -> dict[str, dict[str, list[float]]]:
    """Agrège les CER par (moteur, strate).

    Strate = ``document["script_type"]`` si présent. Retourne ``{}`` si aucun
    document n'expose de strate (pas d'émission possible).
    """
    out: dict[str, dict[str, list[float]]] = {}
    for doc in benchmark_data.get("documents") or []:
        stratum = doc.get("script_type")
        if not stratum:
            continue
        for er in doc.get("engine_results") or []:
            if er.get("error"):
                continue
            cer = er.get("cer")
            if cer is None:
                continue
            name = er.get("engine")
            out.setdefault(name, {}).setdefault(stratum, []).append(float(cer))
    return out


@register_detector(
    FactType.STRATUM_WINNER,
    priority=40,
    importance=FactImportance.MEDIUM,
)
def detect_stratum_winner(benchmark_data: dict) -> list[Fact]:
    """Moteur qui domine nettement sur une strate (≥ 3 documents, CER
    au moins 25 % plus bas que le second sur cette strate).
    """
    agg = _stratum_cer_by_engine(benchmark_data)
    if not agg:
        return []

    # Inverser : {stratum: {engine: mean_cer}}
    by_stratum: dict[str, dict[str, float]] = {}
    for engine, strata in agg.items():
        for stratum, vals in strata.items():
            if len(vals) < 3:
                continue
            by_stratum.setdefault(stratum, {})[engine] = sum(vals) / len(vals)

    facts: list[Fact] = []
    for stratum, engine_cer in by_stratum.items():
        if len(engine_cer) < 2:
            continue
        ordered = sorted(engine_cer.items(), key=lambda kv: kv[1])
        best_name, best_cer = ordered[0]
        second_cer = ordered[1][1]
        if second_cer == 0:
            continue
        if best_cer < second_cer * 0.75:  # dominance ≥ 25 %
            facts.append(Fact(
                type=FactType.STRATUM_WINNER,
                importance=FactImportance.HIGH,
                payload={
                    "engine": best_name,
                    "stratum": stratum,
                    "cer": round(best_cer, 4),
                    "cer_pct": round(best_cer * 100, 2),
                    "second_engine": ordered[1][0],
                    "second_cer": round(second_cer, 4),
                    "second_cer_pct": round(second_cer * 100, 2),
                    "n_docs_stratum": len(agg[best_name][stratum]),
                },
                engines_involved=(best_name,),
                stratum=stratum,
            ))
    return facts


@register_detector(
    FactType.STRATUM_COLLAPSE,
    priority=50,
    importance=FactImportance.HIGH,
)
def detect_stratum_collapse(benchmark_data: dict) -> list[Fact]:
    """Moteur globalement compétitif qui s'effondre sur une strate.

    Déclenché si, pour un moteur, le CER moyen sur une strate ≥ 3 documents
    est plus du double du CER global du même moteur.
    """
    agg = _stratum_cer_by_engine(benchmark_data)
    if not agg:
        return []

    facts: list[Fact] = []
    for engine_name, strata in agg.items():
        summary = _engine_by_name(benchmark_data, engine_name) or {}
        global_cer = summary.get("cer")
        if global_cer is None:
            continue
        global_cer = float(global_cer)
        if global_cer <= 0:
            continue
        for stratum, vals in strata.items():
            if len(vals) < 3:
                continue
            local_cer = sum(vals) / len(vals)
            if local_cer > 2.0 * global_cer and (local_cer - global_cer) > 0.05:
                facts.append(Fact(
                    type=FactType.STRATUM_COLLAPSE,
                    importance=FactImportance.HIGH,
                    payload={
                        "engine": engine_name,
                        "stratum": stratum,
                        "local_cer": round(local_cer, 4),
                        "local_cer_pct": round(local_cer * 100, 2),
                        "global_cer": round(global_cer, 4),
                        "global_cer_pct": round(global_cer * 100, 2),
                        "delta_cer_pct": round((local_cer - global_cer) * 100, 2),
                        "n_docs_stratum": len(vals),
                    },
                    engines_involved=(engine_name,),
                    stratum=stratum,
                ))
    return facts


@register_detector(
    FactType.STRATIFICATION_RECOMMENDED,
    priority=45,  # juste après STRATUM_WINNER (40), avant STRATUM_COLLAPSE (50)
    importance=FactImportance.HIGH,
)
def detect_stratification_recommended(benchmark_data: dict) -> list[Fact]:
    """Avertit quand le corpus est hétérogène et que la vue stratifiée
    apporte un éclairage qualitativement différent du classement global.

    Critère : ``corpus_homogeneity.max_inter_strata_gap > 5 points`` de
    CER médian sur le moteur leader.  Au-delà de 10 points, importance
    ``HIGH`` (situation très hétérogène où le seul classement global
    serait trompeur).

    Lit ``benchmark_data["corpus_homogeneity"]`` exposé par
    ``BenchmarkResult.as_dict()`` (Sprint 45).
    """
    homog = benchmark_data.get("corpus_homogeneity")
    if not homog:
        return []

    gap = homog.get("max_inter_strata_gap")
    if gap is None:
        return []

    gap = float(gap)
    if gap < 0.05:
        return []  # 5 points de CER : seuil de pertinence éditoriale

    leader = str(homog.get("leader") or "")
    n_strata = int(homog.get("n_strata") or 0)
    pair = homog.get("leader_max_gap_strata") or ["", ""]
    if len(pair) < 2:
        return []
    min_strat, max_strat = str(pair[0]), str(pair[1])

    leader_per_stratum = homog.get("leader_per_stratum_median") or {}
    min_med = float(leader_per_stratum.get(min_strat, 0.0))
    max_med = float(leader_per_stratum.get(max_strat, 0.0))

    importance = (
        FactImportance.HIGH if gap >= 0.10 else FactImportance.MEDIUM
    )

    return [Fact(
        type=FactType.STRATIFICATION_RECOMMENDED,
        importance=importance,
        payload={
            "leader": leader,
            "n_strata": n_strata,
            "gap_pct": round(gap * 100, 1),
            "min_stratum": min_strat,
            "max_stratum": max_strat,
            "min_stratum_cer_pct": round(min_med * 100, 2),
            "max_stratum_cer_pct": round(max_med * 100, 2),
        },
        engines_involved=(leader,) if leader else (),
    )]
