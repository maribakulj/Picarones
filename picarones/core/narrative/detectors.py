"""Détecteurs de faits — implémentations Sprint 4.

Chaque détecteur est une fonction pure ``(benchmark_data: dict) -> list[Fact]``.
Convention : un détecteur qui ne trouve rien retourne une liste vide. Il ne
doit jamais lever d'exception — la gestion d'erreur est centralisée dans
``DetectorRegistry.run``.

Règle anti-hallucination : chaque nombre ou nom placé dans ``payload`` doit
venir directement du JSON d'entrée (jamais d'une interpolation). Les tests
du Sprint 4 parsent la synthèse rendue et vérifient que chaque valeur
numérique qu'elle contient est traçable.
"""

from __future__ import annotations

import statistics as _stats
from typing import Optional

from picarones.core.narrative.facts import Fact, FactImportance, FactType
from picarones.core.narrative.registry import register_detector


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------

def _engines_summary(data: dict) -> list[dict]:
    """Accès normalisé à la liste des résumés moteur."""
    return data.get("engines", []) or []


def _engine_by_name(data: dict, name: str) -> Optional[dict]:
    for e in _engines_summary(data):
        if e.get("name") == name:
            return e
    return None


def _n_docs(data: dict) -> int:
    meta = data.get("meta", {}) or {}
    return int(meta.get("document_count") or 0)


# ---------------------------------------------------------------------------
# Sprint 4 — Détecteurs implémentés
# ---------------------------------------------------------------------------

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


def _mean_duration_per_engine(benchmark_data: dict) -> dict[str, float]:
    """Durée moyenne d'exécution par moteur (en secondes par document)."""
    durations: dict[str, list[float]] = {}
    for doc in benchmark_data.get("documents") or []:
        for er in doc.get("engine_results") or []:
            d = er.get("duration")
            if d is None:
                continue
            durations.setdefault(er["engine"], []).append(float(d))
    return {k: sum(v) / len(v) for k, v in durations.items() if v}


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


# ---------------------------------------------------------------------------
# Détecteur Sprint 44 — distribution asymétrique (médiane vs moyenne)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Détecteur Sprint 46 — stratification recommandée (corpus hétérogène)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Détecteur Sprint 73 — moteur hors baseline historique (A.I.3)
# ---------------------------------------------------------------------------

@register_detector(
    FactType.ENGINE_OFF_BASELINE,
    priority=150,
    importance=FactImportance.MEDIUM,
)
def detect_engine_off_baseline(benchmark_data: dict) -> list[Fact]:
    """Émet un Fact pour chaque moteur dont le CER courant s'écarte
    significativement de sa moyenne historique sur le **même corpus**.

    Lit ``benchmark_data["baseline_comparisons"]`` (liste de dicts
    produits par ``compute_engine_baseline`` du module
    ``baseline_comparison`` Sprint 73).  Si la clé est absente ou
    vide, le détecteur reste silencieux — typiquement le cas quand
    aucun historique SQLite n'a été chargé.

    Garde-fous :

    - Si ``n_runs < 5`` (déjà filtré par ``compute_engine_baseline``
      qui retourne ``None`` dans ce cas).
    - Si ``relative_delta`` n'est pas calculable (baseline = 0).
    - Importance ``HIGH`` si ``|relative_delta| ≥ 50 %``, sinon
      ``MEDIUM``.
    """
    comparisons = benchmark_data.get("baseline_comparisons") or []
    if not isinstance(comparisons, (list, tuple)):
        return []
    facts: list[Fact] = []
    for comp in comparisons:
        if not isinstance(comp, dict):
            continue
        if not comp.get("off_baseline"):
            continue
        rel = comp.get("relative_delta")
        if rel is None:
            continue
        engine = comp.get("engine_name")
        cer_current = comp.get("cer_current")
        cer_hist_mean = comp.get("cer_historical_mean")
        n_runs = comp.get("n_runs")
        if engine is None or cer_current is None or cer_hist_mean is None:
            continue
        importance = (
            FactImportance.HIGH if abs(float(rel)) >= 0.50
            else FactImportance.MEDIUM
        )
        facts.append(Fact(
            type=FactType.ENGINE_OFF_BASELINE,
            importance=importance,
            payload={
                "engine": engine,
                "cer_current_pct": round(float(cer_current) * 100, 2),
                "cer_historical_mean_pct": round(
                    float(cer_hist_mean) * 100, 2,
                ),
                "n_runs": int(n_runs or 0),
                "relative_delta_pct": round(float(rel) * 100, 1),
                "direction": "higher" if float(rel) > 0 else "lower",
            },
            engines_involved=(engine,),
        ))
    return facts


# ---------------------------------------------------------------------------
# Détecteur Sprint 90 — moteur instable multi-runs (A.II.4)
# ---------------------------------------------------------------------------

@register_detector(
    FactType.ENGINE_UNSTABLE,
    priority=160,
    importance=FactImportance.HIGH,
)
def detect_engine_unstable(benchmark_data: dict) -> list[Fact]:
    """Émet un Fact pour chaque moteur dont la stabilité multi-runs
    est insuffisante (Sprint 83 + 90).

    Lit ``benchmark_data["multirun_stability"]`` : liste de dicts
    avec ``engine_name`` + champs de ``compute_multirun_stability``
    (cer_cv, identical_run_rate, n_runs, etc.).  Si la clé est
    absente ou vide, le détecteur reste silencieux — typiquement
    le cas quand l'utilisateur n'a pas exécuté `--repeats N`.

    Garde-fous :

    - ``n_runs ≥ 2`` (déjà filtré par
      ``compute_multirun_stability`` qui retourne ``None``).
    - Déclenche si ``cer_cv > 0.10`` (variance relative > 10 % du
      CER moyen) **ou** ``identical_run_rate < 0.50`` (moins
      d'une paire de runs sur deux est identique).
    - Importance ``HIGH`` (l'instabilité discrédite les
      conclusions).
    """
    stabilities = benchmark_data.get("multirun_stability") or []
    if not isinstance(stabilities, (list, tuple)):
        return []
    facts: list[Fact] = []
    for stab in stabilities:
        if not isinstance(stab, dict):
            continue
        engine = stab.get("engine_name") or stab.get("engine")
        if not engine:
            continue
        n_runs = stab.get("n_runs")
        if not isinstance(n_runs, int) or n_runs < 2:
            continue
        cer_cv = stab.get("cer_cv")
        identical_rate = stab.get("identical_run_rate")
        # Critères de déclenchement
        cv_high = (
            isinstance(cer_cv, (int, float)) and float(cer_cv) > 0.10
        )
        runs_diverge = (
            isinstance(identical_rate, (int, float))
            and float(identical_rate) < 0.50
        )
        if not (cv_high or runs_diverge):
            continue
        payload: dict = {
            "engine": engine,
            "n_runs": int(n_runs),
        }
        if isinstance(cer_cv, (int, float)):
            payload["cer_cv"] = float(cer_cv)
            payload["cer_cv_pct"] = round(float(cer_cv) * 100, 1)
        if isinstance(identical_rate, (int, float)):
            payload["identical_run_rate"] = float(identical_rate)
            payload["identical_run_rate_pct"] = round(
                float(identical_rate) * 100, 1,
            )
        # Champs additionnels pour la phrase de synthèse
        cer_mean = stab.get("cer_mean")
        cer_stdev = stab.get("cer_stdev")
        if isinstance(cer_mean, (int, float)):
            payload["cer_mean_pct"] = round(float(cer_mean) * 100, 2)
        if isinstance(cer_stdev, (int, float)):
            payload["cer_stdev_pct"] = round(float(cer_stdev) * 100, 2)
        n_distinct = stab.get("n_distinct_outputs")
        if isinstance(n_distinct, int):
            payload["n_distinct_outputs"] = int(n_distinct)
        facts.append(Fact(
            type=FactType.ENGINE_UNSTABLE,
            importance=FactImportance.HIGH,
            payload=payload,
            engines_involved=(engine,),
        ))
    return facts


# ---------------------------------------------------------------------------
# Détecteur Sprint 92 — régression dans l'historique (A.II.9)
# ---------------------------------------------------------------------------

@register_detector(
    FactType.REGRESSION_IN_HISTORY,
    priority=170,
    importance=FactImportance.MEDIUM,
)
def detect_regression_in_history(benchmark_data: dict) -> list[Fact]:
    """Émet un Fact pour chaque moteur dont l'historique montre
    une dégradation : pente positive significative ou rupture
    brutale (Sprint 92).

    Lit ``benchmark_data["longitudinal_trends"]`` : liste de
    dicts produits par ``compute_corpus_longitudinal`` du module
    ``longitudinal``.  Si la clé est absente ou vide, le
    détecteur reste silencieux — typiquement le cas quand
    aucun historique n'a été chargé ou que la série est trop
    courte.

    Garde-fous :

    - ``n_runs ≥ 3`` (déjà filtré par
      ``compute_engine_longitudinal``).
    - Déclenche si **soit** ``trend.slope`` traduit une
      régression d'au moins ``slope_threshold`` (en CER/jour,
      défaut équivalent à +1 point CER sur 365 jours), **soit**
      ``change_point.delta > change_threshold`` (défaut
      0.01 = +1 point de CER d'un segment à l'autre).
    - Importance ``HIGH`` si la dégradation cumulée
      ``absolute_delta`` ≥ 5 points de CER.
    """
    trends = benchmark_data.get("longitudinal_trends") or []
    if not isinstance(trends, (list, tuple)):
        return []
    slope_threshold = (
        0.01 / 365.0  # +1 point de CER sur 365 jours minimum
    )
    change_threshold = 0.01
    facts: list[Fact] = []
    for entry in trends:
        if not isinstance(entry, dict):
            continue
        engine = entry.get("engine_name")
        if not engine:
            continue
        n_runs = entry.get("n_runs")
        if not isinstance(n_runs, int) or n_runs < 3:
            continue
        trend = entry.get("trend") or {}
        cp = entry.get("change_point")
        slope = trend.get("slope")
        slope_high = (
            isinstance(slope, (int, float))
            and float(slope) > slope_threshold
        )
        cp_high = (
            isinstance(cp, dict)
            and isinstance(cp.get("delta"), (int, float))
            and float(cp["delta"]) > change_threshold
        )
        if not (slope_high or cp_high):
            continue
        absolute_delta = entry.get("absolute_delta") or 0.0
        importance = (
            FactImportance.HIGH
            if isinstance(absolute_delta, (int, float))
            and abs(float(absolute_delta)) >= 0.05
            else FactImportance.MEDIUM
        )
        payload: dict = {
            "engine": engine,
            "n_runs": int(n_runs),
            "absolute_delta_pct": round(
                float(absolute_delta) * 100, 2,
            ) if isinstance(absolute_delta, (int, float)) else 0.0,
            "first_cer_pct": round(
                float(entry.get("first_cer") or 0.0) * 100, 2,
            ),
            "last_cer_pct": round(
                float(entry.get("last_cer") or 0.0) * 100, 2,
            ),
        }
        if slope_high:
            payload["slope_per_year_pct"] = round(
                float(slope) * 365 * 100, 2,
            )
            payload["r_squared"] = round(
                float(trend.get("r_squared") or 0.0), 3,
            )
            payload["pattern"] = "trend"
        if cp_high:
            payload["change_point_timestamp"] = str(
                cp.get("timestamp") or "?",
            )
            payload["change_delta_pct"] = round(
                float(cp["delta"]) * 100, 2,
            )
            payload["mean_before_pct"] = round(
                float(cp.get("mean_before") or 0.0) * 100, 2,
            )
            payload["mean_after_pct"] = round(
                float(cp.get("mean_after") or 0.0) * 100, 2,
            )
            # Si on a aussi une rupture, le pattern domine
            payload["pattern"] = (
                "trend_and_change_point" if slope_high else "change_point"
            )
        facts.append(Fact(
            type=FactType.REGRESSION_IN_HISTORY,
            importance=importance,
            payload=payload,
            engines_involved=(engine,),
        ))
    return facts


# ---------------------------------------------------------------------------
# Détecteur Sprint 36 — opportunité d'ensemble (complémentarité)
# ---------------------------------------------------------------------------

@register_detector(
    FactType.ENSEMBLE_OPPORTUNITY,
    priority=130,
    importance=FactImportance.MEDIUM,
)
def detect_ensemble_opportunity(benchmark_data: dict) -> list[Fact]:
    """Deux moteurs très complémentaires : un voting majoritaire entre eux
    pourrait améliorer significativement le CER token-level.

    Lit la structure ``inter_engine_analysis`` produite par le runner
    (Sprint 35-36) et déclenche si la fraction d'erreurs du meilleur
    moteur récupérable par un ensemble dépasse 25 %.

    L'importance monte à ``HIGH`` quand le gap relatif dépasse 50 %
    (ensemble franchement profitable) — sinon reste à ``MEDIUM``.
    """
    iea = benchmark_data.get("inter_engine_analysis") or {}
    comp = iea.get("complementarity") or {}
    if not comp:
        return []

    relative_gap = float(comp.get("relative_gap") or 0.0)
    if relative_gap < 0.25:
        # En deçà de 25 %, l'ensemble n'apporterait quasi rien — on ne
        # remonte pas le fait pour ne pas bruiter la synthèse.
        return []

    best_engine = comp.get("best_engine") or ""
    if not best_engine:
        return []

    payload: dict = {
        "best_engine": best_engine,
        "best_recall_pct": round(float(comp.get("best_single_recall") or 0.0) * 100, 2),
        "oracle_recall_pct": round(float(comp.get("oracle_recall") or 0.0) * 100, 2),
        "absolute_gap_pct": round(float(comp.get("absolute_gap") or 0.0) * 100, 2),
        "relative_gap_pct": round(relative_gap * 100, 1),
        "doc_count": int(comp.get("doc_count") or 0),
    }

    # Paire la plus complémentaire — la divergence taxonomique, quand
    # disponible, fournit deux moteurs « candidats naturels ».  Sinon on
    # tombe sur le best + le second-best en recall individuel.
    div = iea.get("taxonomy_divergence") or {}
    pair = div.get("max_pair") or []
    pair_a = ""
    pair_b = ""
    divergence_value: Optional[float] = None
    if pair and len(pair) >= 3 and isinstance(pair[2], (int, float)) and pair[2] > 0:
        pair_a, pair_b, divergence_value = str(pair[0]), str(pair[1]), float(pair[2])
    else:
        # Fallback : best engine + second-best engine par recall individuel
        per_engine = comp.get("per_engine_recall") or {}
        if len(per_engine) >= 2:
            ranked = sorted(per_engine.items(), key=lambda kv: kv[1], reverse=True)
            pair_a, pair_b = ranked[0][0], ranked[1][0]

    payload["pair_a"] = pair_a
    payload["pair_b"] = pair_b
    payload["divergence"] = round(divergence_value, 3) if divergence_value is not None else 0.0
    payload["divergence_metric"] = (div.get("metric") or "js")

    importance = (
        FactImportance.HIGH if relative_gap >= 0.5 else FactImportance.MEDIUM
    )
    engines_involved: tuple[str, ...] = (
        (pair_a, pair_b) if pair_a and pair_b else (best_engine,)
    )
    return [Fact(
        type=FactType.ENSEMBLE_OPPORTUNITY,
        importance=importance,
        payload=payload,
        engines_involved=engines_involved,
    )]


# ---------------------------------------------------------------------------
# Enregistrement par défaut — Sprint 29
# ---------------------------------------------------------------------------
#
# Depuis Sprint 29, l'enregistrement passe par ``@register_detector``
# directement sur la définition de chaque fonction (cf. ``registry.py``).
# ``DETECTORS_BY_TYPE`` reste exposé en tant qu'**alias dérivé** pour les
# consommateurs externes qui s'appuient sur le mapping historique
# ``{FactType: callable}``.

from picarones.core.narrative.facts import DetectorFn  # noqa: E402, F401
from picarones.core.narrative.registry import (  # noqa: E402
    iter_detectors as _iter_detectors,
    populate_legacy_registry as _populate_legacy_registry,
)


def _build_detectors_by_type() -> dict[FactType, DetectorFn]:
    """Snapshot du registre déclaratif vers un dict ``{type: fn}``."""
    return {entry.fact_type: entry.fn for entry in _iter_detectors()}


# Vue figée à l'import — utile pour les tests qui parcourent les types
# enregistrés sans instancier un ``DetectorRegistry``.
DETECTORS_BY_TYPE = _build_detectors_by_type()


def register_default_detectors(registry) -> None:
    """Enregistre les détecteurs du registre déclaratif dans un
    ``DetectorRegistry`` historique.

    Sprint 29 : la source de vérité est maintenant le décorateur
    ``@register_detector`` ; cette fonction se contente de pousser
    le contenu du registre vers l'objet ``DetectorRegistry`` que les
    consommateurs externes (``DetectorRegistry.run``) instancient.
    """
    _populate_legacy_registry(registry)
