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
                },
                engines_involved=(engine_name,),
            ))
            break  # un seul avertissement suffit
    return facts


# ---------------------------------------------------------------------------
# Enregistrement par défaut — activé au Sprint 4
# ---------------------------------------------------------------------------

DETECTORS_BY_TYPE = {
    FactType.GLOBAL_LEADER_CER: detect_global_leader_cer,
    FactType.STATISTICAL_TIE: detect_statistical_tie,
    FactType.SIGNIFICANT_GAP: detect_significant_gap,
    FactType.PARETO_ALTERNATIVE: detect_pareto_alternative,
    FactType.STRATUM_WINNER: detect_stratum_winner,
    FactType.STRATUM_COLLAPSE: detect_stratum_collapse,
    FactType.ERROR_PROFILE_OUTLIER: detect_error_profile_outlier,
    FactType.LLM_HALLUCINATION_FLAG: detect_llm_hallucination_flag,
    FactType.ROBUSTNESS_FRAGILE: detect_robustness_fragile,
    FactType.COST_OUTLIER: detect_cost_outlier,
    FactType.SPEED_WINNER: detect_speed_winner,
    FactType.CONFIDENCE_WARNING: detect_confidence_warning,
}


def register_default_detectors(registry) -> None:
    """Enregistre les détecteurs du Sprint 4 dans un ``DetectorRegistry``.

    Les types ``PARETO_ALTERNATIVE`` et ``COST_OUTLIER`` restent des stubs
    jusqu'au Sprint 5 : les enregistrer maintenant ne fait rien de visible
    (liste vide toujours retournée), ce qui est sûr et simplifie le parcours.
    """
    for fact_type, fn in DETECTORS_BY_TYPE.items():
        registry.register(fact_type, fn)
