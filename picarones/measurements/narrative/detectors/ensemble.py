"""Détecteurs narratifs liés à l'*opportunité d'ensemble inter-moteurs* (chantier 5).

1 détecteur déplacé depuis ``narrative/detectors.py`` :

- :func:`detect_ensemble_opportunity` (Sprint 36)
"""

from __future__ import annotations

from typing import Optional

from picarones.measurements.narrative.facts import Fact, FactImportance, FactType
from picarones.measurements.narrative.registry import register_detector



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
