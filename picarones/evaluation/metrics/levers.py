"""Section « Leviers d'amélioration » — Sprint 82 (A.I.9).

A.I.9 du plan d'évolution 2026.

Pourquoi ce module
------------------
Le moteur narratif (Sprint 19) émet des `Fact` qui décrivent **ce
qui s'est passé** dans le benchmark : qui gagne, qui s'effondre,
qui est fragile.  Ce sprint répond à une question
complémentaire : **sur quelle dimension le bénéfice attendu d'une
amélioration serait-il le plus visible ?**

Pas de prescription
-------------------
Picarones est un **outil de recherche**, pas un atelier de
production.  Le module ne dit jamais *« faites X »* ni
*« utilisez le moteur Y »* ; il agrège des **observations
factuelles** déjà calculées dans d'autres modules (Sprints 75-81)
et les présente comme un récapitulatif compact en bas du rapport.
Le chercheur lit, juge et arbitre.

Exemples de leviers émis
------------------------
- *« 65 % des erreurs de Tesseract sont de classe récupérable
  (case_error, ligature_error, abbreviation_error) — un
  post-processing trivial absorberait une partie. »*
- *« 12 % de vos documents concentrent 78 % du CER total
  (Pareto-CER). »*
- *« Le déficit projeté du moteur le plus fragile sur le corpus
  réel est de 4,2 points de CER (Sprint 81). »*
- *« Le top-3 des tokens GT systématiquement modernisés est
  maistre, nostre, veoir (Sprint 80). »*

Structure
---------
Module parallèle au registre narratif Sprint 19 : `Lever` est la
dataclass équivalente à `Fact`, `LeverImportance` reprend la
sémantique de `FactImportance`, `@register_lever` indexe les
détecteurs.  Garde-fou anti-hallucination identique : chaque
nombre rendu doit être présent dans le `payload` du `Lever`.

Les détecteurs lisent **uniquement** des structures déjà
construites par le pipeline du benchmark — ils ne calculent rien
de nouveau, ils synthétisent.  C'est pourquoi le module est
résolument optionnel : si un benchmark n'expose pas
`taxonomy_aggregated`, `inter_engine_analysis`, `corpus_difficulty`,
`lexical_modernization` ou `robustness_projection`, le détecteur
correspondant retourne tout simplement `[]`.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Modèle
# ──────────────────────────────────────────────────────────────────────────


class LeverType(str, Enum):
    """Types de leviers détectés."""

    DOMINANT_RECOVERABLE_CLASS = "dominant_recoverable_class"
    """Une part importante des erreurs d'un moteur est dans des classes
    catégorisées « récupérables » (Sprint 77)."""

    PARETO_CONCENTRATION = "pareto_concentration"
    """Une fraction minoritaire de documents concentre une fraction
    majoritaire du CER total — l'inspection ciblée est rentable."""

    COMPLEMENTARITY_OBSERVATION = "complementarity_observation"
    """Le `complementarity_gap` (Sprint 35) entre l'oracle et le
    meilleur moteur seul est non négligeable — observation factuelle,
    aucune recommandation d'ensemble."""

    LEXICAL_MODERNIZATION_OBSERVATION = "lexical_modernization_observation"
    """Top-N des tokens GT systématiquement modernisés (Sprint 80)."""

    ROBUSTNESS_PROJECTION_OBSERVATION = "robustness_projection_observation"
    """Déficit projeté global le plus important pour un moteur sur
    le corpus réel (Sprint 81)."""


class LeverImportance(int, Enum):
    """Importance éditoriale d'un levier."""

    HIGH = 70
    MEDIUM = 40
    LOW = 10


@dataclass
class Lever:
    """Observation factuelle synthétisable en encart « Leviers ».

    Attributes
    ----------
    type:
        Le type de levier (voir `LeverType`).
    importance:
        Score qui décide l'ordre d'affichage.
    payload:
        Données brutes — **tout chiffre rendu dans le HTML doit
        provenir d'ici**, jamais d'un calcul du renderer.
    engines_involved:
        Noms des moteurs concernés (peut être vide pour un levier
        corpus-wide).
    """

    type: LeverType
    importance: LeverImportance
    payload: dict
    engines_involved: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "type": self.type.value,
            "importance": int(self.importance),
            "payload": self.payload,
            "engines_involved": list(self.engines_involved),
        }


# ──────────────────────────────────────────────────────────────────────────
# Registre
# ──────────────────────────────────────────────────────────────────────────


LeverDetectorFn = Callable[[dict], list[Lever]]


@dataclass(frozen=True)
class LeverDetectorEntry:
    lever_type: LeverType
    fn: LeverDetectorFn
    priority: int


_LEVER_REGISTRY: dict[LeverType, LeverDetectorEntry] = {}
_LEVER_REGISTRY_LOCK = threading.Lock()


def register_lever(
    lever_type: LeverType,
    *,
    priority: int,
) -> Callable[[LeverDetectorFn], LeverDetectorFn]:
    """Décorateur : enregistre un détecteur de levier.

    Une seule fonction par type — réenregistrer lève `ValueError`.
    """
    def _decorator(fn: LeverDetectorFn) -> LeverDetectorFn:
        with _LEVER_REGISTRY_LOCK:
            if lever_type in _LEVER_REGISTRY:
                raise ValueError(
                    f"Détecteur déjà enregistré pour {lever_type.value!r} : "
                    f"{_LEVER_REGISTRY[lever_type].fn.__name__}."
                )
            _LEVER_REGISTRY[lever_type] = LeverDetectorEntry(
                lever_type=lever_type, fn=fn, priority=int(priority),
            )
        return fn
    return _decorator


def unregister_lever(lever_type: LeverType) -> None:
    with _LEVER_REGISTRY_LOCK:
        _LEVER_REGISTRY.pop(lever_type, None)


def iter_lever_detectors() -> list[LeverDetectorEntry]:
    with _LEVER_REGISTRY_LOCK:
        entries = list(_LEVER_REGISTRY.values())
    entries.sort(key=lambda e: e.priority)
    return entries


def detect_levers(benchmark_data: dict) -> list[Lever]:
    """Applique tous les détecteurs enregistrés et trie par importance
    décroissante puis priorité d'enregistrement croissante."""
    levers: list[Lever] = []
    for entry in iter_lever_detectors():
        try:
            result = entry.fn(benchmark_data)
        except Exception as e:
            logger.warning(
                "[levers.detector.%s] fonctionnalité dégradée : %s",
                entry.lever_type.value, e,
            )
            continue
        if result:
            levers.extend(result)
    # Tri stable : importance décroissante d'abord
    levers.sort(key=lambda lv: -int(lv.importance))
    return levers


# ──────────────────────────────────────────────────────────────────────────
# Détecteurs
# ──────────────────────────────────────────────────────────────────────────


# Catégorisation reprise du Sprint 77 (taxonomy_comparison.py).
# Volontairement dupliquée ici pour ne pas introduire d'import
# circulaire — la sémantique est gelée.
_RECOVERABILITY: dict[str, str] = {
    "case_error":         "recoverable",
    "ligature_error":     "recoverable",
    "abbreviation_error": "recoverable",
    "diacritic_error":    "difficult",
    "visual_confusion":   "difficult",
    "hapax":              "difficult",
    "lacuna":             "irrecoverable",
    "oov_character":      "irrecoverable",
    "segmentation_error": "irrecoverable",
}


@register_lever(LeverType.DOMINANT_RECOVERABLE_CLASS, priority=10)
def detect_dominant_recoverable_class(
    benchmark_data: dict,
    *,
    threshold: float = 0.30,
) -> list[Lever]:
    """Émet un levier si ≥ `threshold` des erreurs d'un moteur sont
    classifiées récupérables (catégorisation Sprint 77).

    Lit `benchmark_data["engines"][i]["aggregated_taxonomy"]` —
    structure produite par le runner historique. Si absent, retourne
    [].
    """
    engines = benchmark_data.get("engines") or []
    out: list[Lever] = []
    for engine in engines:
        taxonomy = engine.get("aggregated_taxonomy")
        if not taxonomy:
            continue
        # `taxonomy` peut être {class_name: int} ou un dict avec une
        # sous-clé "counts" — on accepte les deux conventions.
        counts = taxonomy.get("counts") if isinstance(taxonomy, dict) and "counts" in taxonomy else taxonomy
        if not isinstance(counts, dict) or not counts:
            continue
        try:
            int_counts = {k: int(v) for k, v in counts.items() if isinstance(v, (int, float))}
        except (TypeError, ValueError):
            continue
        total = sum(int_counts.values())
        if total <= 0:
            continue
        recoverable_total = sum(
            v for k, v in int_counts.items()
            if _RECOVERABILITY.get(k) == "recoverable"
        )
        share = recoverable_total / total
        if share < threshold:
            continue
        # Classes récupérables non vides triées par count décroissant
        breakdown = sorted(
            (
                (k, v) for k, v in int_counts.items()
                if _RECOVERABILITY.get(k) == "recoverable" and v > 0
            ),
            key=lambda kv: -kv[1],
        )
        importance = (
            LeverImportance.HIGH if share >= 0.50 else LeverImportance.MEDIUM
        )
        out.append(Lever(
            type=LeverType.DOMINANT_RECOVERABLE_CLASS,
            importance=importance,
            payload={
                "engine": engine.get("name") or "?",
                "share_recoverable": share,
                "share_recoverable_pct": round(share * 100, 1),
                "n_recoverable": recoverable_total,
                "n_total_errors": total,
                "top_classes": [
                    {"class": k, "count": v} for k, v in breakdown[:3]
                ],
            },
            engines_involved=(engine.get("name") or "?",),
        ))
    return out


@register_lever(LeverType.PARETO_CONCENTRATION, priority=20)
def detect_pareto_concentration(
    benchmark_data: dict,
    *,
    top_share: float = 0.20,
    cer_share_threshold: float = 0.50,
) -> list[Lever]:
    """Émet un levier si une fraction minoritaire de documents
    (`top_share`) concentre plus de `cer_share_threshold` du CER
    total cumulé sur le moteur leader.

    Lit `benchmark_data["per_doc_cer"][engine_name]` ou tente de
    reconstruire depuis `benchmark_data["engines"][...]["per_doc"]`.
    Si rien d'exploitable, retourne [].
    """
    ranking = benchmark_data.get("ranking") or []
    if not ranking:
        return []
    leader = ranking[0]
    leader_name = leader.get("engine")
    if not leader_name:
        return []

    per_doc_cer: list[float] = []
    # Voie 1 : structure plate "per_doc_cer"
    flat = benchmark_data.get("per_doc_cer") or {}
    if isinstance(flat, dict) and leader_name in flat and isinstance(flat[leader_name], list):
        per_doc_cer = [float(x) for x in flat[leader_name] if isinstance(x, (int, float))]
    else:
        # Voie 2 : engine.per_doc liste de dicts {cer: float}
        for engine in benchmark_data.get("engines") or []:
            if engine.get("name") != leader_name:
                continue
            per_doc = engine.get("per_doc") or []
            for entry in per_doc:
                if isinstance(entry, dict) and isinstance(entry.get("cer"), (int, float)):
                    per_doc_cer.append(float(entry["cer"]))
            break

    if not per_doc_cer:
        return []
    total_cer = sum(per_doc_cer)
    if total_cer <= 0:
        return []

    sorted_cer = sorted(per_doc_cer, reverse=True)
    n = len(sorted_cer)
    n_top = max(1, int(round(top_share * n)))
    top_cer_sum = sum(sorted_cer[:n_top])
    share_of_total = top_cer_sum / total_cer
    if share_of_total < cer_share_threshold:
        return []
    importance = (
        LeverImportance.HIGH if share_of_total >= 0.75
        else LeverImportance.MEDIUM
    )
    return [Lever(
        type=LeverType.PARETO_CONCENTRATION,
        importance=importance,
        payload={
            "engine": leader_name,
            "n_docs": n,
            "n_docs_top": n_top,
            "top_share_pct": round((n_top / n) * 100, 1),
            "cer_share_of_total": share_of_total,
            "cer_share_pct": round(share_of_total * 100, 1),
        },
        engines_involved=(leader_name,),
    )]


@register_lever(LeverType.COMPLEMENTARITY_OBSERVATION, priority=30)
def detect_complementarity_observation(
    benchmark_data: dict,
    *,
    min_relative_gap: float = 0.20,
) -> list[Lever]:
    """Reformule factuellement le `complementarity_gap`

    Lit `benchmark_data["inter_engine_analysis"]`. Garde-fou : ne
    déclenche que si `relative_gap` ≥ `min_relative_gap`. **Aucune
    recommandation d'ensemble** — le levier dit factuellement
    « X points séparent l'oracle du meilleur moteur », c'est tout.
    """
    inter = benchmark_data.get("inter_engine_analysis") or {}
    cgap = inter.get("complementarity_gap") or {}
    relative_gap = cgap.get("relative_gap")
    absolute_gap = cgap.get("absolute_gap")
    if relative_gap is None or absolute_gap is None:
        return []
    try:
        rg = float(relative_gap)
        ag = float(absolute_gap)
    except (TypeError, ValueError):
        return []
    if rg < min_relative_gap:
        return []
    importance = (
        LeverImportance.HIGH if rg >= 0.50 else LeverImportance.MEDIUM
    )
    payload: dict = {
        "absolute_gap": ag,
        "absolute_gap_pct": round(ag * 100, 1),
        "relative_gap": rg,
        "relative_gap_pct": round(rg * 100, 1),
    }
    best_engine = cgap.get("best_engine") or inter.get("best_engine")
    best_recall = cgap.get("best_recall") or inter.get("best_engine_recall")
    oracle_recall = cgap.get("oracle_recall") or inter.get("oracle_recall")
    engines_involved: tuple[str, ...] = ()
    if best_engine:
        payload["best_engine"] = str(best_engine)
        engines_involved = (str(best_engine),)
    if isinstance(best_recall, (int, float)):
        payload["best_recall"] = float(best_recall)
    if isinstance(oracle_recall, (int, float)):
        payload["oracle_recall"] = float(oracle_recall)
    return [Lever(
        type=LeverType.COMPLEMENTARITY_OBSERVATION,
        importance=importance,
        payload=payload,
        engines_involved=engines_involved,
    )]


@register_lever(LeverType.LEXICAL_MODERNIZATION_OBSERVATION, priority=40)
def detect_lexical_modernization_observation(
    benchmark_data: dict,
    *,
    top_n: int = 3,
    min_total: int = 3,
    min_rate: float = 0.50,
) -> list[Lever]:
    """Pour chaque moteur disposant de `lexical_modernization`,
    émet un levier listant les `top_n` tokens GT les plus modernisés.

    Lit `benchmark_data["engines"][i]["lexical_modernization"]` qui
    suit la forme produite par `compute_lexical_modernization` du
    Sprint 80 (`{"n_gt_tokens": int, "tokens": dict}`).
    """
    out: list[Lever] = []
    for engine in benchmark_data.get("engines") or []:
        data = engine.get("lexical_modernization")
        if not isinstance(data, dict):
            continue
        tokens = data.get("tokens") or {}
        if not isinstance(tokens, dict) or not tokens:
            continue
        candidates: list[tuple[str, dict]] = []
        for gt_token, slot in tokens.items():
            if not isinstance(slot, dict):
                continue
            n_total = slot.get("n_total")
            rate = slot.get("rate_modernized")
            if not isinstance(n_total, (int, float)) or not isinstance(rate, (int, float)):
                continue
            if int(n_total) < min_total:
                continue
            if float(rate) < min_rate:
                continue
            candidates.append((gt_token, dict(slot)))
        if not candidates:
            continue
        candidates.sort(
            key=lambda kv: (-float(kv[1].get("rate_modernized", 0.0)),
                            -int(kv[1].get("n_total", 0)),
                            kv[0]),
        )
        top = candidates[:top_n]
        engine_name = engine.get("name") or "?"
        max_rate = max(float(slot.get("rate_modernized", 0.0)) for _, slot in top)
        importance = (
            LeverImportance.HIGH if max_rate >= 0.90 else LeverImportance.MEDIUM
        )
        out.append(Lever(
            type=LeverType.LEXICAL_MODERNIZATION_OBSERVATION,
            importance=importance,
            payload={
                "engine": engine_name,
                "top_tokens": [
                    {
                        "gt_token": gt,
                        "n_total": int(slot.get("n_total", 0)),
                        "rate_modernized": float(slot.get("rate_modernized", 0.0)),
                        "rate_modernized_pct": round(
                            float(slot.get("rate_modernized", 0.0)) * 100, 1,
                        ),
                    }
                    for gt, slot in top
                ],
            },
            engines_involved=(engine_name,),
        ))
    return out


@register_lever(LeverType.ROBUSTNESS_PROJECTION_OBSERVATION, priority=50)
def detect_robustness_projection_observation(
    benchmark_data: dict,
    *,
    min_total_deficit: float = 0.02,
) -> list[Lever]:
    """Lit l'agrégation par moteur de la projection de robustesse
    (Sprint 81). Émet le levier pour le moteur dont
    `total_expected_deficit` est ≥ `min_total_deficit` (par défaut
    2 points de CER).

    Lit `benchmark_data["robustness_projection_aggregated"]` —
    structure produite par `aggregate_projection_per_engine`.
    """
    agg = benchmark_data.get("robustness_projection_aggregated") or {}
    if not isinstance(agg, dict) or not agg:
        return []
    out: list[Lever] = []
    for engine_name, info in agg.items():
        if not isinstance(info, dict):
            continue
        total_deficit = info.get("total_expected_deficit")
        worst_type = info.get("worst_degradation_type")
        worst_deficit = info.get("worst_degradation_deficit")
        if not isinstance(total_deficit, (int, float)):
            continue
        if float(total_deficit) < min_total_deficit:
            continue
        importance = (
            LeverImportance.HIGH if float(total_deficit) >= 0.05
            else LeverImportance.MEDIUM
        )
        payload: dict = {
            "engine": engine_name,
            "total_expected_deficit": float(total_deficit),
            "total_expected_deficit_pct": round(float(total_deficit) * 100, 1),
            "n_degradation_types": int(info.get("n_degradation_types") or 0),
        }
        if isinstance(worst_type, str):
            payload["worst_degradation_type"] = worst_type
        if isinstance(worst_deficit, (int, float)):
            payload["worst_degradation_deficit"] = float(worst_deficit)
            payload["worst_degradation_deficit_pct"] = round(
                float(worst_deficit) * 100, 1,
            )
        out.append(Lever(
            type=LeverType.ROBUSTNESS_PROJECTION_OBSERVATION,
            importance=importance,
            payload=payload,
            engines_involved=(engine_name,),
        ))
    # Tri par déficit décroissant pour stabilité d'affichage.
    out.sort(
        key=lambda lv: -float(lv.payload.get("total_expected_deficit") or 0.0),
    )
    return out


__all__ = [
    "Lever",
    "LeverImportance",
    "LeverType",
    "LeverDetectorEntry",
    "register_lever",
    "unregister_lever",
    "iter_lever_detectors",
    "detect_levers",
    "detect_dominant_recoverable_class",
    "detect_pareto_concentration",
    "detect_complementarity_observation",
    "detect_lexical_modernization_observation",
    "detect_robustness_projection_observation",
]
