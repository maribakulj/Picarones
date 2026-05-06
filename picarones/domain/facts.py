"""Modèle de données du moteur narratif.

Un ``Fact`` est une observation structurée extraite d'un résultat
de benchmark.  Chaque détecteur (cf. ``picarones.reports_v2.narrative``)
retourne zéro, un ou plusieurs ``Fact`` typés.  L'arbitre trie par
``importance`` et sélectionne les faits à afficher dans la synthèse
narrative.

**Règle d'or anti-hallucination** : chaque valeur numérique ou nom
d'entité présent dans ``payload`` doit provenir directement du JSON
d'entrée, jamais d'une génération.  C'est ce qui rend la synthèse
reproductible bit-à-bit et immune à l'hallucination par
construction.

Module dans ``domain/`` parce qu'il définit des types purs (pas
d'I/O, pas de framework).  Consommé par ``reports_v2/narrative/``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class FactType(str, Enum):
    """Types de faits détectables.

    L'ajout d'un nouveau type se fait ici + un détecteur dans
    ``reports_v2/narrative/detectors/`` + un template dans
    ``reports_v2/narrative/templates/{lang}.yaml``.
    """

    GLOBAL_LEADER_CER = "global_leader_cer"
    """Pipeline avec le CER médian le plus bas sur l'ensemble du corpus."""

    STATISTICAL_TIE = "statistical_tie"
    """Top-N pipelines statistiquement indiscernables (Nemenyi)."""

    SIGNIFICANT_GAP = "significant_gap"
    """Écart statistiquement significatif entre le 1ᵉʳ et le 2ᵉ du classement."""

    PARETO_ALTERNATIVE = "pareto_alternative"
    """Pipeline sur la frontière Pareto différent du leader CER pur."""

    STRATUM_WINNER = "stratum_winner"
    """Pipeline qui domine sur une strate spécifique (siècle, langue, type)."""

    STRATUM_COLLAPSE = "stratum_collapse"
    """Pipeline globalement bon qui s'effondre sur une strate spécifique."""

    ERROR_PROFILE_OUTLIER = "error_profile_outlier"
    """Pipeline avec un profil taxonomique atypique (3× plus d'erreurs d'abréviation, etc.)."""

    LLM_HALLUCINATION_FLAG = "llm_hallucination_flag"
    """LLM avec un taux d'hallucination notablement supérieur aux autres."""

    ROBUSTNESS_FRAGILE = "robustness_fragile"
    """Pipeline qui dégrade fortement au-dessus d'un seuil de bruit/flou."""

    COST_OUTLIER = "cost_outlier"
    """Pipeline au ratio coût/qualité très défavorable."""

    SPEED_WINNER = "speed_winner"
    """Pipeline significativement plus rapide pour une qualité comparable."""

    CONFIDENCE_WARNING = "confidence_warning"
    """Intervalle de confiance très large : classement peu fiable."""

    ENSEMBLE_OPPORTUNITY = "ensemble_opportunity"
    """Deux pipelines sont fortement complémentaires : un voting
    majoritaire pourrait améliorer significativement le CER."""

    MEDIAN_MEAN_GAP_WARNING = "median_mean_gap_warning"
    """Distribution des CER fortement asymétrique sur le corpus —
    la moyenne du leader est tirée par quelques documents
    catastrophiques et masque les performances réelles.  La médiane
    est plus représentative."""

    STRATIFICATION_RECOMMENDED = "stratification_recommended"
    """Le corpus est hétérogène : le pipeline leader varie fortement
    selon la strate.  Le lecteur doit consulter la vue stratifiée
    plutôt que de se fier au seul classement global."""

    ENGINE_OFF_BASELINE = "engine_off_baseline"
    """Le CER courant d'un pipeline s'écarte significativement de
    sa moyenne historique sur le même corpus (lue depuis l'historique
    SQLite).  Garde-fous : ≥ 5 runs historiques même corpus +
    |delta_relatif| > 20 %."""

    ENGINE_UNSTABLE = "engine_unstable"
    """Un pipeline LLM/VLM exécuté plusieurs fois sur les mêmes
    documents produit des sorties différentes au-delà d'un seuil
    de variance.  Garde-fous : ≥ 2 runs et seuil sur le coefficient
    de variation du CER (>10 % par défaut) ou sur le rappel de runs
    identiques (<50 %)."""

    REGRESSION_IN_HISTORY = "regression_in_history"
    """Un pipeline montre une tendance ou une rupture défavorable
    sur l'historique SQLite : son CER moyen s'est dégradé sur les
    N derniers runs.  Garde-fous : ≥ 3 runs historiques et soit
    pente > seuil (régression progressive), soit change-point avec
    delta > seuil (rupture brutale)."""

    IMPORTER_FALLBACK_TRIGGERED = "importer_fallback_triggered"
    """Un import distant (HuggingFace, HTR-United, Gallica,
    eScriptorium…) a échoué ou a basculé en mode dégradé pendant
    la constitution du corpus.  Le moteur narratif lit la file
    d'incidents accumulée depuis le dernier benchmark.  Un Fact par
    incident, importance MEDIUM (HIGH si plusieurs incidents sur le
    même importer)."""

    PRICING_STALENESS_WARNING = "pricing_staleness_warning"
    """La table de pricing (``picarones/data/pricing.yaml``) a
    dépassé sa date ``valid_until``.  Les chiffres coût/€ et CO₂
    du rapport ne reflètent plus les tarifs courants.  Le détecteur
    lit ``meta.valid_until`` et compare à la date du jour ; émet un
    Fact d'importance MEDIUM avec le délai dépassé en jours."""


class FactImportance(int, Enum):
    """Score d'importance d'un fait — décide l'ordre et la sélection."""

    CRITICAL = 100
    """À remonter systématiquement en synthèse (ex : leader + écart significatif)."""

    HIGH = 70
    """À remonter sauf si déjà redondant avec un fait critique."""

    MEDIUM = 40
    """À remonter si la synthèse a encore de la place."""

    LOW = 10
    """Informatif, remonté uniquement en vue détaillée."""


@dataclass
class Fact:
    """Observation structurée extraite d'un benchmark.

    Attributes
    ----------
    type:
        Type de fait (voir :class:`FactType`).
    importance:
        Priorité de sélection (voir :class:`FactImportance`).
    payload:
        Dict de données brutes sérialisables.  **Toutes les valeurs
        doivent provenir du JSON d'entrée** — c'est le garde-fou
        anti-hallucination.
    engines_involved:
        Noms des pipelines concernés.  Utilisé par l'arbitre pour
        détecter les redondances (deux faits sur le même pipeline
        = fusion ou sélection).  Le nom du champ reste
        ``engines_involved`` pour la rétrocompat ; la sémantique a
        glissé de "moteur OCR" (legacy) à "pipeline" (rewrite) sans
        changement de format.
    stratum:
        Strate concernée (ex : "XVIIe siècle", "latin médiéval")
        ou ``None``.
    """

    type: FactType
    importance: FactImportance
    payload: dict
    engines_involved: tuple[str, ...] = ()
    stratum: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "type": self.type.value,
            "importance": int(self.importance),
            "payload": self.payload,
            "engines_involved": list(self.engines_involved),
            "stratum": self.stratum,
        }


#: Signature d'un détecteur — pure et déterministe.
DetectorFn = Callable[[dict], list[Fact]]


@dataclass
class DetectorRegistry:
    """Registre central des détecteurs de faits.

    Un détecteur est enregistré via ``register(fact_type, fn)``.
    ``run`` appelle tous les détecteurs enregistrés et renvoie la
    liste consolidée.

    Tolérant aux exceptions : un détecteur qui plante émet un
    ``logger.warning`` mais ne fait pas tomber les autres.
    """

    _detectors: dict[FactType, DetectorFn] = field(default_factory=dict)

    def register(self, fact_type: FactType, fn: DetectorFn) -> None:
        self._detectors[fact_type] = fn

    def unregister(self, fact_type: FactType) -> None:
        self._detectors.pop(fact_type, None)

    def registered_types(self) -> tuple[FactType, ...]:
        return tuple(self._detectors.keys())

    def run(self, benchmark_data: dict) -> list[Fact]:
        facts: list[Fact] = []
        for fact_type, fn in self._detectors.items():
            try:
                result = fn(benchmark_data)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[narrative.detector.%s] fonctionnalité dégradée : %s",
                    fact_type.value, exc,
                )
                continue
            if result:
                facts.extend(result)
        return facts


_DEFAULT_REGISTRY = DetectorRegistry()


def detect_all(
    benchmark_data: dict,
    registry: Optional[DetectorRegistry] = None,
) -> list[Fact]:
    """Applique tous les détecteurs enregistrés au benchmark donné."""
    if registry is None:
        registry = _DEFAULT_REGISTRY
    return registry.run(benchmark_data)


__all__ = [
    "DetectorFn",
    "DetectorRegistry",
    "Fact",
    "FactImportance",
    "FactType",
    "detect_all",
]
