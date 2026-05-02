"""Modèle de données du moteur narratif.

Un ``Fact`` est une observation structurée extraite d'un ``BenchmarkResult``.
Chaque détecteur retourne zéro, un ou plusieurs ``Fact`` typés. L'arbitre
(Sprint 4) trie par ``importance`` et sélectionne les faits à afficher.

Règle d'or (à vérifier par tests) : chaque valeur numérique ou nom d'entité
présent dans ``payload`` doit provenir directement du JSON d'entrée, jamais
d'une génération. C'est ce qui rend la synthèse reproductible bit-à-bit et
immune à l'hallucination par construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class FactType(str, Enum):
    """Types de faits détectables.

    L'ajout d'un nouveau type se fait ici + un détecteur dans ``detectors.py``
    + un template dans ``narrative/templates_{lang}.yaml`` (Sprint 4).
    """

    GLOBAL_LEADER_CER = "global_leader_cer"
    """Moteur avec le CER médian le plus bas sur l'ensemble du corpus."""

    STATISTICAL_TIE = "statistical_tie"
    """Top-N moteurs statistiquement indiscernables (Nemenyi, Sprint 3)."""

    SIGNIFICANT_GAP = "significant_gap"
    """Écart statistiquement significatif entre le 1ᵉʳ et le 2ᵉ du classement."""

    PARETO_ALTERNATIVE = "pareto_alternative"
    """Moteur sur la frontière Pareto différent du leader CER pur (Sprint 5)."""

    STRATUM_WINNER = "stratum_winner"
    """Moteur qui domine sur une strate spécifique (siècle, langue, type)."""

    STRATUM_COLLAPSE = "stratum_collapse"
    """Moteur globalement bon qui s'effondre sur une strate spécifique."""

    ERROR_PROFILE_OUTLIER = "error_profile_outlier"
    """Moteur avec un profil taxonomique atypique (ex : 3× plus d'erreurs d'abréviation)."""

    LLM_HALLUCINATION_FLAG = "llm_hallucination_flag"
    """LLM avec un taux d'hallucination notablement supérieur aux autres."""

    ROBUSTNESS_FRAGILE = "robustness_fragile"
    """Moteur qui dégrade fortement au-dessus d'un seuil de bruit/flou."""

    COST_OUTLIER = "cost_outlier"
    """Moteur au ratio coût/qualité très défavorable (Sprint 5)."""

    SPEED_WINNER = "speed_winner"
    """Moteur significativement plus rapide pour une qualité comparable."""

    CONFIDENCE_WARNING = "confidence_warning"
    """Intervalle de confiance très large : classement peu fiable."""

    ENSEMBLE_OPPORTUNITY = "ensemble_opportunity"
    """Deux moteurs sont fortement complémentaires : un voting majoritaire
    pourrait améliorer significativement le CER (Sprint 36)."""

    MEDIAN_MEAN_GAP_WARNING = "median_mean_gap_warning"
    """Distribution des CER fortement asymétrique sur le corpus —
    la moyenne du leader est tirée par quelques documents catastrophiques
    et masque les performances réelles. La médiane (utilisée pour le tri
    par défaut depuis Sprint 44) est plus représentative."""

    STRATIFICATION_RECOMMENDED = "stratification_recommended"
    """Le corpus est hétérogène du point de vue script_type : le moteur
    leader varie fortement selon la strate. Le lecteur doit consulter
    la vue stratifiée plutôt que de se fier au seul classement global
    (Sprint 46)."""

    ENGINE_OFF_BASELINE = "engine_off_baseline"
    """Le CER courant d'un moteur s'écarte significativement de sa
    moyenne historique sur le même corpus (lue depuis l'historique
    SQLite, Sprint 8). Lit ``BenchmarkHistory`` via le module
    ``baseline_comparison`` (Sprint 73). Garde-fous : ≥ 5 runs
    historiques même corpus + |delta_relatif| > 20 %."""

    ENGINE_UNSTABLE = "engine_unstable"
    """Un moteur LLM/VLM exécuté plusieurs fois sur les mêmes
    documents produit des sorties différentes au-delà d'un seuil
    de variance (Sprint 90).  Lit ``compute_multirun_stability``
    (Sprint 83).  Garde-fous : ≥ 2 runs et seuil sur le coefficient
    de variation du CER (>10 % par défaut) ou sur le rappel de
    runs identiques (<50 %)."""

    REGRESSION_IN_HISTORY = "regression_in_history"
    """Un moteur montre une tendance ou une rupture défavorable
    sur l'historique SQLite : son CER moyen s'est dégradé sur
    les N derniers runs (Sprint 92).  Lit
    ``compute_corpus_longitudinal`` du module ``longitudinal``.
    Garde-fous : ≥ 3 runs historiques et soit pente > seuil
    (régression progressive), soit change-point avec delta >
    seuil (rupture brutale)."""

    IMPORTER_FALLBACK_TRIGGERED = "importer_fallback_triggered"
    """Un import distant (HuggingFace, HTR-United, Gallica, eScriptorium…)
    a échoué ou a basculé en mode dégradé pendant la constitution du
    corpus (Sprint A3, item B-3).  Le moteur narratif lit
    ``picarones.extras.importers.consume_fallback_log()`` qui retourne
    et **vide** la liste des incidents accumulés depuis le dernier
    benchmark.  Un Fact par incident, importance MEDIUM (HIGH si
    plusieurs incidents sur le même importer)."""

    PRICING_STALENESS_WARNING = "pricing_staleness_warning"
    """La table de pricing (``picarones/data/pricing.yaml``) a dépassé
    sa date ``valid_until`` (Sprint A8, item m-14).  Les chiffres
    coût/€ et CO₂ du rapport ne reflètent plus les tarifs courants
    des fournisseurs cloud.  Le détecteur lit ``meta.valid_until`` et
    compare à la date du jour ; si dépassée, émet un Fact d'importance
    MEDIUM avec le délai dépassé en jours."""


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
        Type de fait (voir ``FactType``).
    importance:
        Priorité de sélection (voir ``FactImportance``).
    payload:
        Dict de données brutes sérialisables. **Toutes les valeurs doivent
        provenir du JSON d'entrée** — c'est le garde-fou anti-hallucination.
    engines_involved:
        Noms des moteurs concernés. Utilisé par l'arbitre pour détecter
        les redondances (deux faits sur le même moteur = fusion ou sélection).
    stratum:
        Strate concernée (ex : "XVIIe siècle", "latin médiéval") ou None.
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


# ---------------------------------------------------------------------------
# Registre de détecteurs
# ---------------------------------------------------------------------------

# Signature d'un détecteur : prend le dict JSON du benchmark, retourne une liste
# de Fact (potentiellement vide). Doit être pure et déterministe.
DetectorFn = Callable[[dict], list[Fact]]


@dataclass
class DetectorRegistry:
    """Registre central des détecteurs de faits.

    Un détecteur est enregistré via ``register(fact_type, fn)``. ``detect_all``
    appelle tous les détecteurs enregistrés et renvoie la liste consolidée.
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
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "[narrative.detector.%s] fonctionnalité dégradée : %s",
                    fact_type.value, e,
                )
                continue
            if result:
                facts.extend(result)
        return facts


def detect_all(benchmark_data: dict, registry: Optional[DetectorRegistry] = None) -> list[Fact]:
    """Applique tous les détecteurs enregistrés au benchmark donné.

    Point d'entrée du Sprint 4. Pour Sprint 1, le registre par défaut est vide :
    les détecteurs concrets sont ajoutés sprint par sprint.
    """
    if registry is None:
        registry = _DEFAULT_REGISTRY
    return registry.run(benchmark_data)


_DEFAULT_REGISTRY = DetectorRegistry()
