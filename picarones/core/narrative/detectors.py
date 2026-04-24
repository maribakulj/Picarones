"""Détecteurs de faits — stubs Sprint 1, implémentations sprint par sprint.

Chaque détecteur est une fonction pure ``(benchmark_data: dict) -> list[Fact]``.
Le sprint qui implémente chaque détecteur est indiqué dans le docstring.

Convention : un détecteur qui ne trouve rien retourne une liste vide. Il ne
doit jamais lever d'exception — la gestion d'erreur est centralisée dans
``DetectorRegistry.run``.
"""

from __future__ import annotations

from picarones.core.narrative.facts import Fact, FactImportance, FactType


# ---------------------------------------------------------------------------
# Détecteurs Sprint 4 (implémentations à venir)
# ---------------------------------------------------------------------------

def detect_global_leader_cer(benchmark_data: dict) -> list[Fact]:
    """Détecte le moteur avec le CER médian le plus bas.

    Implémentation Sprint 4. Lit ``benchmark_data["ranking"]``, identifie le
    leader, émet un Fact ``GLOBAL_LEADER_CER`` d'importance CRITICAL.
    """
    return []


def detect_statistical_tie(benchmark_data: dict) -> list[Fact]:
    """Détecte les groupes de moteurs statistiquement indiscernables.

    Implémentation Sprint 4, utilisant les résultats Nemenyi du Sprint 3
    (champ ``benchmark_data["statistics"]["nemenyi"]``).
    """
    return []


def detect_significant_gap(benchmark_data: dict) -> list[Fact]:
    """Détecte un écart significatif entre le 1ᵉʳ et le 2ᵉ du classement."""
    return []


def detect_pareto_alternative(benchmark_data: dict) -> list[Fact]:
    """Détecte un moteur Pareto-dominant différent du leader CER.

    Implémentation Sprint 5 (nécessite la modélisation coût).
    """
    return []


def detect_stratum_winner(benchmark_data: dict) -> list[Fact]:
    """Détecte un moteur qui domine sur une strate (siècle, langue, type)."""
    return []


def detect_stratum_collapse(benchmark_data: dict) -> list[Fact]:
    """Détecte un moteur globalement bon qui s'effondre sur une strate."""
    return []


def detect_error_profile_outlier(benchmark_data: dict) -> list[Fact]:
    """Détecte un profil taxonomique atypique.

    Utilise ``engine_reports[*].aggregated_taxonomy`` pour comparer la
    distribution des 9 classes entre moteurs.
    """
    return []


def detect_llm_hallucination_flag(benchmark_data: dict) -> list[Fact]:
    """Détecte un LLM au taux d'hallucination anormalement élevé.

    Activé par le Sprint 1 (câblage ``aggregated_hallucination``). Lit
    ``engine_reports[*].aggregated_hallucination.hallucinating_doc_rate`` et
    émet un Fact si un moteur dépasse significativement la médiane.
    Implémentation complète Sprint 4.
    """
    return []


def detect_robustness_fragile(benchmark_data: dict) -> list[Fact]:
    """Détecte un moteur qui dégrade fortement au-dessus d'un seuil de bruit."""
    return []


def detect_cost_outlier(benchmark_data: dict) -> list[Fact]:
    """Détecte un moteur au ratio coût/qualité très défavorable.

    Implémentation Sprint 5.
    """
    return []


def detect_speed_winner(benchmark_data: dict) -> list[Fact]:
    """Détecte un moteur significativement plus rapide pour qualité comparable."""
    return []


def detect_confidence_warning(benchmark_data: dict) -> list[Fact]:
    """Détecte un intervalle de confiance très large → classement peu fiable."""
    return []


# ---------------------------------------------------------------------------
# Enregistrement par défaut — à activer sprint par sprint
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
"""Table de correspondance ``FactType → détecteur``. L'enregistrement effectif
dans le registre par défaut sera activé au Sprint 4 quand les implémentations
seront prêtes. Pour Sprint 1, garder le registre par défaut vide évite que des
stubs polluent la synthèse."""
