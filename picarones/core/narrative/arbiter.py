"""Arbitre de sélection des faits narratifs.

L'arbitre transforme une liste potentiellement longue de ``Fact`` détectés
en une synthèse courte (3 à 5 phrases) adaptée à l'ouverture du rapport.

Règles de sélection :
  1. Tri par importance décroissante, puis par type (ordre canonique).
  2. Non-redondance : un seul fait par moteur, sauf si les types sont
     complémentaires (ex. ``GLOBAL_LEADER_CER`` + ``SIGNIFICANT_GAP``
     concernent le leader mais apportent une information différente).
  3. Limite : au maximum ``max_facts`` faits retenus (défaut 5).
  4. Déterminisme : tri stable sur (−importance, ordre canonique du type,
     noms des moteurs) pour garantir une sortie bit-à-bit identique.

Les détecteurs peuvent émettre plusieurs faits du même type (ex. plusieurs
``STATISTICAL_TIE`` si plusieurs groupes distincts). L'arbitre ne fusionne
pas mais peut limiter par type.
"""

from __future__ import annotations

from typing import Iterable

from picarones.core.narrative.facts import Fact, FactImportance, FactType


# Ordre canonique des types pour départager les ex-aequo à l'importance égale.
_TYPE_ORDER: tuple[FactType, ...] = (
    FactType.GLOBAL_LEADER_CER,
    FactType.STATISTICAL_TIE,
    FactType.SIGNIFICANT_GAP,
    FactType.STRATUM_WINNER,
    FactType.STRATUM_COLLAPSE,
    FactType.ERROR_PROFILE_OUTLIER,
    FactType.LLM_HALLUCINATION_FLAG,
    FactType.ROBUSTNESS_FRAGILE,
    FactType.PARETO_ALTERNATIVE,
    FactType.SPEED_WINNER,
    FactType.COST_OUTLIER,
    FactType.CONFIDENCE_WARNING,
)
_TYPE_INDEX: dict[FactType, int] = {t: i for i, t in enumerate(_TYPE_ORDER)}


# Paires de types qui ne sont PAS considérées comme redondantes même quand
# elles concernent le même moteur. Tout autre couple → un seul fait retenu
# pour le moteur (le plus important).
_COMPLEMENTARY_PAIRS: frozenset[frozenset[FactType]] = frozenset({
    frozenset({FactType.GLOBAL_LEADER_CER, FactType.SIGNIFICANT_GAP}),
    frozenset({FactType.GLOBAL_LEADER_CER, FactType.SPEED_WINNER}),
    frozenset({FactType.GLOBAL_LEADER_CER, FactType.CONFIDENCE_WARNING}),
    frozenset({FactType.STATISTICAL_TIE, FactType.SPEED_WINNER}),
})


def _sort_key(fact: Fact) -> tuple:
    """Clé de tri stable : importance (desc), type canonique, moteurs."""
    return (
        -int(fact.importance),
        _TYPE_INDEX.get(fact.type, len(_TYPE_ORDER)),
        tuple(sorted(fact.engines_involved)),
        fact.stratum or "",
    )


def _is_redundant(candidate: Fact, kept: Fact) -> bool:
    """Vrai si ``candidate`` apporte trop peu par rapport à ``kept``.

    Deux faits sont redondants s'ils concernent exactement le même moteur,
    ont le même type, et la même strate (s'il y en a une). Des types
    différents sur le même moteur ne sont considérés redondants que s'ils
    n'appartiennent pas aux paires complémentaires (ex : un leader peut
    aussi être rapide ; c'est complémentaire).
    """
    if candidate.type == kept.type and candidate.stratum == kept.stratum:
        return set(candidate.engines_involved) == set(kept.engines_involved)
    if set(candidate.engines_involved) == set(kept.engines_involved):
        pair = frozenset({candidate.type, kept.type})
        return pair not in _COMPLEMENTARY_PAIRS
    return False


def _remove_contradictions(facts: list[Fact]) -> list[Fact]:
    """Supprime les faits incohérents sur le plan statistique.

    Règle centrale : si Nemenyi (post-hoc corrigé pour comparaisons multiples)
    place deux moteurs dans le même groupe d'ex-aequo, alors un ``SIGNIFICANT_GAP``
    basé sur Wilcoxon non corrigé entre ces deux mêmes moteurs est trompeur
    pour un lecteur non statisticien. Nemenyi l'emporte.
    """
    tied_groups: list[set[str]] = []
    for f in facts:
        if f.type == FactType.STATISTICAL_TIE:
            tied_groups.append(set(f.engines_involved))

    def _is_contradicted(fact: Fact) -> bool:
        if fact.type != FactType.SIGNIFICANT_GAP:
            return False
        pair = set(fact.engines_involved)
        return any(pair <= group for group in tied_groups)

    return [f for f in facts if not _is_contradicted(f)]


def select_facts(
    facts: Iterable[Fact],
    max_facts: int = 5,
    min_importance: FactImportance = FactImportance.MEDIUM,
) -> list[Fact]:
    """Sélectionne la synthèse finale à partir d'une liste brute de faits.

    Parameters
    ----------
    facts:
        Liste de ``Fact`` brute issue de ``DetectorRegistry.run``.
    max_facts:
        Nombre maximal de faits retenus (défaut : 5).
    min_importance:
        Seuil minimal d'importance. Les faits ``LOW`` sont exclus par défaut.

    Returns
    -------
    Liste ordonnée, prête à être rendue. Toujours ≤ ``max_facts``.
    """
    facts_list = [f for f in facts if int(f.importance) >= int(min_importance)]
    facts_list = _remove_contradictions(facts_list)
    ranked = sorted(facts_list, key=_sort_key)

    selected: list[Fact] = []
    for fact in ranked:
        if any(_is_redundant(fact, kept) for kept in selected):
            continue
        selected.append(fact)
        if len(selected) >= max_facts:
            break
    return selected
