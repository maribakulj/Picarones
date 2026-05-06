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

from typing import Iterable, Sequence

from picarones.core.facts import Fact, FactImportance, FactType


# Ordre canonique des types pour départager les ex-aequo à l'importance égale.
#
# Politique éditoriale — exposée et documentée dans
# ``docs/explanation/narrative-engine.md`` § Editorial policy.
# L'ordre encode quels faits sont remontés en priorité quand plusieurs ont
# la même ``FactImportance``. Surchargeable via le paramètre ``type_order``
# de ``select_facts`` sans patcher le code.
#
# Sprint 29 : la valeur n'est plus codée en dur ici — elle est dérivée du
# registre déclaratif (``@register_detector(..., priority=N)``). Ajouter
# un détecteur en bonne position se fait donc en éditant **un seul**
# fichier (``detectors.py``) au lieu de quatre comme avant.
def _compute_default_type_order() -> tuple[FactType, ...]:
    # Import local pour éviter la dépendance circulaire au chargement.
    from picarones.measurements.narrative.registry import default_type_order
    order = default_type_order()
    # Filet de sécurité : tant que les détecteurs n'ont pas été importés
    # (cas des tests qui mockent le registre), on retombe sur un ordre
    # canonique gravé pour ne pas planter ``select_facts``.
    if not order:
        return _FALLBACK_TYPE_ORDER
    return order


# Ordre statique gardé en mémoire : utilisé si jamais le registre est vide
# au moment où ``arbiter`` est chargé (chargement partiel par les tests).
_FALLBACK_TYPE_ORDER: tuple[FactType, ...] = (
    FactType.GLOBAL_LEADER_CER,
    FactType.STATISTICAL_TIE,
    FactType.SIGNIFICANT_GAP,
    FactType.STRATUM_WINNER,
    # Sprint 46 — priority 45, juste après STRATUM_WINNER (40),
    # avant STRATUM_COLLAPSE (50). La recommandation de stratification
    # nuance directement les autres faits par strate.
    FactType.STRATIFICATION_RECOMMENDED,
    FactType.STRATUM_COLLAPSE,
    FactType.ERROR_PROFILE_OUTLIER,
    FactType.LLM_HALLUCINATION_FLAG,
    FactType.ROBUSTNESS_FRAGILE,
    FactType.PARETO_ALTERNATIVE,
    FactType.SPEED_WINNER,
    FactType.COST_OUTLIER,
    FactType.CONFIDENCE_WARNING,
    FactType.ENSEMBLE_OPPORTUNITY,
    FactType.MEDIAN_MEAN_GAP_WARNING,
    # Sprint 73 — priority 150, après MEDIAN_MEAN_GAP_WARNING (140).
    # Le détecteur off-baseline donne le contexte historique, qui
    # vient en fin de synthèse comme « note ».
    FactType.ENGINE_OFF_BASELINE,
    # Sprint 90 — priority 160, ferme la synthèse avec la mise en
    # garde sur la reproductibilité.  Une instabilité multi-runs
    # discrédite toute autre conclusion sur ce moteur ; on la
    # remonte en dernier pour ne pas l'enterrer.
    FactType.ENGINE_UNSTABLE,
    # Sprint 92 — priority 170, après ENGINE_UNSTABLE.  La
    # régression historique complète A.I.3 (off-baseline) en
    # caractérisant la tendance : l'écart courant est-il une
    # dégradation graduelle, une rupture brutale, ou un bruit ?
    FactType.REGRESSION_IN_HISTORY,
    # Sprint A3 — priority 180, en queue.  Les incidents d'importer
    # sont contextuels à l'acquisition de données (non au ranking) ;
    # ils viennent en toute fin de synthèse comme avertissement sur
    # la qualité du corpus.
    FactType.IMPORTER_FALLBACK_TRIGGERED,
    # Sprint A8 — priority 200, dernier item (informationnel pur).
    # Avertit que la table de pricing utilisée a dépassé sa date de
    # validité — la lecture coût/CO₂ doit être nuancée.
    FactType.PRICING_STALENESS_WARNING,
)


# ``DEFAULT_TYPE_ORDER`` reste un attribut module accessible. On le calcule
# à l'import si possible, sinon on prend le fallback ; ``select_facts``
# recalcule à chaque appel pour absorber les ajouts de détecteurs après
# l'import initial (extensions tierces).
DEFAULT_TYPE_ORDER: tuple[FactType, ...] = _compute_default_type_order()

# Alias rétro-compatible.
_TYPE_ORDER = DEFAULT_TYPE_ORDER
_TYPE_INDEX: dict[FactType, int] = {t: i for i, t in enumerate(DEFAULT_TYPE_ORDER)}


# Paires de types qui ne sont PAS considérées comme redondantes même quand
# elles concernent le même moteur. Tout autre couple → un seul fait retenu
# pour le moteur (le plus important).
_COMPLEMENTARY_PAIRS: frozenset[frozenset[FactType]] = frozenset({
    frozenset({FactType.GLOBAL_LEADER_CER, FactType.SIGNIFICANT_GAP}),
    frozenset({FactType.GLOBAL_LEADER_CER, FactType.SPEED_WINNER}),
    frozenset({FactType.GLOBAL_LEADER_CER, FactType.CONFIDENCE_WARNING}),
    frozenset({FactType.STATISTICAL_TIE, FactType.SPEED_WINNER}),
    # Sprint 44 — l'avertissement d'asymétrie nuance le leader
    # plutôt que de le doubler : on veut les deux phrases ensemble.
    frozenset({FactType.GLOBAL_LEADER_CER, FactType.MEDIAN_MEAN_GAP_WARNING}),
    # Sprint 46 — la recommandation de stratification est un méta-conseil
    # qui s'ajoute au leader sans le contredire ; les deux peuvent
    # cohabiter même quand ils concernent le même moteur.
    frozenset({FactType.GLOBAL_LEADER_CER, FactType.STRATIFICATION_RECOMMENDED}),
    # Sprint 90 — l'instabilité multi-runs nuance les conclusions
    # sur le moteur leader sans les contredire : un moteur peut être
    # leader **et** instable, et c'est précisément l'information
    # critique pour la reproductibilité scientifique.
    frozenset({FactType.GLOBAL_LEADER_CER, FactType.ENGINE_UNSTABLE}),
    # Sprint 92 — la régression historique caractérise la tendance
    # du leader : un leader peut être en régression progressive,
    # info critique pour décider quand re-tester.
    frozenset({FactType.GLOBAL_LEADER_CER, FactType.REGRESSION_IN_HISTORY}),
    # Off-baseline (Sprint 73) dit "écart anormal sur ce corpus" ;
    # regression-in-history (Sprint 92) dit "tendance dans le
    # temps" — les deux se complètent sans se redonder.
    frozenset({FactType.ENGINE_OFF_BASELINE, FactType.REGRESSION_IN_HISTORY}),
})


def _sort_key(fact: Fact, type_index: dict[FactType, int]) -> tuple:
    """Clé de tri stable : importance (desc), type canonique, moteurs."""
    return (
        -int(fact.importance),
        type_index.get(fact.type, len(type_index)),
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
    type_order: Sequence[FactType] | None = None,
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
    type_order:
        Surcharge optionnelle de l'ordre canonique des types pour départager
        les faits d'égale importance. ``None`` (défaut) utilise
        ``DEFAULT_TYPE_ORDER``. Une institution peut passer son propre ordre
        sans patcher le code — voir ``docs/explanation/narrative-engine.md``.

    Returns
    -------
    Liste ordonnée, prête à être rendue. Toujours ≤ ``max_facts``.
    """
    if type_order is None:
        # Sprint 29 — recalcul à chaque appel pour absorber les détecteurs
        # enregistrés après l'import d'arbiter (extensions tierces qui
        # font ``@register_detector`` dans un module utilisateur).
        from picarones.measurements.narrative.registry import default_type_order
        live_order = default_type_order() or _FALLBACK_TYPE_ORDER
        type_index = {t: i for i, t in enumerate(live_order)}
    else:
        type_index = {t: i for i, t in enumerate(type_order)}

    facts_list = [f for f in facts if int(f.importance) >= int(min_importance)]
    facts_list = _remove_contradictions(facts_list)
    ranked = sorted(facts_list, key=lambda f: _sort_key(f, type_index))

    selected: list[Fact] = []
    for fact in ranked:
        if any(_is_redundant(fact, kept) for kept in selected):
            continue
        selected.append(fact)
        if len(selected) >= max_facts:
            break
    return selected
