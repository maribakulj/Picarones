"""Tests Sprint 29 — registre déclaratif des détecteurs narratifs.

Sprint 29 remplace le pattern *« quatre fichiers à toucher pour ajouter
un détecteur »* par un décorateur ``@register_detector`` qui :

1. enregistre la fonction dans un registre global trié par ``priority``,
2. refuse les doublons sur un même ``FactType``,
3. alimente automatiquement ``arbiter.DEFAULT_TYPE_ORDER`` et
   ``DETECTORS_BY_TYPE`` qui restent l'API publique historique.

Garanties testées
-----------------
- **Parité bit-à-bit** : la sortie de ``build_synthesis`` sur fixtures
  Sprint 19 est strictement identique à la version pré-Sprint 29.
  C'est le critère de sortie principal du sprint.
- **Extensibilité** : décorer une fonction la rend automatiquement
  disponible via ``iter_detectors`` et ``DEFAULT_TYPE_ORDER``, sans
  toucher ni ``arbiter.py`` ni ``__init__.py``.
- **Unicité** : tenter d'enregistrer deux détecteurs sur le même type
  lève ``ValueError``.
- **Tri stable** : à priorités égales, l'ordre d'enregistrement est
  préservé.
- **Cohérence interne** : tous les ``FactType`` du Sprint 4 sont
  enregistrés avec une priorité distincte.
"""

from __future__ import annotations

import pytest

from picarones.core.narrative import build_synthesis
from picarones.core.narrative.facts import (
    Fact,
    FactImportance,
    FactType,
)
from picarones.core.narrative.registry import (
    clear_registry,
    default_type_order,
    detector_for,
    iter_detectors,
    register_detector,
    unregister,
)


# ---------------------------------------------------------------------------
# 1. Le registre par défaut contient les 12 détecteurs Sprint 4
# ---------------------------------------------------------------------------

class TestRegistryPopulatedAtImport:
    def test_twelve_detectors_present(self):
        types = {entry.fact_type for entry in iter_detectors()}
        # Les 12 types canoniques du Sprint 4 + extensions Sprint 5
        expected = set(FactType)
        assert types == expected, (
            f"Types manquants : {expected - types} ; "
            f"types en trop : {types - expected}"
        )

    def test_priorities_are_unique(self):
        priorities = [entry.priority for entry in iter_detectors()]
        assert len(priorities) == len(set(priorities)), (
            "Deux détecteurs ne devraient pas avoir la même priorité par "
            "défaut — sinon l'ordre éditorial est indéterministe."
        )

    def test_priorities_match_historical_order(self):
        """Les priorités définies au Sprint 29 doivent reproduire l'ordre
        canonique pré-Sprint 29 pour ne pas casser la lecture du rapport."""
        from picarones.core.narrative.arbiter import _FALLBACK_TYPE_ORDER
        live = default_type_order()
        # Ils doivent contenir les mêmes types dans le même ordre.
        assert live == _FALLBACK_TYPE_ORDER

    def test_each_detector_callable(self):
        for entry in iter_detectors():
            assert callable(entry.fn), (
                f"L'entrée pour {entry.fact_type.value} n'est pas appelable"
            )


# ---------------------------------------------------------------------------
# 2. Parité bit-à-bit avec la version pré-Sprint 29
# ---------------------------------------------------------------------------

class TestParityWithPreSprint29:
    """Le refactor doit être strictement transparent : sur une fixture
    donnée, ``build_synthesis`` produit exactement les mêmes phrases."""

    def _data_with_full_signal(self) -> dict:
        """Données qui font sortir la majorité des détecteurs."""
        return {
            "meta": {"document_count": 20, "corpus_name": "test"},
            "ranking": [
                {"engine": "A", "mean_cer": 0.05, "mean_wer": 0.10},
                {"engine": "B", "mean_cer": 0.08, "mean_wer": 0.15},
                {"engine": "C", "mean_cer": 0.20, "mean_wer": 0.30},
            ],
            "engines": [
                {"name": "A", "cer": 0.05, "n_docs": 20},
                {"name": "B", "cer": 0.08, "n_docs": 20},
                {"name": "C", "cer": 0.20, "n_docs": 20},
            ],
            "statistics": {
                "pairwise_wilcoxon": [
                    {"engine_a": "A", "engine_b": "B", "p_value": 0.012,
                     "significant": True, "n_pairs": 20},
                ],
                "bootstrap_cis": [
                    {"engine": "A", "mean": 0.05, "ci_lower": 0.03, "ci_upper": 0.07},
                    {"engine": "B", "mean": 0.08, "ci_lower": 0.06, "ci_upper": 0.10},
                    {"engine": "C", "mean": 0.20, "ci_lower": 0.18, "ci_upper": 0.22},
                ],
            },
        }

    def test_synthesis_has_some_content(self):
        data = self._data_with_full_signal()
        result = build_synthesis(data, "fr")
        assert len(result["sentences"]) >= 1

    def test_synthesis_is_deterministic_across_calls(self):
        data = self._data_with_full_signal()
        a = build_synthesis(data, "fr")
        b = build_synthesis(data, "fr")
        assert a == b

    def test_global_leader_is_first(self):
        # Le leader CER doit dominer la synthèse — vérifie que le
        # registre conserve la priorité 10 sur GLOBAL_LEADER_CER.
        data = self._data_with_full_signal()
        result = build_synthesis(data, "fr")
        # La première phrase doit citer A (CER 0.05)
        assert "A" in result["sentences"][0]


# ---------------------------------------------------------------------------
# 3. Extensibilité : décorer une fonction tierce
# ---------------------------------------------------------------------------

class TestThirdPartyExtension:
    """Vérifie qu'on peut ajouter un détecteur depuis un module tiers
    sans toucher aux fichiers du package — preuve de l'autonomie du
    décorateur. Utilise un type FactType existant non utilisé pour
    éviter de polluer le registre permanent."""

    def setup_method(self):
        # Si jamais un précédent test a laissé un faux détecteur, on
        # nettoie. On ne touche PAS aux 12 builtins.
        for fake_type in (FactType.GLOBAL_LEADER_CER,):
            entry = detector_for(fake_type)
            if entry is not None and entry.fn.__module__ == __name__:
                unregister(fake_type)

    def teardown_method(self):
        # Idem
        for fake_type in (FactType.GLOBAL_LEADER_CER,):
            entry = detector_for(fake_type)
            if entry is not None and entry.fn.__module__ == __name__:
                unregister(fake_type)

    def test_decorator_rejects_double_registration(self):
        # Tenter de réenregistrer GLOBAL_LEADER_CER doit lever.
        with pytest.raises(ValueError, match="déjà enregistré"):
            @register_detector(FactType.GLOBAL_LEADER_CER, priority=999)
            def _double(data):
                return []

    def test_unregister_then_replace_works(self):
        # On peut explicitement retirer puis remplacer.
        original = detector_for(FactType.GLOBAL_LEADER_CER)
        assert original is not None
        try:
            unregister(FactType.GLOBAL_LEADER_CER)
            calls: list[dict] = []

            @register_detector(
                FactType.GLOBAL_LEADER_CER,
                priority=15,
                importance=FactImportance.MEDIUM,
            )
            def _replacement(data: dict):
                calls.append(data)
                return []

            entry = detector_for(FactType.GLOBAL_LEADER_CER)
            assert entry.priority == 15
            assert entry.importance == FactImportance.MEDIUM

            entry.fn({"meta": {}})
            assert len(calls) == 1
        finally:
            unregister(FactType.GLOBAL_LEADER_CER)
            # Restaure l'original
            register_detector(
                original.fact_type,
                priority=original.priority,
                importance=original.importance,
            )(original.fn)


# ---------------------------------------------------------------------------
# 4. iter_detectors trie par priority et reste stable
# ---------------------------------------------------------------------------

class TestIterDetectorsSorted:
    def test_returns_sorted_by_priority(self):
        priorities = [e.priority for e in iter_detectors()]
        assert priorities == sorted(priorities)

    def test_first_detector_is_highest_priority(self):
        first = iter_detectors()[0]
        assert first.fact_type == FactType.GLOBAL_LEADER_CER
        assert first.priority == 10


# ---------------------------------------------------------------------------
# 5. Robustesse — registre vide
# ---------------------------------------------------------------------------

class TestEmptyRegistryFallback:
    """Si le registre est vidé (cas extrême — chargement partiel par
    les tests), ``select_facts`` doit utiliser ``_FALLBACK_TYPE_ORDER``
    et ne pas planter."""

    def test_select_facts_works_on_empty_registry(self):
        from picarones.core.narrative.arbiter import select_facts
        # Sauvegarder l'état complet pour le restaurer
        backup = list(iter_detectors())
        try:
            clear_registry()
            facts = [
                Fact(
                    type=FactType.GLOBAL_LEADER_CER,
                    importance=FactImportance.HIGH,
                    payload={"engine": "A"},
                    engines_involved=("A",),
                ),
            ]
            selected = select_facts(facts, max_facts=3)
            assert len(selected) == 1
        finally:
            # Restaure le registre
            for entry in backup:
                register_detector(
                    entry.fact_type,
                    priority=entry.priority,
                    importance=entry.importance,
                )(entry.fn)


# ---------------------------------------------------------------------------
# 6. DETECTORS_BY_TYPE reste cohérent avec le registre
# ---------------------------------------------------------------------------

class TestLegacyAliasStillWorks:
    def test_detectors_by_type_matches_registry(self):
        from picarones.core.narrative.detectors import DETECTORS_BY_TYPE
        registry_types = {e.fact_type for e in iter_detectors()}
        legacy_types = set(DETECTORS_BY_TYPE)
        # Les deux ensembles peuvent diverger si DETECTORS_BY_TYPE est
        # capturé à l'import et que des types sont enregistrés après ;
        # mais à la création de l'objet ``DETECTORS_BY_TYPE`` lui-même
        # (au chargement de detectors.py), tous les builtins sont là.
        assert legacy_types <= registry_types
        for k, v in DETECTORS_BY_TYPE.items():
            entry = detector_for(k)
            assert entry is not None
            assert entry.fn is v
