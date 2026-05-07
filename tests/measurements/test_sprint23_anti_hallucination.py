"""Tests Sprint 23 — intégrité anti-hallucination du moteur narratif.

Le Sprint 23 ferme le trou méthodologique laissé par le Sprint 19 : le test
de traçabilité des nombres dans la synthèse rendue tolérait une whitelist
``{"95", "100"}`` de littéraux non-traçables au payload. Cette whitelist
est désormais vide ; toute valeur numérique apparaissant dans la synthèse
doit provenir du ``Fact.payload`` d'un détecteur.

Ce module vérifie quatre choses :

1. Les payloads des détecteurs concernés (``CONFIDENCE_WARNING``,
   ``PARETO_ALTERNATIVE``, ``COST_OUTLIER``) exposent bien les nouveaux
   champs (``confidence_level``, ``cost_unit_pages``).
2. Les templates FR/EN ne contiennent plus les littéraux ``95`` ni ``1000``
   en dehors d'un placeholder ``{...}``.
3. Le test de traçabilité reste vert avec une whitelist vide.
4. La stabilité du bootstrap est testée : deux seeds produisent des bornes
   d'IC à ±0,5 pp pour ``n=20`` documents — garantit que l'IC affiché
   dans le rapport est représentatif (sinon il faudrait passer
   ``n_iter=5000``).
5. Le pipeline narratif EN bout-en-bout produit des phrases anglaises
   bien formées (pas de placeholder non substitué) sur fixtures réalistes.
6. ``select_facts`` accepte un ``type_order`` custom et le respecte.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from picarones.measurements.narrative import (
    Fact,
    FactImportance,
    FactType,
    build_synthesis,
    select_facts,
)
from picarones.measurements.narrative.arbiter import DEFAULT_TYPE_ORDER
from picarones.evaluation.statistics import bootstrap_ci

ROOT = Path(__file__).parent.parent.parent
TEMPLATES_DIR = ROOT / "picarones" / "measurements" / "narrative" / "templates"


# ---------------------------------------------------------------------------
# Fixtures locales — minimum viable pour faire émettre chaque détecteur
# ---------------------------------------------------------------------------

def _full_data() -> dict:
    """Données qui déclenchent ``CONFIDENCE_WARNING`` (IC large) et le Pareto."""
    return {
        "ranking": [
            {"engine": "A", "mean_cer": 0.05, "wer": 0.10},
            {"engine": "B", "mean_cer": 0.06, "wer": 0.12},
            {"engine": "C", "mean_cer": 0.20, "wer": 0.30},
        ],
        "n_documents": 20,
        "statistics": {
            "bootstrap_cis": [
                # IC large pour A → confidence_warning
                {"engine": "A", "mean": 0.05, "ci_lower": 0.01, "ci_upper": 0.15},
                {"engine": "B", "mean": 0.06, "ci_lower": 0.05, "ci_upper": 0.07},
                {"engine": "C", "mean": 0.20, "ci_lower": 0.18, "ci_upper": 0.22},
            ],
        },
        "pareto": {
            "cost": {
                "front": ["A", "B"],
                "points": [
                    {"engine": "A", "cer": 0.05, "cost": 50.0},
                    {"engine": "B", "cer": 0.06, "cost": 5.0},  # alternative pas chère
                    {"engine": "C", "cer": 0.20, "cost": 300.0},  # cost outlier
                ],
            },
        },
    }


# ---------------------------------------------------------------------------
# 1. Payloads exposent les nouveaux champs
# ---------------------------------------------------------------------------

class TestPayloadsCarryFormerlyHardcodedConstants:
    def test_confidence_warning_payload_carries_confidence_level(self):
        from picarones.measurements.narrative.detectors import detect_confidence_warning

        facts = detect_confidence_warning(_full_data())
        assert facts, "fixture devrait déclencher au moins un confidence_warning"
        for f in facts:
            assert f.payload.get("confidence_level") == 95, (
                "Le seuil 95 doit être propagé dans le payload "
                "(plus de littéral hardcodé dans le template)."
            )

    def test_pareto_alternative_payload_carries_cost_unit(self):
        from picarones.measurements.narrative.detectors import detect_pareto_alternative

        facts = detect_pareto_alternative(_full_data())
        assert facts, "fixture devrait déclencher au moins un pareto_alternative"
        for f in facts:
            assert f.payload.get("cost_unit_pages") == 1000

    def test_cost_outlier_payload_carries_cost_unit(self):
        from picarones.measurements.narrative.detectors import detect_cost_outlier

        facts = detect_cost_outlier(_full_data())
        assert facts, "fixture devrait déclencher au moins un cost_outlier"
        for f in facts:
            assert f.payload.get("cost_unit_pages") == 1000


# ---------------------------------------------------------------------------
# 2. Les templates ne hardcodent plus les littéraux 95 et 1000
# ---------------------------------------------------------------------------

# Toute occurrence d'un nombre HORS d'un placeholder ``{...}`` est
# considérée comme un littéral hardcodé. On scanne en remplaçant d'abord
# tous les placeholders par un marqueur neutre.
_PLACEHOLDER_RE = re.compile(r"\{[^{}]+\}")
_NUMBER_RE = re.compile(r"\b\d+\b")


def _strip_placeholders(template: str) -> str:
    return _PLACEHOLDER_RE.sub("PLACEHOLDER", template)


@pytest.mark.parametrize("lang", ["fr", "en"])
class TestTemplatesNoHardcodedLiterals:
    def test_no_hardcoded_95(self, lang):
        import yaml

        path = TEMPLATES_DIR / f"{lang}.yaml"
        templates = yaml.safe_load(path.read_text(encoding="utf-8"))
        for key, tpl in templates.items():
            stripped = _strip_placeholders(tpl)
            numbers = _NUMBER_RE.findall(stripped)
            assert "95" not in numbers, (
                f"Template {lang}/{key} contient encore le littéral 95 ; "
                "doit utiliser {confidence_level}."
            )
            assert "1000" not in numbers, (
                f"Template {lang}/{key} contient encore le littéral 1000 ; "
                "doit utiliser {cost_unit_pages}."
            )


# ---------------------------------------------------------------------------
# 3. Pipeline complet produit une synthèse traçable, whitelist vide
# ---------------------------------------------------------------------------

class TestEndToEndWithEmptyWhitelist:
    @pytest.mark.parametrize("lang", ["fr", "en"])
    def test_synthesis_renders_without_unsubstituted_placeholders(self, lang):
        result = build_synthesis(_full_data(), lang)
        for sentence in result["sentences"]:
            assert "{" not in sentence and "}" not in sentence, (
                f"Placeholder non substitué dans la synthèse {lang} : {sentence!r}"
            )

    @pytest.mark.parametrize("lang", ["fr", "en"])
    def test_every_number_traceable_with_empty_whitelist(self, lang):
        from picarones.measurements.narrative import extract_numbers

        from tests.measurements.test_sprint19_narrative_engine import _numbers_in_payload

        result = build_synthesis(_full_data(), lang)
        allowed: set[str] = set()
        for f in result["facts"]:
            allowed |= _numbers_in_payload(f.get("payload", {}))

        unknown: list[tuple[str, str]] = []
        for sentence in result["sentences"]:
            for num in extract_numbers(sentence):
                num_norm = num.replace(",", ".")
                if num_norm not in allowed:
                    unknown.append((num, sentence))
        assert not unknown, (
            f"[{lang}] Nombres non traçables au payload : {unknown}"
        )


# ---------------------------------------------------------------------------
# 4. Stabilité du bootstrap entre seeds
# ---------------------------------------------------------------------------

class TestBootstrapStabilityAcrossSeeds:
    """Vérifie que ``bootstrap_ci`` à n_iter=1000 est suffisamment stable.

    Pour 20 documents avec un CER moyen ~5 %, l'écart entre deux seeds sur
    chacune des bornes (lower, upper) doit rester inférieur à 0,5 point de
    pourcentage de CER (= 0.005 en absolu). Si ce test échoue à l'avenir,
    cela signifie qu'il faut passer à ``n_iter=5000`` pour fiabiliser
    l'IC affiché dans le rapport.
    """

    def test_bootstrap_stable_for_typical_cer_distribution(self):
        # 20 valeurs de CER autour de 5 % — distribution réaliste.
        values = [
            0.02, 0.03, 0.04, 0.04, 0.045, 0.05, 0.05, 0.05, 0.055, 0.055,
            0.06, 0.06, 0.06, 0.065, 0.07, 0.07, 0.075, 0.08, 0.085, 0.10,
        ]
        lo1, hi1 = bootstrap_ci(values, n_iter=1000, seed=42)
        lo2, hi2 = bootstrap_ci(values, n_iter=1000, seed=7)
        assert abs(lo1 - lo2) < 0.005, (
            f"Borne basse instable entre seeds (Δ = {abs(lo1 - lo2):.4f}) ; "
            "envisager n_iter=5000."
        )
        assert abs(hi1 - hi2) < 0.005, (
            f"Borne haute instable entre seeds (Δ = {abs(hi1 - hi2):.4f}) ; "
            "envisager n_iter=5000."
        )

    def test_bootstrap_strictly_deterministic_same_seed(self):
        values = [0.01, 0.05, 0.1, 0.2]
        a = bootstrap_ci(values, n_iter=1000, seed=42)
        b = bootstrap_ci(values, n_iter=1000, seed=42)
        assert a == b, "Bootstrap doit être bit-à-bit reproductible sur seed identique."


# ---------------------------------------------------------------------------
# 5. select_facts respecte un type_order custom
# ---------------------------------------------------------------------------

class TestSelectFactsCustomTypeOrder:
    def _make_facts(self) -> list[Fact]:
        return [
            Fact(
                type=FactType.GLOBAL_LEADER_CER,
                importance=FactImportance.HIGH,
                payload={"engine": "A"},
                engines_involved=("A",),
            ),
            Fact(
                type=FactType.SPEED_WINNER,
                importance=FactImportance.HIGH,
                payload={"engine": "B"},
                engines_involved=("B",),
            ),
            Fact(
                type=FactType.PARETO_ALTERNATIVE,
                importance=FactImportance.HIGH,
                payload={"engine": "C"},
                engines_involved=("C",),
            ),
        ]

    def test_default_order_puts_global_leader_first(self):
        selected = select_facts(self._make_facts(), max_facts=3)
        assert selected[0].type == FactType.GLOBAL_LEADER_CER

    def test_custom_order_promotes_speed_winner(self):
        custom = (
            FactType.SPEED_WINNER,
            FactType.GLOBAL_LEADER_CER,
            FactType.PARETO_ALTERNATIVE,
        ) + tuple(t for t in DEFAULT_TYPE_ORDER if t not in {
            FactType.SPEED_WINNER,
            FactType.GLOBAL_LEADER_CER,
            FactType.PARETO_ALTERNATIVE,
        })
        selected = select_facts(self._make_facts(), max_facts=3, type_order=custom)
        assert selected[0].type == FactType.SPEED_WINNER, (
            "Avec un type_order custom plaçant SPEED_WINNER en premier, "
            "il doit ressortir avant GLOBAL_LEADER_CER à importance égale."
        )

    def test_unknown_types_in_custom_order_fall_to_end(self):
        # Un type_order réduit (ne mentionne que GLOBAL_LEADER_CER) ; les autres
        # types sont relégués à la fin sans crash.
        custom = (FactType.GLOBAL_LEADER_CER,)
        selected = select_facts(self._make_facts(), max_facts=3, type_order=custom)
        assert selected[0].type == FactType.GLOBAL_LEADER_CER
        assert len(selected) == 3
