"""Tests Sprint 31 — cas dégénérés du module ``picarones/core/pricing.py``.

Le test ``test_sprint20_pareto_pricing.py`` couvre les chemins
nominaux. Sprint 31 ajoute la couverture des **cas pathologiques** :

- pricing.yaml malformé (meta vide, engines absent, YAML invalide) ;
- ``estimate_cost`` avec entrées extrêmes (durée nulle, taux 0,
  modèle inconnu mêmes lookups) ;
- ``compute_pareto_front`` sur ensembles dégénérés (vide, 1 point,
  toutes valeurs identiques, NaN partout, objectifs incohérents).

Le but est de garantir qu'aucun de ces cas ne lève une exception
non documentée — un faux résultat doit être traité comme un signal
faible (``unknown``, ``[]``, ``score=1.0``), pas comme un crash qui
bloquerait la génération du rapport.
"""

from __future__ import annotations

import pytest

from picarones.evaluation.metrics.pricing import (
    EngineCost,
    PricingDefaults,
    build_costs_for_benchmark,
    estimate_cost,
    load_pricing_database,
)
from picarones.evaluation.statistics import compute_pareto_front


# ---------------------------------------------------------------------------
# 1. Robustesse de load_pricing_database
# ---------------------------------------------------------------------------

class TestLoadPricingDegenerate:
    def test_meta_only_file(self, tmp_path):
        path = tmp_path / "p.yaml"
        path.write_text("meta:\n  currency: EUR\n", encoding="utf-8")
        defaults, table = load_pricing_database(path)
        assert defaults.currency == "EUR"
        assert table == {}

    def test_engines_only_file(self, tmp_path):
        path = tmp_path / "p.yaml"
        path.write_text("engines:\n  fake: {type: local}\n", encoding="utf-8")
        defaults, table = load_pricing_database(path)
        # Defaults retombent sur les constantes par défaut
        assert isinstance(defaults, PricingDefaults)
        assert "fake" in table

    def test_completely_empty_yaml(self, tmp_path):
        path = tmp_path / "p.yaml"
        path.write_text("", encoding="utf-8")
        defaults, table = load_pricing_database(path)
        assert isinstance(defaults, PricingDefaults)
        assert table == {}

    def test_invalid_yaml_returns_empty(self, tmp_path):
        path = tmp_path / "p.yaml"
        path.write_text("meta: : :\n  invalid", encoding="utf-8")
        defaults, table = load_pricing_database(path)
        # Comportement dégradé documenté : retourne PricingDefaults() + {}
        assert isinstance(defaults, PricingDefaults)
        assert table == {}


# ---------------------------------------------------------------------------
# 2. estimate_cost cas extrêmes
# ---------------------------------------------------------------------------

class TestEstimateCostDegenerate:
    def test_unknown_engine_returns_unknown_type(self):
        cost = estimate_cost("ne-pas-exister", table={}, defaults=PricingDefaults())
        assert isinstance(cost, EngineCost)
        assert cost.type == "unknown"
        assert cost.assumptions  # message d'avertissement présent

    def test_zero_seconds_per_page_does_not_raise(self):
        # Un moteur infiniment rapide (théorique) ne doit pas crasher
        cost = estimate_cost(
            "tesseract",
            measured_seconds_per_page=0.0,
        )
        # On accepte un coût nul ou indéfini, mais pas d'exception.
        assert isinstance(cost, EngineCost)

    def test_negative_seconds_per_page_does_not_raise(self):
        # Anti-glitch : un timer qui descend (valeur négative) ne doit
        # pas crasher. Le module ne clamp PAS la valeur (comportement
        # documenté ici) — les vrais runs ne produisent jamais de
        # durées négatives. Le test vérifie juste l'absence d'exception.
        cost = estimate_cost(
            "tesseract",
            measured_seconds_per_page=-1.0,
        )
        assert isinstance(cost, EngineCost)
        # On accepte n'importe quelle valeur de coût ; ce qui compte
        # c'est qu'aucune exception ne soit levée.

    def test_hourly_rate_override_zero(self):
        cost = estimate_cost(
            "tesseract",
            measured_seconds_per_page=10.0,
            hourly_rate_override_eur=0.0,
        )
        # Coût zéro acceptable (cas universitaire / matériel amorti)
        if cost.cost_per_1k_pages_eur is not None:
            assert cost.cost_per_1k_pages_eur == pytest.approx(0.0)

    def test_as_dict_serializable(self):
        cost = estimate_cost("tesseract", measured_seconds_per_page=2.0)
        d = cost.as_dict()
        assert isinstance(d, dict)
        # Quelques clefs publiques attendues
        for k in ("engine_key", "type", "cost_per_1k_pages_eur"):
            assert k in d


# ---------------------------------------------------------------------------
# 3. compute_pareto_front cas pathologiques
# ---------------------------------------------------------------------------

class TestParetoFrontDegenerate:
    def test_single_point_is_dominant(self):
        front = compute_pareto_front(
            [{"engine": "A", "cer": 0.05, "cost": 10.0}],
            objectives=("cer", "cost"),
        )
        assert front == ["A"]

    def test_all_points_identical_keep_all(self):
        # Quand plusieurs moteurs ont les mêmes coordonnées, tous sont
        # sur le front (aucun ne domine strictement les autres).
        pts = [
            {"engine": "A", "cer": 0.05, "cost": 10.0},
            {"engine": "B", "cer": 0.05, "cost": 10.0},
            {"engine": "C", "cer": 0.05, "cost": 10.0},
        ]
        front = set(compute_pareto_front(pts, objectives=("cer", "cost")))
        assert front == {"A", "B", "C"}

    def test_nan_values_do_not_crash(self):
        # NaN n'est pas filtré explicitement (comparaisons NaN < x = False).
        # On vérifie juste l'absence d'exception ; B doit toujours figurer
        # sur le front comme point Pareto-valide.
        pts = [
            {"engine": "A", "cer": float("nan"), "cost": 10.0},
            {"engine": "B", "cer": 0.05, "cost": 10.0},
        ]
        front = compute_pareto_front(pts, objectives=("cer", "cost"))
        assert isinstance(front, list)
        assert "B" in front

    def test_missing_objective_field_skipped(self):
        pts = [
            {"engine": "A", "cer": 0.05},  # cost manquant
            {"engine": "B", "cer": 0.06, "cost": 5.0},
        ]
        front = compute_pareto_front(pts, objectives=("cer", "cost"))
        assert "B" in front
        # A n'a pas pu être évalué → exclu
        assert "A" not in front

    def test_empty_points_returns_empty_list(self):
        assert compute_pareto_front([], objectives=("cer", "cost")) == []

    def test_single_objective_pareto_is_min(self):
        # Avec une seule dimension, Pareto = optimum global
        pts = [
            {"engine": "A", "cer": 0.10},
            {"engine": "B", "cer": 0.05},  # le meilleur
            {"engine": "C", "cer": 0.20},
        ]
        front = compute_pareto_front(pts, objectives=("cer",))
        assert front == ["B"]

    def test_maximize_dimension(self):
        # Pour un objectif à maximiser (ex. accuracy), le front prend
        # les valeurs les plus hautes.
        pts = [
            {"engine": "A", "accuracy": 0.90, "cost": 10.0},
            {"engine": "B", "accuracy": 0.95, "cost": 10.0},
            {"engine": "C", "accuracy": 0.99, "cost": 10.0},
        ]
        front = set(compute_pareto_front(
            pts,
            objectives=("accuracy", "cost"),
            minimize=(False, True),
        ))
        # C maximise accuracy à coût égal — il domine A et B
        assert "C" in front

    def test_invalid_minimize_length_raises(self):
        pts = [{"engine": "A", "cer": 0.05, "cost": 10.0}]
        with pytest.raises((ValueError, AssertionError)):
            compute_pareto_front(
                pts,
                objectives=("cer", "cost"),
                minimize=(True,),  # longueur ≠ objectives
            )


# ---------------------------------------------------------------------------
# 4. build_costs_for_benchmark — cas dégénérés
# ---------------------------------------------------------------------------

class TestBuildCostsDegenerate:
    def test_empty_engine_list_returns_empty(self):
        costs = build_costs_for_benchmark([], {})
        assert costs == {}

    def test_engines_without_measured_durations(self, tmp_path):
        # pricing.yaml minimal, juste un moteur local.
        path = tmp_path / "p.yaml"
        path.write_text(
            "meta:\n  currency: EUR\n"
            "engines:\n  tesseract: {type: local, indicative_seconds_per_page: 2.0}\n",
            encoding="utf-8",
        )
        # Pas de durée mesurée pour tesseract — l'algo retombe sur la
        # valeur indicative de la table.
        engines_summary = [{"name": "tesseract"}]
        durations_by_engine: dict[str, float] = {}
        costs = build_costs_for_benchmark(
            engines_summary, durations_by_engine, pricing_path=path,
        )
        assert "tesseract" in costs
        # ``costs`` retourne des dicts (cf. EngineCost.as_dict())
        assert costs["tesseract"]["type"] == "local"
