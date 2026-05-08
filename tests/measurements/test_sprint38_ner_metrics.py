"""Tests Sprint 38 — métriques NER (couche de calcul pure).

Le module ``picarones.measurements.ner`` expose :

- la dataclass ``Entity`` ;
- ``compute_ner_metrics(ref, hyp, iou_threshold=0.5)`` qui aligne deux
  listes d'entités par IoU ≥ seuil et renvoie precision/recall/F1
  globaux et par catégorie, plus la liste des entités hallucinées
  (faux positifs) et manquées (faux négatifs) ;
- ``ner_f1`` enregistrée dans le registre typé Sprint 34 pour la
  jonction ``(ENTITIES, ENTITIES)``.

Les tests vérifient :

1. Cas parfait → F1 = 1, TP = N, FP = FN = 0.
2. Faux négatif seul → recall < 1, precision = 1.
3. Hallucination seule → precision < 1, et l'entité est listée dans
   ``hallucinated_entities``.
4. Mauvais label : pas de match même si les spans sont identiques.
5. IoU sous le seuil → pas de match.
6. IoU au-dessus du seuil → match malgré décalage léger.
7. Multi-catégorie : le détail par catégorie est correct.
8. Une entité ne peut être matchée qu'une fois (greedy IoU desc).
9. Cas dégénérés (listes vides, entités identiques avec deux fois la
   même position GT) sans crash.
10. Validation : start > end lève à la construction de Entity.
11. Le registre typé renvoie bien ner_f1 pour (ENTITIES, ENTITIES).
"""

from __future__ import annotations

import pytest

from picarones.evaluation.metric_registry import compute_at_junction, select_metrics
from picarones.domain.artifacts import ArtifactType
from picarones.evaluation.metrics.ner import Entity, compute_ner_metrics, ner_f1


# ──────────────────────────────────────────────────────────────────────────
# 1-3. Cas standards : parfait, FN seul, FP seul
# ──────────────────────────────────────────────────────────────────────────


class TestStandardCases:
    @pytest.fixture
    def ref(self) -> list[dict]:
        return [
            {"label": "PER", "start": 0, "end": 17, "text": "Marie de Bourgogne"},
            {"label": "PER", "start": 22, "end": 42, "text": "Charles le Téméraire"},
            {"label": "DATE", "start": 47, "end": 51, "text": "1477"},
        ]

    def test_perfect_match(self, ref: list[dict]) -> None:
        m = compute_ner_metrics(ref, list(ref))
        assert m["global"]["precision"] == pytest.approx(1.0)
        assert m["global"]["recall"] == pytest.approx(1.0)
        assert m["global"]["f1"] == pytest.approx(1.0)
        assert m["true_positives"] == 3
        assert m["false_positives"] == 0
        assert m["false_negatives"] == 0
        assert m["hallucinated_entities"] == []
        assert m["missed_entities"] == []

    def test_one_false_negative(self, ref: list[dict]) -> None:
        # On rate Charles
        hyp = [ref[0], ref[2]]
        m = compute_ner_metrics(ref, hyp)
        assert m["global"]["precision"] == pytest.approx(1.0)
        assert m["global"]["recall"] == pytest.approx(2 / 3)
        # F1 = 2 * 1 * (2/3) / (1 + 2/3) = 0.8
        assert m["global"]["f1"] == pytest.approx(0.8)
        assert len(m["missed_entities"]) == 1
        assert m["missed_entities"][0]["text"] == "Charles le Téméraire"

    def test_one_hallucination(self, ref: list[dict]) -> None:
        hyp = ref + [
            {"label": "PER", "start": 100, "end": 117, "text": "Personne Inventée"}
        ]
        m = compute_ner_metrics(ref, hyp)
        assert m["global"]["precision"] == pytest.approx(3 / 4)
        assert m["global"]["recall"] == pytest.approx(1.0)
        assert m["false_positives"] == 1
        assert len(m["hallucinated_entities"]) == 1
        assert m["hallucinated_entities"][0]["text"] == "Personne Inventée"


# ──────────────────────────────────────────────────────────────────────────
# 4. Mauvais label
# ──────────────────────────────────────────────────────────────────────────


class TestLabelMismatch:
    def test_different_label_no_match(self) -> None:
        ref = [{"label": "PER", "start": 0, "end": 5, "text": "Paris"}]
        hyp = [{"label": "LOC", "start": 0, "end": 5, "text": "Paris"}]
        m = compute_ner_metrics(ref, hyp)
        assert m["true_positives"] == 0
        assert m["false_negatives"] == 1
        assert m["false_positives"] == 1

    def test_label_case_insensitive(self) -> None:
        ref = [{"label": "PER", "start": 0, "end": 5, "text": "Marie"}]
        hyp = [{"label": "per", "start": 0, "end": 5, "text": "Marie"}]
        m = compute_ner_metrics(ref, hyp)
        assert m["true_positives"] == 1


# ──────────────────────────────────────────────────────────────────────────
# 5-6. IoU
# ──────────────────────────────────────────────────────────────────────────


class TestIoU:
    def test_iou_below_threshold_no_match(self) -> None:
        # GT span [0, 10), hyp span [9, 20) → overlap = 1, union = 20 → IoU = 0.05
        ref = [{"label": "PER", "start": 0, "end": 10, "text": "..."}]
        hyp = [{"label": "PER", "start": 9, "end": 20, "text": "..."}]
        m = compute_ner_metrics(ref, hyp, iou_threshold=0.5)
        assert m["true_positives"] == 0
        assert m["false_negatives"] == 1
        assert m["false_positives"] == 1

    def test_iou_above_threshold_matches(self) -> None:
        # GT [0, 10), hyp [0, 8) → overlap = 8, union = 10 → IoU = 0.8 ≥ 0.5
        ref = [{"label": "PER", "start": 0, "end": 10, "text": "..."}]
        hyp = [{"label": "PER", "start": 0, "end": 8, "text": "..."}]
        m = compute_ner_metrics(ref, hyp, iou_threshold=0.5)
        assert m["true_positives"] == 1

    def test_custom_iou_threshold(self) -> None:
        ref = [{"label": "PER", "start": 0, "end": 10, "text": "..."}]
        hyp = [{"label": "PER", "start": 5, "end": 15, "text": "..."}]
        # IoU = 5 / 15 ≈ 0.33
        assert compute_ner_metrics(ref, hyp, iou_threshold=0.5)["true_positives"] == 0
        assert compute_ner_metrics(ref, hyp, iou_threshold=0.3)["true_positives"] == 1


# ──────────────────────────────────────────────────────────────────────────
# 7. Multi-catégorie
# ──────────────────────────────────────────────────────────────────────────


class TestMultiCategory:
    def test_per_category_breakdown(self) -> None:
        ref = [
            {"label": "PER", "start": 0, "end": 5, "text": "A"},
            {"label": "PER", "start": 10, "end": 15, "text": "B"},
            {"label": "LOC", "start": 20, "end": 25, "text": "C"},
            {"label": "DATE", "start": 30, "end": 34, "text": "1789"},
        ]
        hyp = [
            {"label": "PER", "start": 0, "end": 5, "text": "A"},   # match
            # PER B raté → FN PER
            {"label": "LOC", "start": 20, "end": 25, "text": "C"},  # match
            {"label": "DATE", "start": 30, "end": 34, "text": "1789"},  # match
            {"label": "ORG", "start": 50, "end": 60, "text": "Hallu"},  # FP ORG
        ]
        m = compute_ner_metrics(ref, hyp)
        per_cat = m["per_category"]
        assert set(per_cat) == {"PER", "LOC", "DATE", "ORG"}
        # PER : 1 TP / 1 FN → precision = 1, recall = 0.5, F1 = 2/3
        assert per_cat["PER"]["precision"] == pytest.approx(1.0)
        assert per_cat["PER"]["recall"] == pytest.approx(0.5)
        assert per_cat["PER"]["f1"] == pytest.approx(2 / 3)
        assert per_cat["PER"]["support"] == 2
        # LOC et DATE parfaits
        assert per_cat["LOC"]["f1"] == pytest.approx(1.0)
        assert per_cat["DATE"]["f1"] == pytest.approx(1.0)
        # ORG : que des FP → precision = 0, support = 0
        assert per_cat["ORG"]["precision"] == pytest.approx(0.0)
        assert per_cat["ORG"]["support"] == 0


# ──────────────────────────────────────────────────────────────────────────
# 8. Greedy IoU décroissant — une entité ne peut être matchée qu'une fois
# ──────────────────────────────────────────────────────────────────────────


class TestGreedyAlignment:
    def test_each_entity_matched_once(self) -> None:
        """Si une hypothèse chevauche deux GT, elle ne peut matcher
        qu'une seule (la plus IoU élevée)."""
        ref = [
            {"label": "PER", "start": 0, "end": 10, "text": "A"},
            {"label": "PER", "start": 5, "end": 15, "text": "B"},
        ]
        hyp = [{"label": "PER", "start": 0, "end": 10, "text": "?"}]
        m = compute_ner_metrics(ref, hyp, iou_threshold=0.3)
        assert m["true_positives"] == 1
        assert m["false_negatives"] == 1
        # Pas de FP — l'hypothèse a été utilisée
        assert m["false_positives"] == 0

    def test_best_iou_wins(self) -> None:
        """Quand deux hypothèses sont candidates pour la même GT,
        celle avec l'IoU le plus élevé gagne."""
        ref = [{"label": "PER", "start": 0, "end": 10, "text": "X"}]
        hyp = [
            {"label": "PER", "start": 5, "end": 12, "text": "weak"},
            {"label": "PER", "start": 0, "end": 10, "text": "strong"},
        ]
        m = compute_ner_metrics(ref, hyp, iou_threshold=0.3)
        # 1 match (le strong) + 1 hallucination (le weak)
        assert m["true_positives"] == 1
        assert m["false_positives"] == 1
        assert m["hallucinated_entities"][0]["text"] == "weak"


# ──────────────────────────────────────────────────────────────────────────
# 9. Cas dégénérés
# ──────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_both_empty(self) -> None:
        m = compute_ner_metrics([], [])
        assert m["global"]["f1"] == 0.0
        assert m["per_category"] == {}
        assert m["true_positives"] == 0

    def test_only_reference_empty(self) -> None:
        m = compute_ner_metrics([], [{"label": "PER", "start": 0, "end": 5}])
        assert m["false_positives"] == 1
        assert m["global"]["precision"] == 0.0

    def test_only_hypothesis_empty(self) -> None:
        m = compute_ner_metrics([{"label": "PER", "start": 0, "end": 5}], [])
        assert m["false_negatives"] == 1
        assert m["global"]["recall"] == 0.0


# ──────────────────────────────────────────────────────────────────────────
# 10. Validation Entity
# ──────────────────────────────────────────────────────────────────────────


class TestEntityValidation:
    def test_invalid_span_raises(self) -> None:
        with pytest.raises(ValueError, match="span"):
            Entity(label="PER", start=10, end=5, text="x")

    def test_dict_to_entity_coercion(self) -> None:
        ref = [{"label": "PER", "start": 0, "end": 5, "text": "Marie"}]
        # passe un Entity côté hypothèse
        hyp = [Entity(label="PER", start=0, end=5, text="Marie")]
        m = compute_ner_metrics(ref, hyp)
        assert m["true_positives"] == 1


# ──────────────────────────────────────────────────────────────────────────
# 11. Intégration registre typé Sprint 34
# ──────────────────────────────────────────────────────────────────────────


class TestRegistryIntegration:
    def test_ner_f1_registered_for_entities_pair(self) -> None:
        # Force l'enregistrement
        import picarones.measurements.ner  # noqa: F401

        selected = select_metrics(
            (ArtifactType.ENTITIES, ArtifactType.ENTITIES),
        )
        names = {spec.name for spec in selected}
        assert "ner_f1" in names

    def test_compute_at_junction_uses_ner_f1(self) -> None:
        import picarones.measurements.ner  # noqa: F401

        ref = [{"label": "PER", "start": 0, "end": 5, "text": "Marie"}]
        hyp = [{"label": "PER", "start": 0, "end": 5, "text": "Marie"}]
        out = compute_at_junction(
            ref, hyp,
            (ArtifactType.ENTITIES, ArtifactType.ENTITIES),
        )
        assert out["ner_f1"] == pytest.approx(1.0)

    def test_ner_f1_shortcut_returns_same_as_compute(self) -> None:
        ref = [
            {"label": "PER", "start": 0, "end": 5, "text": "A"},
            {"label": "LOC", "start": 10, "end": 15, "text": "B"},
        ]
        hyp = [
            {"label": "PER", "start": 0, "end": 5, "text": "A"},
            # LOC raté
        ]
        full = compute_ner_metrics(ref, hyp)
        assert ner_f1(ref, hyp) == pytest.approx(full["global"]["f1"])
