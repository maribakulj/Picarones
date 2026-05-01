"""Tests Sprint 54 — Layout F1 par type de région.

Couvre :

1. ``Region`` validation (bbox invalide → ValueError, area calculée).
2. ``_iou_bbox`` mathématique (identité, disjoint, partiel).
3. **Cas standards** :
   - Layout parfait → F1 = 1
   - Mauvais type sur la même bbox → 0 TP pour ce type
   - Hallucination (région inventée) → FP
   - Région ratée (manquante) → FN
   - IoU sous le seuil → pas d'appariement
4. **Multi-type** : breakdown per_type cohérent avec les comptages
   globaux.
5. **Alignement greedy** : 2 hypothèses pour 1 GT → la meilleure
   gagne, l'autre devient FP.
6. **Cas dégénérés** : listes vides, None, IoU custom.
7. ``layout_f1`` raccourci équivalent à ``compute_layout_metrics["f1"]``.
"""

from __future__ import annotations

import pytest

from picarones.measurements.layout import (
    Region,
    _iou_bbox,
    compute_layout_metrics,
    layout_f1,
)


# ──────────────────────────────────────────────────────────────────────────
# 1. Region validation
# ──────────────────────────────────────────────────────────────────────────


class TestRegionDataclass:
    def test_valid_construction(self) -> None:
        r = Region("r1", "TextRegion", (0, 0, 100, 200))
        assert r.id == "r1"
        assert r.area == 20_000

    def test_invalid_bbox_raises(self) -> None:
        with pytest.raises(ValueError, match="bbox invalide"):
            Region("r1", "TextRegion", (0, 0, 0, 100))
        with pytest.raises(ValueError, match="bbox invalide"):
            Region("r1", "TextRegion", (0, 0, 100, -5))


# ──────────────────────────────────────────────────────────────────────────
# 2. IoU bbox
# ──────────────────────────────────────────────────────────────────────────


class TestIouBbox:
    def test_identical_bbox_iou_one(self) -> None:
        a = Region("a", "X", (0, 0, 100, 100))
        assert _iou_bbox(a, a) == pytest.approx(1.0)

    def test_disjoint_bbox_iou_zero(self) -> None:
        a = Region("a", "X", (0, 0, 100, 100))
        b = Region("b", "X", (200, 200, 50, 50))
        assert _iou_bbox(a, b) == 0.0

    def test_partial_overlap(self) -> None:
        # a = [0,0,100,100], b = [50,50,100,100]
        # intersection : 50x50 = 2500
        # union : 10000 + 10000 - 2500 = 17500
        # iou = 2500/17500 ≈ 0.143
        a = Region("a", "X", (0, 0, 100, 100))
        b = Region("b", "X", (50, 50, 100, 100))
        assert _iou_bbox(a, b) == pytest.approx(2500 / 17500)


# ──────────────────────────────────────────────────────────────────────────
# 3. Cas standards
# ──────────────────────────────────────────────────────────────────────────


class TestStandardCases:
    def test_perfect_layout(self) -> None:
        ref = [
            Region("r1", "TextRegion", (0, 0, 100, 100)),
            Region("r2", "MarginNote", (200, 0, 50, 100)),
        ]
        m = compute_layout_metrics(ref, list(ref))
        assert m["global"]["f1"] == pytest.approx(1.0)
        assert m["true_positives"] == 2
        assert m["false_positives"] == 0
        assert m["false_negatives"] == 0

    def test_wrong_type_breaks_match(self) -> None:
        # Même bbox mais type différent → pas d'appariement
        ref = [Region("r1", "TextRegion", (0, 0, 100, 100))]
        hyp = [Region("r1", "MarginNote", (0, 0, 100, 100))]
        m = compute_layout_metrics(ref, hyp)
        assert m["true_positives"] == 0
        assert m["false_negatives"] == 1
        assert m["false_positives"] == 1

    def test_hallucinated_region_is_fp(self) -> None:
        ref = [Region("r1", "TextRegion", (0, 0, 100, 100))]
        hyp = [
            Region("r1", "TextRegion", (0, 0, 100, 100)),
            Region("rX", "TextRegion", (500, 500, 50, 50)),  # inventée
        ]
        m = compute_layout_metrics(ref, hyp)
        assert m["true_positives"] == 1
        assert m["false_positives"] == 1
        assert m["hallucinated_regions"][0]["id"] == "rX"

    def test_missing_region_is_fn(self) -> None:
        ref = [
            Region("r1", "TextRegion", (0, 0, 100, 100)),
            Region("r2", "TextRegion", (200, 0, 100, 100)),
        ]
        hyp = [Region("r1", "TextRegion", (0, 0, 100, 100))]
        m = compute_layout_metrics(ref, hyp)
        assert m["true_positives"] == 1
        assert m["false_negatives"] == 1
        assert m["missed_regions"][0]["id"] == "r2"

    def test_iou_below_threshold_no_match(self) -> None:
        # Recouvrement IoU = 2500/17500 ≈ 0.14 < 0.5
        ref = [Region("r1", "TextRegion", (0, 0, 100, 100))]
        hyp = [Region("r1", "TextRegion", (50, 50, 100, 100))]
        m = compute_layout_metrics(ref, hyp, iou_threshold=0.5)
        assert m["true_positives"] == 0

    def test_iou_above_threshold_matches(self) -> None:
        # Recouvrement IoU = 6400/13600 ≈ 0.47, sous 0.5 mais sur 0.4
        ref = [Region("r1", "TextRegion", (0, 0, 100, 100))]
        hyp = [Region("r1", "TextRegion", (20, 20, 100, 100))]
        m_strict = compute_layout_metrics(ref, hyp, iou_threshold=0.5)
        m_loose = compute_layout_metrics(ref, hyp, iou_threshold=0.4)
        assert m_strict["true_positives"] == 0
        assert m_loose["true_positives"] == 1


# ──────────────────────────────────────────────────────────────────────────
# 4. Multi-type breakdown
# ──────────────────────────────────────────────────────────────────────────


class TestPerTypeBreakdown:
    def test_per_type_metrics(self) -> None:
        ref = [
            Region("r1", "TextRegion",  (0, 0, 100, 100)),
            Region("r2", "TextRegion",  (200, 0, 100, 100)),
            Region("r3", "MarginNote",  (0, 200, 100, 50)),
            Region("r4", "Header",      (0, 300, 200, 30)),
        ]
        hyp = [
            Region("r1", "TextRegion",  (0, 0, 100, 100)),       # match
            # r2 manquante → FN TextRegion
            Region("r3", "MarginNote",  (0, 200, 100, 50)),      # match
            Region("rX", "Footer",      (0, 400, 200, 30)),      # FP Footer
            # r4 Header manquante → FN Header
        ]
        m = compute_layout_metrics(ref, hyp)
        per_type = m["per_type"]
        # TextRegion : 1 TP + 1 FN → P=1, R=0.5, F1=2/3
        assert per_type["TextRegion"]["true_positives" if False else "f1"] == pytest.approx(2 / 3)
        # MarginNote : 1 TP, parfait
        assert per_type["MarginNote"]["f1"] == pytest.approx(1.0)
        # Header : 1 FN → P=0, R=0, F1=0
        assert per_type["Header"]["f1"] == 0.0
        # Footer : 1 FP → P=0, R=0
        assert per_type["Footer"]["f1"] == 0.0


# ──────────────────────────────────────────────────────────────────────────
# 5. Alignement greedy
# ──────────────────────────────────────────────────────────────────────────


class TestGreedyAlignment:
    def test_best_iou_wins(self) -> None:
        # GT : 1 région.  Hypothèse : 2 régions, l'une parfaite,
        # l'autre faiblement chevauchante.  La meilleure gagne.
        ref = [Region("r1", "TextRegion", (0, 0, 100, 100))]
        hyp = [
            Region("h_weak",   "TextRegion", (60, 60, 100, 100)),  # faible IoU
            Region("h_strong", "TextRegion", (0, 0, 100, 100)),    # parfait
        ]
        m = compute_layout_metrics(ref, hyp, iou_threshold=0.1)
        # Le strong gagne, le weak devient FP
        assert m["true_positives"] == 1
        assert m["false_positives"] == 1
        assert m["hallucinated_regions"][0]["id"] == "h_weak"


# ──────────────────────────────────────────────────────────────────────────
# 6. Cas dégénérés
# ──────────────────────────────────────────────────────────────────────────


class TestDegenerateCases:
    def test_both_empty(self) -> None:
        m = compute_layout_metrics([], [])
        assert m["global"]["f1"] == 0.0
        assert m["per_type"] == {}

    def test_only_reference_empty(self) -> None:
        m = compute_layout_metrics([], [Region("r1", "X", (0, 0, 10, 10))])
        assert m["false_positives"] == 1
        assert m["true_positives"] == 0

    def test_only_hypothesis_empty(self) -> None:
        m = compute_layout_metrics([Region("r1", "X", (0, 0, 10, 10))], [])
        assert m["false_negatives"] == 1
        assert m["true_positives"] == 0

    def test_none_inputs(self) -> None:
        m = compute_layout_metrics(None, None)
        assert m["global"]["f1"] == 0.0

    def test_dict_input_coerced(self) -> None:
        # L'utilisateur peut passer des dicts au format {id, type, bbox}
        ref = [{"id": "r1", "type": "TextRegion", "bbox": (0, 0, 100, 100)}]
        hyp = [{"id": "r1", "type": "TextRegion", "bbox": (0, 0, 100, 100)}]
        assert layout_f1(ref, hyp) == pytest.approx(1.0)


# ──────────────────────────────────────────────────────────────────────────
# 7. Type matching case-insensitive
# ──────────────────────────────────────────────────────────────────────────


class TestTypeNormalization:
    def test_type_case_insensitive(self) -> None:
        ref = [Region("r1", "TextRegion", (0, 0, 100, 100))]
        hyp = [Region("r1", "textregion", (0, 0, 100, 100))]
        assert layout_f1(ref, hyp) == pytest.approx(1.0)


# ──────────────────────────────────────────────────────────────────────────
# 8. Shortcut layout_f1
# ──────────────────────────────────────────────────────────────────────────


class TestShortcut:
    def test_shortcut_matches_full_call(self) -> None:
        ref = [
            Region("r1", "TextRegion", (0, 0, 100, 100)),
            Region("r2", "MarginNote", (200, 0, 50, 100)),
        ]
        hyp = [
            Region("r1", "TextRegion", (0, 0, 100, 100)),
            # r2 manquante
        ]
        full = compute_layout_metrics(ref, hyp)
        assert layout_f1(ref, hyp) == pytest.approx(full["global"]["f1"])
