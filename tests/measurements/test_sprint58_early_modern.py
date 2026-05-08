"""Tests Sprint 58 — marqueurs typographiques imprimé ancien.

Couvre :

1. ``get_category`` : caractères classés correctement dans les
   5 catégories (ligatures, long_s, dotless_i, ampersand,
   nasal_tildes) ; caractères non typographiques → ``None``.
2. ``detect_markers`` :
   - reconnaissance des marqueurs pré-composés
   - reconnaissance des séquences ``voyelle + U+0303`` comme
     nasal_tildes
   - texte vide / None
3. ``compute_early_modern_metrics`` :
   - **Diplomatique** : tous marqueurs préservés → preservation = 1
   - **Modernisant** : marqueurs remplacés (ſ→s, ﬁ→fi, ı→i, ã→a) →
     preservation faible
   - **Mixte** : breakdown per_category cohérent
   - cas dégénérés (GT sans marqueur, vide, None)
4. **Cas réaliste** : un document XVIIᵉ avec 5 marqueurs ; trois
   moteurs avec trois conventions différentes → discriminés.
5. ``missed_markers`` : liste les marqueurs ratés avec leur index
   et catégorie.
6. Intégration registre typé.
"""

from __future__ import annotations

import pytest

from picarones.evaluation.metrics.early_modern_typography import (
    AMPERSAND,
    DOTLESS_I,
    LIGATURES,
    LONG_S,
    NASAL_TILDE_PRECOMPOSED,
    compute_early_modern_metrics,
    detect_markers,
    early_modern_preservation,
    get_category,
)
from picarones.evaluation.metric_registry import compute_at_junction, select_metrics
from picarones.domain.artifacts import ArtifactType


# ──────────────────────────────────────────────────────────────────────────
# 1. get_category
# ──────────────────────────────────────────────────────────────────────────


class TestGetCategory:
    @pytest.mark.parametrize(
        "char,expected",
        [
            # Ligatures typographiques
            ("ﬁ", "ligatures"),
            ("ﬂ", "ligatures"),
            ("ﬀ", "ligatures"),
            ("ﬃ", "ligatures"),
            ("ﬆ", "ligatures"),
            # S long
            ("ſ", "long_s"),
            # i sans point
            ("ı", "dotless_i"),
            # Esperluette
            ("&", "ampersand"),
            # Tildes nasaux pré-composés
            ("ã", "nasal_tildes"),
            ("Ã", "nasal_tildes"),
            ("õ", "nasal_tildes"),
            ("ñ", "nasal_tildes"),
            ("ũ", "nasal_tildes"),
            # Caractères usuels → None
            ("a", None),
            ("é", None),
            ("s", None),
            ("i", None),
            (" ", None),
        ],
    )
    def test_categorize(self, char: str, expected: str | None) -> None:
        assert get_category(char) == expected

    def test_empty_input(self) -> None:
        assert get_category("") is None


# ──────────────────────────────────────────────────────────────────────────
# 2. detect_markers
# ──────────────────────────────────────────────────────────────────────────


class TestDetectMarkers:
    def test_detects_all_categories(self) -> None:
        text = "ﬁ ſ ı & ã"
        markers = detect_markers(text)
        cats = sorted(cat for _i, _m, cat in markers)
        assert cats == [
            "ampersand", "dotless_i", "ligatures", "long_s", "nasal_tildes",
        ]

    def test_detects_combining_tilde_sequence(self) -> None:
        # 'a' + U+0303 (combining tilde) → nasal_tildes
        text = "ã"
        markers = detect_markers(text)
        assert len(markers) == 1
        idx, marker, cat = markers[0]
        assert cat == "nasal_tildes"
        assert marker == "ã"
        assert idx == 0

    def test_preserves_order(self) -> None:
        text = "ı puis ſ puis ﬁ"
        markers = detect_markers(text)
        cats = [cat for _i, _m, cat in markers]
        assert cats == ["dotless_i", "long_s", "ligatures"]

    def test_empty_input(self) -> None:
        assert detect_markers("") == []
        assert detect_markers(None) == []

    def test_text_without_markers(self) -> None:
        assert detect_markers("hello world") == []


# ──────────────────────────────────────────────────────────────────────────
# 3. compute_early_modern_metrics — cas standards
# ──────────────────────────────────────────────────────────────────────────


class TestComputeMetrics:
    @pytest.fixture
    def gt(self) -> str:
        return "le ſerpent ﬁnement & ã ı"

    def test_diplomatic_full_preservation(self, gt: str) -> None:
        m = compute_early_modern_metrics(gt, gt)
        assert m["global_preservation"] == pytest.approx(1.0)
        assert m["n_markers_preserved"] == m["n_markers_reference"]
        assert m["missed_markers"] == []

    def test_modernizing_loses_typographic_markers(self, gt: str) -> None:
        # Tous les marqueurs typographiques remplacés par leurs
        # équivalents modernes ; & est conservé (rarement modernisé)
        hyp = "le serpent finement & a i"
        m = compute_early_modern_metrics(gt, hyp)
        # Esperluette préservée, le reste perdu
        assert m["per_category"]["ampersand"]["preservation"] == 1.0
        assert m["per_category"]["long_s"]["preservation"] == 0.0
        assert m["per_category"]["ligatures"]["preservation"] == 0.0
        assert m["per_category"]["dotless_i"]["preservation"] == 0.0
        assert m["per_category"]["nasal_tildes"]["preservation"] == 0.0
        # Global : 1/5 = 0.2
        assert m["global_preservation"] == pytest.approx(0.2)

    def test_mixed_engine_per_category(self, gt: str) -> None:
        # Préserve s long + ampersand, perd les autres
        hyp = "le ſerpent finement & a i"
        m = compute_early_modern_metrics(gt, hyp)
        assert m["per_category"]["long_s"]["preservation"] == 1.0
        assert m["per_category"]["ampersand"]["preservation"] == 1.0
        assert m["per_category"]["ligatures"]["preservation"] == 0.0
        # 2/5 = 0.4
        assert m["global_preservation"] == pytest.approx(0.4)


# ──────────────────────────────────────────────────────────────────────────
# 4. Cas dégénérés
# ──────────────────────────────────────────────────────────────────────────


class TestDegenerateCases:
    def test_gt_without_markers(self) -> None:
        m = compute_early_modern_metrics("hello", "hello")
        assert m["n_markers_reference"] == 0
        assert m["global_preservation"] == 0.0
        assert m["per_category"] == {}

    def test_empty_gt(self) -> None:
        m = compute_early_modern_metrics("", "anything")
        assert m["n_markers_reference"] == 0
        assert m["global_preservation"] == 0.0

    def test_none_inputs(self) -> None:
        m = compute_early_modern_metrics(None, None)
        assert m["n_markers_reference"] == 0

    def test_empty_hyp_with_markers_in_gt(self) -> None:
        m = compute_early_modern_metrics("ﬁ ſ", "")
        assert m["n_markers_preserved"] == 0
        assert m["global_preservation"] == 0.0
        assert len(m["missed_markers"]) == 2


# ──────────────────────────────────────────────────────────────────────────
# 5. missed_markers
# ──────────────────────────────────────────────────────────────────────────


class TestMissedMarkers:
    def test_missed_markers_have_index_and_category(self) -> None:
        gt = "ﬁ et ſ"
        # ſ remplacé par s, ﬁ remplacé par fi
        hyp = "fi et s"
        m = compute_early_modern_metrics(gt, hyp)
        # Les deux marqueurs sont ratés
        assert len(m["missed_markers"]) == 2
        cats = {entry["category"] for entry in m["missed_markers"]}
        assert cats == {"ligatures", "long_s"}
        # Chaque entrée a un index, marker, category
        for entry in m["missed_markers"]:
            assert "index" in entry
            assert "marker" in entry
            assert "category" in entry


# ──────────────────────────────────────────────────────────────────────────
# 6. Comptage exhaustif
# ──────────────────────────────────────────────────────────────────────────


class TestExhaustiveAccounting:
    def test_preserved_plus_missed_equals_total(self) -> None:
        gt = "ﬁ ſ ı & ã ﬂ ﬃ"
        hyp = "fi s i & a fl ﬃ"
        m = compute_early_modern_metrics(gt, hyp)
        assert (
            m["n_markers_preserved"] + len(m["missed_markers"])
            == m["n_markers_reference"]
        )


# ──────────────────────────────────────────────────────────────────────────
# 7. Sets exposés
# ──────────────────────────────────────────────────────────────────────────


class TestExposedSets:
    def test_ligatures_non_empty(self) -> None:
        assert len(LIGATURES) >= 5

    def test_categories_disjoint(self) -> None:
        # Les sets pré-composés sont disjoints (pas de chevauchement)
        sets = [LIGATURES, LONG_S, DOTLESS_I, AMPERSAND, NASAL_TILDE_PRECOMPOSED]
        for i, a in enumerate(sets):
            for b in sets[i + 1:]:
                assert a & b == frozenset(), (
                    f"Chevauchement entre catégories : {a & b!r}"
                )


# ──────────────────────────────────────────────────────────────────────────
# 8. Raccourci
# ──────────────────────────────────────────────────────────────────────────


class TestShortcut:
    def test_shortcut_matches_full_call(self) -> None:
        gt = "ﬁ ſ &"
        hyp = "fi s &"
        full = compute_early_modern_metrics(gt, hyp)
        assert early_modern_preservation(gt, hyp) == pytest.approx(
            full["global_preservation"],
        )


# ──────────────────────────────────────────────────────────────────────────
# 9. Intégration registre typé
# ──────────────────────────────────────────────────────────────────────────


class TestRegistryIntegration:
    def test_metric_registered(self) -> None:
        # Force l'import qui peuple le registre
        import picarones.evaluation.metrics.early_modern_typography  # noqa: F401

        selected = select_metrics(
            (ArtifactType.TEXT, ArtifactType.TEXT),
        )
        names = {spec.name for spec in selected}
        assert "early_modern_preservation" in names

    def test_compute_at_junction(self) -> None:
        out = compute_at_junction(
            "ﬁ ſ &", "ﬁ ſ &",
            (ArtifactType.TEXT, ArtifactType.TEXT),
        )
        assert out["early_modern_preservation"] == pytest.approx(1.0)
