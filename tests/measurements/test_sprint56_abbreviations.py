"""Tests Sprint 56 — score d'expansion d'abréviations médiévales.

Couvre :

1. ``detect_abbreviations`` :
   - reconnaissance des caractères Unicode dédiés (ꝑ, ꝓ, ⁊, etc.)
   - reconnaissance des séquences ``lettre + U+0303`` (p̃, q̃)
   - tolérance NFC/NFD
   - texte vide / None / sans abréviation
2. ``compute_abbreviation_metrics`` :
   - **Diplomatique** : forme abrégée préservée → strict=1, expansion=1
   - **Modernisant** : forme développée → strict=0, expansion=1
     (signal clé du plan d'évolution)
   - **Mauvais OCR** : ni l'abrégé ni la développée → 0/0
   - Mixte : 1 préservée, 1 développée → strict=0.5, expansion=1
   - GT sans abréviation → tous compteurs à 0, scores à 0
3. ``per_abbreviation`` détaille par abbreviation rencontrée.
4. **Cas réaliste du plan** : un GT avec ꝑ + ꝓ + ⁊ ; trois moteurs
   ayant adopté trois conventions différentes → strict/expansion
   permettent de les classer.
5. Frontière de mots pour les expansions courtes (« et », « us »).
6. Intégration registre typé : ``abbreviation_strict_score`` et
   ``abbreviation_expansion_score`` enregistrés pour ``(TEXT, TEXT)``.
"""

from __future__ import annotations

import pytest

from picarones.evaluation.metrics.abbreviations import (
    ABBREVIATION_EXPANSIONS,
    abbreviation_expansion_score,
    abbreviation_strict_score,
    compute_abbreviation_metrics,
    detect_abbreviations,
)
from picarones.evaluation.metric_registry import compute_at_junction, select_metrics
from picarones.domain.artifacts import ArtifactType


# ──────────────────────────────────────────────────────────────────────────
# 1. Détection
# ──────────────────────────────────────────────────────────────────────────


class TestDetection:
    def test_detects_unicode_abbreviations(self) -> None:
        text = "ꝑ ꝓ ꝗ ꝙ ⁊"
        out = detect_abbreviations(text)
        assert out == ["ꝑ", "ꝓ", "ꝗ", "ꝙ", "⁊"]

    def test_detects_combining_tilde_sequences(self) -> None:
        # p̃ = "p" + U+0303 (combining tilde)
        text = "p̃ q̃"
        out = detect_abbreviations(text)
        assert "p̃" in out
        assert "q̃" in out

    def test_preserves_duplicates(self) -> None:
        # Trois ꝑ → liste avec trois entrées
        out = detect_abbreviations("ꝑꝑꝑ")
        assert out == ["ꝑ", "ꝑ", "ꝑ"]

    def test_empty_text(self) -> None:
        assert detect_abbreviations("") == []
        assert detect_abbreviations(None) == []

    def test_text_without_abbreviation(self) -> None:
        assert detect_abbreviations("Lorem ipsum dolor") == []


# ──────────────────────────────────────────────────────────────────────────
# 2. Cas standards : diplomatique / modernisant / mauvais OCR
# ──────────────────────────────────────────────────────────────────────────


class TestStandardScenarios:
    @pytest.fixture
    def gt(self) -> str:
        # 4 abréviations différentes dans le GT
        return "ꝑ ad ꝓ et ⁊ ꝗ"

    def test_diplomatic_engine(self, gt: str) -> None:
        # Préservation des formes abrégées Unicode
        m = compute_abbreviation_metrics(gt, gt)
        assert m["strict_score"] == 1.0
        assert m["expansion_score"] == 1.0

    def test_modernizing_engine(self, gt: str) -> None:
        # Développement des abréviations
        hyp = "per ad pro et et qui"
        m = compute_abbreviation_metrics(gt, hyp)
        assert m["strict_score"] == 0.0
        assert m["expansion_score"] == pytest.approx(1.0)

    def test_bad_ocr(self, gt: str) -> None:
        # Ni la forme abrégée ni le développement n'apparaissent
        hyp = "x x x x x x"
        m = compute_abbreviation_metrics(gt, hyp)
        assert m["strict_score"] == 0.0
        assert m["expansion_score"] == 0.0

    def test_mixed_strategy(self, gt: str) -> None:
        # 2 préservées (ꝑ, ⁊), 2 développées (pro, qui)
        hyp = "ꝑ ad pro et ⁊ qui"
        m = compute_abbreviation_metrics(gt, hyp)
        # 4 abrégés au total : 2 strict (ꝑ et ⁊ préservés)
        assert m["n_strict_preserved"] == 2
        # Mais les 4 sont au moins par expansion
        assert m["n_expansion_preserved"] == 4


# ──────────────────────────────────────────────────────────────────────────
# 3. per_abbreviation détaillé
# ──────────────────────────────────────────────────────────────────────────


class TestPerAbbreviationBreakdown:
    def test_per_abbr_records(self) -> None:
        m = compute_abbreviation_metrics("ꝑ et ꝓ", "per et ꝓ")
        records = m["per_abbreviation"]
        assert len(records) == 2
        # ꝑ : pas préservé strict, mais "per" présent → expansion ok
        rec_p = next(r for r in records if r["abbr"] == "ꝑ")
        assert rec_p["strict_preserved"] is False
        assert rec_p["expansion_preserved"] is True
        # ꝓ : préservé strict (donc aussi expansion)
        rec_pro = next(r for r in records if r["abbr"] == "ꝓ")
        assert rec_pro["strict_preserved"] is True
        assert rec_pro["expansion_preserved"] is True


# ──────────────────────────────────────────────────────────────────────────
# 4. Cas dégénérés
# ──────────────────────────────────────────────────────────────────────────


class TestDegenerateCases:
    def test_gt_without_abbreviation(self) -> None:
        m = compute_abbreviation_metrics("Lorem ipsum dolor", "Lorem ipsum")
        assert m["n_abbreviations_in_reference"] == 0
        assert m["strict_score"] == 0.0
        assert m["expansion_score"] == 0.0

    def test_empty_inputs(self) -> None:
        m = compute_abbreviation_metrics("", "")
        assert m["n_abbreviations_in_reference"] == 0
        assert m["strict_score"] == 0.0

    def test_none_inputs(self) -> None:
        m = compute_abbreviation_metrics(None, None)
        assert m["n_abbreviations_in_reference"] == 0

    def test_empty_hypothesis_with_abbreviations_in_gt(self) -> None:
        m = compute_abbreviation_metrics("ꝑ ꝓ", "")
        assert m["n_abbreviations_in_reference"] == 2
        assert m["strict_score"] == 0.0
        assert m["expansion_score"] == 0.0


# ──────────────────────────────────────────────────────────────────────────
# 5. Frontière de mot pour expansions courtes
# ──────────────────────────────────────────────────────────────────────────


class TestShortExpansionWordBoundary:
    def test_et_requires_word_boundary(self) -> None:
        # GT a ⁊ → développement attendu = "et" (court → requiert
        # frontière de mot pour ne pas matcher trivialement
        # "permettre", etc.)
        # Ici l'hyp ne contient pas le mot "et" comme unité, donc
        # expansion_preserved = False.
        m = compute_abbreviation_metrics("⁊", "permettre quelque chose")
        assert m["expansion_score"] == 0.0

    def test_et_matches_at_word_boundary(self) -> None:
        m = compute_abbreviation_metrics("⁊", "fer et acier")
        assert m["expansion_score"] == 1.0


# ──────────────────────────────────────────────────────────────────────────
# 6. Raccourcis
# ──────────────────────────────────────────────────────────────────────────


class TestShortcuts:
    def test_strict_shortcut(self) -> None:
        full = compute_abbreviation_metrics("ꝑ ꝓ", "ꝑ pro")
        assert abbreviation_strict_score("ꝑ ꝓ", "ꝑ pro") == pytest.approx(
            full["strict_score"],
        )

    def test_expansion_shortcut(self) -> None:
        full = compute_abbreviation_metrics("ꝑ ꝓ", "ꝑ pro")
        assert abbreviation_expansion_score("ꝑ ꝓ", "ꝑ pro") == pytest.approx(
            full["expansion_score"],
        )


# ──────────────────────────────────────────────────────────────────────────
# 7. Intégration registre typé
# ──────────────────────────────────────────────────────────────────────────


class TestRegistryIntegration:
    def test_metrics_registered_for_text_text(self) -> None:
        # Force l'import qui peuple le registre
        import picarones.measurements.abbreviations  # noqa: F401

        selected = select_metrics(
            (ArtifactType.TEXT, ArtifactType.TEXT),
        )
        names = {spec.name for spec in selected}
        assert "abbreviation_strict_score" in names
        assert "abbreviation_expansion_score" in names

    def test_compute_at_junction_returns_both(self) -> None:
        out = compute_at_junction(
            "ꝑ et ꝓ",
            "ꝑ et ꝓ",
            (ArtifactType.TEXT, ArtifactType.TEXT),
        )
        assert out["abbreviation_strict_score"] == pytest.approx(1.0)
        assert out["abbreviation_expansion_score"] == pytest.approx(1.0)


# ──────────────────────────────────────────────────────────────────────────
# 8. Sanité de la table d'expansions
# ──────────────────────────────────────────────────────────────────────────


class TestExpansionTable:
    def test_table_non_empty(self) -> None:
        # Au moins les 6 abréviations principales (Capelli)
        assert len(ABBREVIATION_EXPANSIONS) >= 6

    def test_each_abbreviation_has_at_least_one_expansion(self) -> None:
        for abbr, expansions in ABBREVIATION_EXPANSIONS.items():
            assert len(expansions) >= 1, (
                f"L'abréviation {abbr!r} doit avoir au moins une expansion."
            )

    def test_all_expansions_are_lowercase(self) -> None:
        for expansions in ABBREVIATION_EXPANSIONS.values():
            for exp in expansions:
                assert exp == exp.lower(), (
                    f"Expansion {exp!r} doit être en minuscules."
                )
