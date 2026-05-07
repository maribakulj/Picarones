"""Tests Sprint 60 — numéraux romains transversaux.

Couvre :

1. ``roman_to_int`` / ``int_to_roman`` : conversion bidirectionnelle,
   tolérance casse + ``j`` médiéval, validation stricte des formes
   absurdes.
2. ``detect_roman_numerals`` : détection par regex, ``min_length``,
   tolérance casse, frontière de mot.
3. ``compute_roman_numeral_metrics`` : 5 statuts (strict_preserved,
   case_changed, j_dropped, converted_to_arabic, lost) discriminés
   sur scénarios standards.
4. Cas réalistes : manuscrit médiéval (mcclxxxij), imprimé ancien
   (Tome IV, chap. VII), moderne (Louis XIV, MCMXIV).
5. Comptage exhaustif : la somme des per_status égale n_total.
6. Cas dégénérés.
7. Intégration registre typé.
"""

from __future__ import annotations

import pytest

from picarones.core.metric_registry import compute_at_junction, select_metrics
from picarones.core.modules import ArtifactType
from picarones.evaluation.metrics.roman_numerals import (
    ALL_STATUSES,
    STATUS_CASE_CHANGED,
    STATUS_CONVERTED_TO_ARABIC,
    STATUS_J_DROPPED,
    STATUS_LOST,
    STATUS_STRICT_PRESERVED,
    compute_roman_numeral_metrics,
    detect_roman_numerals,
    int_to_roman,
    roman_numeral_strict_score,
    roman_numeral_value_score,
    roman_to_int,
)


# ──────────────────────────────────────────────────────────────────────────
# 1. roman_to_int / int_to_roman
# ──────────────────────────────────────────────────────────────────────────


class TestRomanToInt:
    @pytest.mark.parametrize(
        "s,expected",
        [
            ("I", 1), ("II", 2), ("III", 3),
            ("IV", 4), ("V", 5), ("IX", 9),
            ("X", 10), ("XIV", 14), ("XL", 40), ("XLIV", 44),
            ("L", 50), ("XC", 90), ("XCIX", 99),
            ("C", 100), ("CD", 400), ("D", 500), ("CM", 900),
            ("M", 1000), ("MCMXIV", 1914), ("MDCCLXXXIX", 1789),
        ],
    )
    def test_standard_uppercase(self, s: str, expected: int) -> None:
        assert roman_to_int(s) == expected

    @pytest.mark.parametrize(
        "s,expected",
        [
            ("i", 1), ("iv", 4), ("xiv", 14),
            ("mcmxiv", 1914), ("mdcclxxxix", 1789),
        ],
    )
    def test_lowercase_accepted(self, s: str, expected: int) -> None:
        assert roman_to_int(s) == expected

    @pytest.mark.parametrize(
        "s,expected",
        [
            ("ij", 2), ("iij", 3), ("iiij", 4),
            ("vij", 7), ("viij", 8),
            ("xij", 12), ("xiij", 13),
            ("mcclxxxij", 1282),
        ],
    )
    def test_medieval_j_final(self, s: str, expected: int) -> None:
        assert roman_to_int(s) == expected

    @pytest.mark.parametrize(
        "s",
        [
            "", "ABC", "XYZ", "VV", "LL", "DD",
            "IIIII", "XXXXX", "VVVV",
        ],
    )
    def test_invalid_returns_none(self, s: str) -> None:
        assert roman_to_int(s) is None

    def test_none_returns_none(self) -> None:
        assert roman_to_int(None) is None

    def test_medieval_relaxed_iiii(self) -> None:
        # IIII (forme médiévale) accepté
        assert roman_to_int("IIII") == 4
        assert roman_to_int("XXXX") == 40


class TestIntToRoman:
    @pytest.mark.parametrize(
        "n,expected",
        [
            (1, "I"), (4, "IV"), (9, "IX"), (14, "XIV"),
            (40, "XL"), (90, "XC"), (99, "XCIX"),
            (400, "CD"), (900, "CM"),
            (1789, "MDCCLXXXIX"), (1914, "MCMXIV"), (2024, "MMXXIV"),
        ],
    )
    def test_standard(self, n: int, expected: str) -> None:
        assert int_to_roman(n) == expected

    def test_round_trip(self) -> None:
        for n in (1, 7, 49, 99, 444, 1066, 1789, 1914, 2024):
            assert roman_to_int(int_to_roman(n)) == n

    def test_invalid_input(self) -> None:
        with pytest.raises(ValueError):
            int_to_roman(0)
        with pytest.raises(ValueError):
            int_to_roman(-5)


# ──────────────────────────────────────────────────────────────────────────
# 2. detect_roman_numerals
# ──────────────────────────────────────────────────────────────────────────


class TestDetectRomanNumerals:
    def test_detects_uppercase(self) -> None:
        text = "Louis XIV mourut en MDCCXV"
        out = detect_roman_numerals(text)
        nums = [(n, v) for _i, n, v in out]
        assert ("XIV", 14) in nums
        assert ("MDCCXV", 1715) in nums

    def test_detects_medieval_lowercase_with_j(self) -> None:
        text = "fait en lan mcclxxxij chap iij"
        out = detect_roman_numerals(text)
        nums = [(n, v) for _i, n, v in out]
        assert ("mcclxxxij", 1282) in nums
        assert ("iij", 3) in nums

    def test_returns_indices(self) -> None:
        text = "abc XIV def"
        out = detect_roman_numerals(text)
        assert out == [(4, "XIV", 14)]

    def test_min_length_filters_single_letter(self) -> None:
        # « I » seul ambigu (pronom anglais)
        text = "I am here in chapter II"
        out_default = detect_roman_numerals(text)
        nums_default = [n for _i, n, _v in out_default]
        assert "I" in nums_default
        assert "II" in nums_default
        out_filtered = detect_roman_numerals(text, min_length=2)
        nums_filtered = [n for _i, n, _v in out_filtered]
        assert "I" not in nums_filtered
        assert "II" in nums_filtered

    def test_word_boundary_blocks_embedded(self) -> None:
        # « VIVE » contient « V " et « IV " mais pas en frontière
        # de mot — pas de match.
        text = "VIVE le roi"
        out = detect_roman_numerals(text)
        nums = [n for _i, n, _v in out]
        assert "V" not in nums
        assert "IV" not in nums

    def test_invalid_chars_skipped(self) -> None:
        # « VV " est ramassé par la regex puis rejeté par la
        # validation — pas dans le résultat.
        text = "essai VV ici"
        out = detect_roman_numerals(text)
        assert out == []

    def test_empty_input(self) -> None:
        assert detect_roman_numerals("") == []
        assert detect_roman_numerals(None) == []

    def test_text_without_numerals(self) -> None:
        assert detect_roman_numerals("hello world without any") == []

    def test_preserves_order(self) -> None:
        text = "chap II puis chap V puis chap XIV"
        out = detect_roman_numerals(text)
        nums = [n for _i, n, _v in out]
        assert nums == ["II", "V", "XIV"]


# ──────────────────────────────────────────────────────────────────────────
# 3. compute_roman_numeral_metrics — 5 statuts
# ──────────────────────────────────────────────────────────────────────────


class TestClassificationStatuses:
    def test_strict_preserved(self) -> None:
        m = compute_roman_numeral_metrics("Louis XIV", "Louis XIV")
        assert m["per_status"][STATUS_STRICT_PRESERVED] == 1
        assert m["global_strict_score"] == pytest.approx(1.0)

    def test_case_changed(self) -> None:
        # GT en minuscule, hyp en majuscule (modernisation typo)
        m = compute_roman_numeral_metrics("Louis xiv", "Louis XIV")
        assert m["per_status"][STATUS_CASE_CHANGED] == 1
        assert m["per_status"][STATUS_STRICT_PRESERVED] == 0
        assert m["global_strict_score"] == 0.0
        assert m["global_value_score"] == pytest.approx(1.0)

    def test_j_dropped(self) -> None:
        # GT médiévale avec j final, hyp normalise en i
        m = compute_roman_numeral_metrics("chap iij", "chap iii")
        assert m["per_status"][STATUS_J_DROPPED] == 1
        assert m["global_value_score"] == pytest.approx(1.0)

    def test_converted_to_arabic(self) -> None:
        m = compute_roman_numeral_metrics("Louis XIV", "Louis 14")
        assert m["per_status"][STATUS_CONVERTED_TO_ARABIC] == 1
        assert m["global_value_score"] == pytest.approx(1.0)

    def test_lost(self) -> None:
        m = compute_roman_numeral_metrics("Louis XIV", "Louis le quatorzième")
        assert m["per_status"][STATUS_LOST] == 1
        assert m["global_value_score"] == 0.0

    def test_priority_strict_over_arabic(self) -> None:
        # Si l'hyp contient à la fois la forme stricte ET le chiffre
        # arabe, on classe strict (priorité).
        m = compute_roman_numeral_metrics(
            "Louis XIV", "Louis XIV (14 du nom)",
        )
        assert m["per_status"][STATUS_STRICT_PRESERVED] == 1
        assert m["per_status"][STATUS_CONVERTED_TO_ARABIC] == 0


class TestPerStatusBreakdown:
    def test_breakdown_sums_to_total(self) -> None:
        gt = "lan mcclxxxij chap iij Louis XIV en MCMXIV"
        hyp = "lan MCCLXXXIJ chap iii Louis 14 en MCMXIV"
        m = compute_roman_numeral_metrics(gt, hyp)
        assert sum(m["per_status"].values()) == m["n_numerals_reference"]

    def test_per_status_initializes_all(self) -> None:
        # Même sans aucun numéral, per_status doit lister tous les
        # statuts à 0.
        m = compute_roman_numeral_metrics("hello", "hello")
        assert set(m["per_status"].keys()) == set(ALL_STATUSES)

    def test_per_numeral_entries(self) -> None:
        m = compute_roman_numeral_metrics("XIV puis V", "XIV puis 5")
        per_num = m["per_numeral"]
        assert len(per_num) == 2
        for entry in per_num:
            assert "index" in entry
            assert "numeral" in entry
            assert "value" in entry
            assert "status" in entry

    def test_lost_numerals_listed(self) -> None:
        m = compute_roman_numeral_metrics("XIV V XX", "neant")
        lost = m["lost_numerals"]
        assert len(lost) == 3
        values = sorted(e["value"] for e in lost)
        assert values == [5, 14, 20]


# ──────────────────────────────────────────────────────────────────────────
# 4. Cas réalistes par période patrimoniale
# ──────────────────────────────────────────────────────────────────────────


class TestRealisticMedievalCharter:
    """Charte médiévale en minuscule avec ``j`` final."""

    @pytest.fixture
    def gt(self) -> str:
        return "fait en lan de grace mcclxxxij le iij iour de mai"

    def test_diplomatic(self, gt: str) -> None:
        m = compute_roman_numeral_metrics(gt, gt)
        assert m["global_strict_score"] == pytest.approx(1.0)

    def test_modernized_to_uppercase_no_j(self, gt: str) -> None:
        # Modernisation typo + orthographique : majuscule + j→i
        hyp = "fait en lan de grace MCCLXXXII le III iour de mai"
        m = compute_roman_numeral_metrics(gt, hyp)
        assert m["global_strict_score"] == 0.0
        assert m["global_value_score"] == pytest.approx(1.0)
        # Au moins un j_dropped (sur iij → III)
        assert m["per_status"][STATUS_J_DROPPED] >= 1

    def test_arabic_conversion(self, gt: str) -> None:
        hyp = "fait en lan de grace 1282 le 3 iour de mai"
        m = compute_roman_numeral_metrics(gt, hyp)
        assert m["per_status"][STATUS_CONVERTED_TO_ARABIC] == 2
        assert m["global_value_score"] == pytest.approx(1.0)


class TestRealisticEarlyModernPrint:
    """Tome IV, chap. VII (imprimé ancien)."""

    def test_diplomatic(self) -> None:
        gt = "Tome IV chap VII p XXIII"
        m = compute_roman_numeral_metrics(gt, gt)
        assert m["global_strict_score"] == pytest.approx(1.0)

    def test_arabic_conversion(self) -> None:
        gt = "Tome IV chap VII p XXIII"
        hyp = "Tome 4 chap 7 p 23"
        m = compute_roman_numeral_metrics(gt, hyp)
        assert m["per_status"][STATUS_CONVERTED_TO_ARABIC] == 3
        assert m["global_value_score"] == pytest.approx(1.0)


class TestRealisticModernSovereign:
    """Louis XIV, MCMXIV — moderne."""

    def test_diplomatic(self) -> None:
        gt = "Louis XIV mourut en MDCCXV ; la guerre en MCMXIV"
        m = compute_roman_numeral_metrics(gt, gt)
        assert m["global_strict_score"] == pytest.approx(1.0)

    def test_partial_arabic(self) -> None:
        # Le souverain reste en romain, la date passe en arabe
        gt = "Louis XIV mourut en MDCCXV"
        hyp = "Louis XIV mourut en 1715"
        m = compute_roman_numeral_metrics(gt, hyp)
        assert m["per_status"][STATUS_STRICT_PRESERVED] == 1
        assert m["per_status"][STATUS_CONVERTED_TO_ARABIC] == 1


# ──────────────────────────────────────────────────────────────────────────
# 5. Cas dégénérés
# ──────────────────────────────────────────────────────────────────────────


class TestDegenerateCases:
    def test_gt_without_numerals(self) -> None:
        m = compute_roman_numeral_metrics("hello world", "hello world")
        assert m["n_numerals_reference"] == 0
        assert m["global_strict_score"] == 0.0
        assert m["global_value_score"] == 0.0
        assert m["per_numeral"] == []

    def test_empty_gt(self) -> None:
        m = compute_roman_numeral_metrics("", "anything")
        assert m["n_numerals_reference"] == 0

    def test_none_inputs(self) -> None:
        m = compute_roman_numeral_metrics(None, None)
        assert m["n_numerals_reference"] == 0

    def test_empty_hyp_with_numerals(self) -> None:
        m = compute_roman_numeral_metrics("XIV V", "")
        assert m["n_numerals_reference"] == 2
        assert m["per_status"][STATUS_LOST] == 2
        assert m["global_value_score"] == 0.0


# ──────────────────────────────────────────────────────────────────────────
# 6. Raccourcis + registre typé
# ──────────────────────────────────────────────────────────────────────────


class TestShortcuts:
    def test_strict_shortcut_matches_full_call(self) -> None:
        gt = "Louis XIV"
        hyp = "Louis XIV"
        full = compute_roman_numeral_metrics(gt, hyp)
        assert roman_numeral_strict_score(gt, hyp) == pytest.approx(
            full["global_strict_score"],
        )

    def test_value_shortcut_matches_full_call(self) -> None:
        gt = "Louis XIV"
        hyp = "Louis 14"
        full = compute_roman_numeral_metrics(gt, hyp)
        assert roman_numeral_value_score(gt, hyp) == pytest.approx(
            full["global_value_score"],
        )


class TestRegistryIntegration:
    def test_metrics_registered(self) -> None:
        import picarones.measurements.roman_numerals  # noqa: F401

        selected = select_metrics(
            (ArtifactType.TEXT, ArtifactType.TEXT),
        )
        names = {spec.name for spec in selected}
        assert "roman_numeral_strict_score" in names
        assert "roman_numeral_value_score" in names

    def test_compute_at_junction_strict(self) -> None:
        out = compute_at_junction(
            "Louis XIV", "Louis XIV",
            (ArtifactType.TEXT, ArtifactType.TEXT),
        )
        assert out["roman_numeral_strict_score"] == pytest.approx(1.0)

    def test_compute_at_junction_value(self) -> None:
        out = compute_at_junction(
            "Louis XIV", "Louis 14",
            (ArtifactType.TEXT, ArtifactType.TEXT),
        )
        assert out["roman_numeral_strict_score"] == pytest.approx(0.0)
        assert out["roman_numeral_value_score"] == pytest.approx(1.0)
