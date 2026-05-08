"""Tests Sprint 57 — couverture MUFI (clôture axe A.II.3 philologique).

Couvre :

1. ``is_mufi_char`` :
   - caractères PUA (E000-F8FF) → True
   - Latin Extended-D (ꝑ, etc.) → True
   - lettres médiévales explicites (þ, ð, ƿ, ſ, æ, ƀ, ȝ…) → True
   - ligatures Alphabetic Presentation Forms (ﬁ, ﬂ) → True
   - lettres latines courantes (a, A, é) → False
   - chaîne vide → False
   - ``custom_chars`` étend la liste reconnue
2. ``compute_mufi_coverage`` :
   - GT diplomatique vs hyp diplomatique → coverage = 1
   - GT MUFI vs hyp modernisée (tout latin moderne) → coverage = 0
   - cas partiel : breakdown ``per_char`` cohérent
   - liste ``missed_chars`` exhaustive
3. **Cas dégénérés** :
   - GT vide / sans MUFI → coverage = 0
   - hyp vide → coverage = 0
   - GT et hyp identiques avec MUFI → coverage = 1
4. ``custom_chars`` : étend la détection (ex. accepter ``ñ``).
5. Coverage exhaustive : ``n_preserved + len(missed_chars) ==
   n_mufi_chars_reference`` quand toutes les positions sont
   classées.
6. Intégration registre typé : ``mufi_coverage`` enregistré pour
   ``(TEXT, TEXT)``.
"""

from __future__ import annotations

import pytest

from picarones.evaluation.metric_registry import compute_at_junction, select_metrics
from picarones.domain.artifacts import ArtifactType
from picarones.evaluation.metrics.mufi import (
    compute_mufi_coverage,
    is_mufi_char,
    mufi_coverage,
)


# ──────────────────────────────────────────────────────────────────────────
# 1. is_mufi_char
# ──────────────────────────────────────────────────────────────────────────


class TestIsMufiChar:
    @pytest.mark.parametrize(
        "char,expected",
        [
            # Lettres latines courantes → non MUFI
            ("a", False), ("Z", False), ("é", False), ("ç", False),
            ("ñ", False),  # caractère espagnol, pas MUFI par défaut
            ("0", False), (" ", False), ("", False),
            # Lettres médiévales explicites → MUFI
            ("þ", True), ("Þ", True), ("ð", True), ("Ð", True),
            ("ƿ", True), ("Ƿ", True), ("ſ", True),
            ("æ", True), ("Æ", True), ("œ", True), ("ø", True),
            ("ƀ", True), ("ȝ", True),
            # Latin Extended-D → MUFI
            ("ꝑ", True), ("ꝓ", True), ("ꝗ", True),
            # Alphabetic Presentation Forms → MUFI
            ("ﬁ", True), ("ﬂ", True),
            # Combining Diacritical Marks Supplement → MUFI
            # (U+1DC0 par exemple — combining dotted grave)
            ("᷀", True),
        ],
    )
    def test_known_chars(self, char: str, expected: bool) -> None:
        assert is_mufi_char(char) is expected

    def test_pua_range(self) -> None:
        # Quelques points dans la PUA E000-F8FF
        for cp in (0xE000, 0xE500, 0xF000, 0xF8FF):
            assert is_mufi_char(chr(cp)) is True

    def test_custom_chars_extend(self) -> None:
        # ñ n'est pas MUFI par défaut, mais devient MUFI si custom
        assert is_mufi_char("ñ") is False
        assert is_mufi_char("ñ", frozenset({"ñ"})) is True


# ──────────────────────────────────────────────────────────────────────────
# 2. compute_mufi_coverage
# ──────────────────────────────────────────────────────────────────────────


class TestComputeCoverage:
    def test_diplomatic_engine_full_coverage(self) -> None:
        gt = "þæt ƿæſ ꝑ ð"
        m = compute_mufi_coverage(gt, gt)
        assert m["coverage"] == pytest.approx(1.0)
        assert m["n_mufi_chars_preserved"] == m["n_mufi_chars_reference"]
        assert m["missed_chars"] == []

    def test_modernizing_engine_zero_coverage(self) -> None:
        gt = "þæt ƿæſ ꝑ ð"
        # Toutes les MUFI sont remplacées par des équivalents latins
        # modernes
        hyp = "tha waes per d"
        m = compute_mufi_coverage(gt, hyp)
        assert m["coverage"] == 0.0
        assert m["n_mufi_chars_preserved"] == 0

    def test_partial_coverage_with_per_char_breakdown(self) -> None:
        gt = "þæt ƿæſ ꝑ"
        # Partiel : þ, æ (1 sur 2), ꝑ préservés ; ƿ, ſ, æ (1/2) ratés
        hyp = "þæt was ꝑ"
        m = compute_mufi_coverage(gt, hyp)
        # Total MUFI dans GT : þ + æ + æ + ƿ + ſ + ꝑ = 6
        assert m["n_mufi_chars_reference"] == 6
        # Preserved : þ, premier æ, ꝑ → 3
        assert m["n_mufi_chars_preserved"] == 3
        per_char = m["per_char"]
        assert per_char["þ"]["coverage"] == 1.0
        assert per_char["ꝑ"]["coverage"] == 1.0
        assert per_char["ƿ"]["coverage"] == 0.0
        assert per_char["ſ"]["coverage"] == 0.0


# ──────────────────────────────────────────────────────────────────────────
# 3. Cas dégénérés
# ──────────────────────────────────────────────────────────────────────────


class TestDegenerateCases:
    def test_gt_without_mufi(self) -> None:
        m = compute_mufi_coverage("hello world", "hello world")
        assert m["n_mufi_chars_reference"] == 0
        assert m["coverage"] == 0.0
        assert m["per_char"] == {}

    def test_empty_gt(self) -> None:
        m = compute_mufi_coverage("", "anything")
        assert m["n_mufi_chars_reference"] == 0
        assert m["coverage"] == 0.0

    def test_none_inputs(self) -> None:
        m = compute_mufi_coverage(None, None)
        assert m["n_mufi_chars_reference"] == 0
        assert m["coverage"] == 0.0

    def test_empty_hyp_with_mufi_gt(self) -> None:
        m = compute_mufi_coverage("þæt", "")
        assert m["n_mufi_chars_preserved"] == 0
        assert m["coverage"] == 0.0
        # Tous les MUFI sont dans missed
        assert "þ" in m["missed_chars"]
        assert "æ" in m["missed_chars"]


# ──────────────────────────────────────────────────────────────────────────
# 4. Custom chars
# ──────────────────────────────────────────────────────────────────────────


class TestCustomChars:
    def test_custom_chars_count_in_total(self) -> None:
        # Sans custom : ñ n'est pas MUFI, donc texte sans MUFI
        assert compute_mufi_coverage("año", "año")["n_mufi_chars_reference"] == 0
        # Avec custom : ñ devient MUFI → 1 dans GT, 1 préservé
        m = compute_mufi_coverage("año", "año", custom_chars=["ñ"])
        assert m["n_mufi_chars_reference"] == 1
        assert m["coverage"] == pytest.approx(1.0)


# ──────────────────────────────────────────────────────────────────────────
# 5. Coverage exhaustive
# ──────────────────────────────────────────────────────────────────────────


class TestExhaustiveAccounting:
    def test_preserved_plus_missed_equals_total(self) -> None:
        gt = "þæt ƿæſ ꝑ ð ﬁ"
        hyp = "þæt was ꝑ d fi"
        m = compute_mufi_coverage(gt, hyp)
        # n_preserved + len(missed_chars) == n_total
        assert (
            m["n_mufi_chars_preserved"] + len(m["missed_chars"])
            == m["n_mufi_chars_reference"]
        )


# ──────────────────────────────────────────────────────────────────────────
# 6. Raccourci
# ──────────────────────────────────────────────────────────────────────────


class TestShortcut:
    def test_shortcut_matches_full_call(self) -> None:
        gt = "þæt ƿæſ ꝑ"
        hyp = "þæt was ꝑ"
        full = compute_mufi_coverage(gt, hyp)
        assert mufi_coverage(gt, hyp) == pytest.approx(full["coverage"])


# ──────────────────────────────────────────────────────────────────────────
# 7. Intégration registre typé
# ──────────────────────────────────────────────────────────────────────────


class TestRegistryIntegration:
    def test_metric_registered_for_text_text(self) -> None:
        # Force l'import qui peuple le registre
        import picarones.measurements.mufi  # noqa: F401

        selected = select_metrics(
            (ArtifactType.TEXT, ArtifactType.TEXT),
        )
        names = {spec.name for spec in selected}
        assert "mufi_coverage" in names

    def test_compute_at_junction(self) -> None:
        out = compute_at_junction(
            "þæt", "þæt",
            (ArtifactType.TEXT, ArtifactType.TEXT),
        )
        assert out["mufi_coverage"] == pytest.approx(1.0)
