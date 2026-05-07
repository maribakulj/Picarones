"""Tests Sprint 55 — Précision par bloc Unicode.

Couvre :

1. ``get_block`` : caractères des blocs principaux correctement
   classifiés ; caractères inconnus → ``"Other"``.
2. ``compute_unicode_block_accuracy`` :
   - Texte identique → toutes les accuracies à 1.0
   - Texte vide → résultat dégénéré sans crash
   - Substitutions ciblées par bloc (ASCII préservé, présentation
     latine perdue) → cas réaliste du plan
   - Insertions et suppressions
3. **Cas réaliste du plan d'évolution** : OCR modernisant remplace
   ``ſ`` par ``s`` et ``ﬁ`` par ``fi`` → 100 % Latin de Base mais
   0 % Présentation latine et 0 % Latin Extended-A.
4. ``unicode_block_global_accuracy`` raccourci équivalent à
   ``compute["global_accuracy"]``.
5. **Intégration registre typé** : ``unicode_block_global_accuracy``
   sélectionnée pour la jonction ``(TEXT, TEXT)``.
"""

from __future__ import annotations

import pytest

from picarones.evaluation.metric_registry import compute_at_junction, select_metrics
from picarones.domain.artifacts import ArtifactType
from picarones.measurements.unicode_blocks import (
    compute_unicode_block_accuracy,
    get_block,
    unicode_block_global_accuracy,
)


# ──────────────────────────────────────────────────────────────────────────
# 1. get_block
# ──────────────────────────────────────────────────────────────────────────


class TestGetBlock:
    @pytest.mark.parametrize(
        "char,expected_block",
        [
            ("a", "Basic Latin"),
            ("A", "Basic Latin"),
            (" ", "Basic Latin"),
            ("é", "Latin-1 Supplement"),
            ("ç", "Latin-1 Supplement"),
            ("ƒ", "Latin Extended-B"),
            ("ſ", "Latin Extended-A"),     # s long médiéval
            ("ﬁ", "Alphabetic Presentation Forms"),  # ligature fi
            ("ﬂ", "Alphabetic Presentation Forms"),
            ("́", "Combining Diacritical Marks"),  # ́ combinant aigu
        ],
    )
    def test_known_blocks(self, char: str, expected_block: str) -> None:
        assert get_block(char) == expected_block

    def test_empty_string_returns_other(self) -> None:
        assert get_block("") == "Other"

    def test_unknown_char_returns_other(self) -> None:
        # Émoji → pas dans la table patrimoniale
        assert get_block("🦊") == "Other"


# ──────────────────────────────────────────────────────────────────────────
# 2. compute_unicode_block_accuracy — cas généraux
# ──────────────────────────────────────────────────────────────────────────


class TestComputeAccuracy:
    def test_identical_text_full_accuracy(self) -> None:
        m = compute_unicode_block_accuracy("hello world", "hello world")
        assert m["global_accuracy"] == pytest.approx(1.0)
        assert m["per_block"]["Basic Latin"]["accuracy"] == pytest.approx(1.0)
        assert m["per_block"]["Basic Latin"]["correct"] == 11
        assert m["per_block"]["Basic Latin"]["total"] == 11

    def test_empty_reference(self) -> None:
        m = compute_unicode_block_accuracy("", "anything")
        assert m["per_block"] == {}
        assert m["global_accuracy"] == 0.0
        assert m["n_chars_reference"] == 0

    def test_empty_hypothesis(self) -> None:
        m = compute_unicode_block_accuracy("hello", "")
        assert m["global_accuracy"] == 0.0
        assert m["per_block"]["Basic Latin"]["correct"] == 0
        assert m["per_block"]["Basic Latin"]["total"] == 5

    def test_none_inputs(self) -> None:
        m = compute_unicode_block_accuracy(None, None)
        assert m["per_block"] == {}
        assert m["global_accuracy"] == 0.0

    def test_partial_substitution_per_block(self) -> None:
        # Les é (Latin-1 Sup) sont préservés ; les a (Basic Latin)
        # sont remplacés par X.
        gt = "éaéaéa"
        hyp = "éXéXéX"
        m = compute_unicode_block_accuracy(gt, hyp)
        # Latin-1 Sup : 3 é correctes
        assert m["per_block"]["Latin-1 Supplement"]["accuracy"] == pytest.approx(1.0)
        # Basic Latin : 0/3 (les a sont substitués)
        assert m["per_block"]["Basic Latin"]["accuracy"] == 0.0


# ──────────────────────────────────────────────────────────────────────────
# 3. Cas réaliste du plan d'évolution
# ──────────────────────────────────────────────────────────────────────────


class TestRealisticModernization:
    def test_modernizing_ocr_loses_presentation_forms(self) -> None:
        """OCR qui remplace ſ par s et ﬁ par fi → 100 % Latin de Base
        préservé, mais 0 % de Présentation latine et de Latin
        Extended-A.  C'est l'illustration directe du plan : "ce moteur
        restitue 95 % du Latin de Base mais 12 % de présentation
        latine".
        """
        gt = "le ſerpent ﬁnement"
        ocr_modern = "le serpent finement"
        m = compute_unicode_block_accuracy(gt, ocr_modern)
        # Présentation latine (ﬁ remplacée) : 0%
        assert m["per_block"]["Alphabetic Presentation Forms"]["accuracy"] == 0.0
        # Latin Extended-A (ſ remplacé) : 0% (1 occurrence dans "ſerpent")
        assert m["per_block"]["Latin Extended-A"]["accuracy"] == 0.0
        # Basic Latin : préservé à 100% (les espaces, lettres ASCII)
        assert m["per_block"]["Basic Latin"]["accuracy"] == pytest.approx(1.0)


# ──────────────────────────────────────────────────────────────────────────
# 4. Insertions / suppressions
# ──────────────────────────────────────────────────────────────────────────


class TestInsertionDeletion:
    def test_inserted_char_does_not_count(self) -> None:
        # GT = "abc", hyp = "aXbc" : insertion de X → toutes les
        # positions GT restent correctement restituées.
        m = compute_unicode_block_accuracy("abc", "aXbc")
        assert m["per_block"]["Basic Latin"]["accuracy"] == pytest.approx(1.0)

    def test_deletion_lowers_accuracy(self) -> None:
        # GT = "abc", hyp = "ac" : "b" est supprimé → 2/3 préservés.
        m = compute_unicode_block_accuracy("abc", "ac")
        assert m["per_block"]["Basic Latin"]["correct"] == 2
        assert m["per_block"]["Basic Latin"]["total"] == 3
        assert m["per_block"]["Basic Latin"]["accuracy"] == pytest.approx(2 / 3)


# ──────────────────────────────────────────────────────────────────────────
# 5. Coverage — tous les caractères GT classés
# ──────────────────────────────────────────────────────────────────────────


class TestCoverage:
    def test_total_chars_match_reference_length(self) -> None:
        gt = "Hello, æther — vol. ﬁ. é"
        m = compute_unicode_block_accuracy(gt, gt)
        total = sum(d["total"] for d in m["per_block"].values())
        assert total == len(gt)
        assert m["n_chars_reference"] == len(gt)


# ──────────────────────────────────────────────────────────────────────────
# 6. Raccourci global
# ──────────────────────────────────────────────────────────────────────────


class TestShortcut:
    def test_shortcut_matches_full_call(self) -> None:
        gt = "le ſerpent ﬁnement"
        ocr = "le serpent finement"
        full = compute_unicode_block_accuracy(gt, ocr)
        assert unicode_block_global_accuracy(gt, ocr) == pytest.approx(
            full["global_accuracy"],
        )


# ──────────────────────────────────────────────────────────────────────────
# 7. Intégration registre typé
# ──────────────────────────────────────────────────────────────────────────


class TestRegistryIntegration:
    def test_metric_registered_for_text_text(self) -> None:
        # Force l'import qui peuple le registre
        import picarones.measurements.unicode_blocks  # noqa: F401

        selected = select_metrics(
            (ArtifactType.TEXT, ArtifactType.TEXT),
        )
        names = {spec.name for spec in selected}
        assert "unicode_block_global_accuracy" in names

    def test_compute_at_junction(self) -> None:
        out = compute_at_junction(
            "hello",
            "hello",
            (ArtifactType.TEXT, ArtifactType.TEXT),
        )
        assert out["unicode_block_global_accuracy"] == pytest.approx(1.0)
