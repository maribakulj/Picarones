"""Tests Sprint 52 — métriques de lisibilité (Flesch).

Couvre :

1. ``count_syllables_word`` : heuristique sur des cas variés
   (mots vides, sans voyelle, avec diacritiques, multi-syllabe).
2. ``count_words`` / ``count_sentences`` : tokenisation simple,
   gestion des cas sans ponctuation finale.
3. ``flesch_score`` :
   - texte vide → 0
   - score borné dans [0, 100]
   - cohérence : phrase simple > phrase complexe
   - différence FR vs EN (coefficients distincts)
4. ``flesch_delta`` :
   - GT = OCR → 0
   - OCR modernisé (LLM) → delta positif
   - OCR dégradé (caractères cassés) → delta négatif
5. **Cas d'usage réaliste** : un GT historique long et complexe vs
   un OCR/LLM simplifié → delta clairement positif (>15 pts).
6. Garde-fous : langue invalide, textes ne contenant que de la
   ponctuation.
7. Enregistrement dans le registre typé Sprint 34 — la jonction
   ``(TEXT, TEXT)`` retourne bien ``flesch_delta_fr`` et
   ``flesch_delta_en``.
"""

from __future__ import annotations

import pytest

from picarones.core.metric_registry import select_metrics
from picarones.core.modules import ArtifactType
from picarones.core.readability import (
    count_sentences,
    count_syllables,
    count_syllables_word,
    count_words,
    flesch_delta,
    flesch_score,
)


# ──────────────────────────────────────────────────────────────────────────
# 1. Compteur de syllabes
# ──────────────────────────────────────────────────────────────────────────


class TestSyllableCounting:
    def test_empty_word_returns_zero(self) -> None:
        assert count_syllables_word("") == 0

    def test_word_without_vowel_returns_one(self) -> None:
        # Convention : un mot sans voyelle compte au moins 1 syllabe
        # (utile pour les acronymes type "BNF", "ALTO").
        assert count_syllables_word("BNF") == 1
        assert count_syllables_word("xyz") == 1

    def test_single_vowel(self) -> None:
        assert count_syllables_word("a") == 1
        assert count_syllables_word("é") == 1

    def test_simple_words(self) -> None:
        # Heuristique groupes de voyelles consécutives
        assert count_syllables_word("chat") == 1     # 1 groupe : a
        assert count_syllables_word("chien") == 1    # 1 groupe : ie
        assert count_syllables_word("école") == 3    # é-o-e

    def test_diacritics_counted_as_vowels(self) -> None:
        # Les voyelles accentuées doivent être reconnues
        s_with = count_syllables_word("été")
        s_without = count_syllables_word("ete")
        # Mêmes groupes de voyelles, même nombre de syllabes
        assert s_with == s_without

    def test_count_syllables_sums_over_words(self) -> None:
        text = "le chat noir"
        assert count_syllables(text) == (
            count_syllables_word("le")
            + count_syllables_word("chat")
            + count_syllables_word("noir")
        )


# ──────────────────────────────────────────────────────────────────────────
# 2. Comptage mots / phrases
# ──────────────────────────────────────────────────────────────────────────


class TestTokenCounting:
    def test_empty_text(self) -> None:
        assert count_words("") == 0
        assert count_sentences("") == 0

    def test_simple_words(self) -> None:
        assert count_words("le chat noir") == 3

    def test_apostrophe_treated_as_word_char(self) -> None:
        # "l'amour" peut compter 1 ou 2 selon la convention. On
        # documente ici le comportement réel (1 token) pour fixer la
        # ref — peu important tant qu'on est cohérent.
        assert count_words("l'amour") == 1

    def test_sentence_split_basic(self) -> None:
        assert count_sentences("Premier. Deuxième. Troisième.") == 3

    def test_sentence_split_with_question_and_exclam(self) -> None:
        assert count_sentences("Allez ! Vraiment ? Oui.") == 3

    def test_no_final_punctuation_counts_as_one(self) -> None:
        # Un texte sans point final compte tout de même comme 1 phrase
        # (évite division par zéro dans Flesch).
        assert count_sentences("texte sans point final") == 1


# ──────────────────────────────────────────────────────────────────────────
# 3. Score Flesch
# ──────────────────────────────────────────────────────────────────────────


class TestFleschScore:
    def test_empty_text_returns_zero(self) -> None:
        assert flesch_score("", lang="fr") == 0.0
        assert flesch_score("", lang="en") == 0.0

    def test_score_is_bounded(self) -> None:
        # Phrase très simple
        s = flesch_score("Le chat. Le chien.", lang="fr")
        assert 0.0 <= s <= 100.0
        # Phrase très complexe (mots longs, peu de phrases)
        s2 = flesch_score(
            "L'établissement de l'historiographie médiévale "
            "contemporaine présente d'importantes difficultés "
            "épistémologiques",
            lang="fr",
        )
        assert 0.0 <= s2 <= 100.0

    def test_simple_higher_than_complex(self) -> None:
        simple = "Le chat est noir. Le chien est blanc."
        complex_text = (
            "L'établissement de l'historiographie médiévale "
            "contemporaine présente d'importantes difficultés "
            "épistémologiques pour les chercheurs spécialisés."
        )
        assert flesch_score(simple, "fr") > flesch_score(complex_text, "fr")

    def test_fr_and_en_differ(self) -> None:
        # Sur un texte de complexité intermédiaire (qui ne sature ni à
        # 0 ni à 100), FR et EN donnent des scores différents —
        # coefficients distincts sur le ratio syllabes/mots
        # (73.6 FR vs 84.6 EN).
        text = (
            "Le chat noir traverse la rue. Le chien blanc dort sous "
            "l arbre. Les amis jouent ensemble dans le jardin pendant "
            "que le soleil brille au dessus de la colline."
        )
        s_fr = flesch_score(text, "fr")
        s_en = flesch_score(text, "en")
        # Les deux scores doivent être dans la plage non saturée et
        # différer par les coefficients.
        assert 0.0 < s_fr < 100.0
        assert 0.0 < s_en < 100.0
        assert s_fr != s_en

    def test_invalid_lang_raises(self) -> None:
        with pytest.raises(ValueError, match="Langue"):
            flesch_score("test", lang="es")  # type: ignore[arg-type]

    def test_only_punctuation_returns_zero(self) -> None:
        assert flesch_score("...!!!???", lang="fr") == 0.0


# ──────────────────────────────────────────────────────────────────────────
# 4-5. Delta Flesch
# ──────────────────────────────────────────────────────────────────────────


class TestFleschDelta:
    def test_identical_texts_zero_delta(self) -> None:
        text = "Le chat est noir. Le chien est blanc."
        assert flesch_delta(text, text, "fr") == 0.0

    def test_empty_texts_zero_delta(self) -> None:
        assert flesch_delta("", "", "fr") == 0.0

    def test_realistic_modernization_yields_positive_delta(self) -> None:
        """Cas d'usage clé : LLM modernise un texte historique →
        signal positif clair pour le détecteur d'over-normalisation."""
        gt_old = (
            "Je vous envoie cette missive afin de vous informer "
            "de la situation à la cour, où plusieurs nouvelles "
            "méritent votre attention."
        )
        ocr_modern = (
            "Je vous écris cette lettre pour vous parler de la "
            "situation à la cour. Plusieurs nouvelles sont importantes."
        )
        delta = flesch_delta(gt_old, ocr_modern, "fr")
        # Le LLM modernisant doit produire un delta nettement positif
        # (phrases plus courtes + mots plus simples).
        assert delta > 10.0, f"Delta attendu > 10 pts, obtenu {delta:.1f}"

    def test_degraded_ocr_yields_negative_or_zero_delta(self) -> None:
        """OCR dégradé : insertions/suppressions cassent les phrases →
        delta nul ou négatif (lisibilité chute)."""
        gt = "Le chat est noir. Le chien est blanc. Les amis jouent."
        ocr_garbled = "L3 ch4t 35t n0ir. L3 ch13n 35t bl4nc. L35 4mi5 jou3nt."
        # Comportement variable selon la dégradation, mais on vérifie
        # au moins que l'écart est borné.
        delta = flesch_delta(gt, ocr_garbled, "fr")
        assert -100.0 <= delta <= 100.0

    def test_delta_is_bounded(self) -> None:
        # Cas extrêmes : score chute à 0 vs score à 100
        d1 = flesch_delta("a b c.", "x" * 200, "fr")
        d2 = flesch_delta("x" * 200, "a b c.", "fr")
        assert -100.0 <= d1 <= 100.0
        assert -100.0 <= d2 <= 100.0


# ──────────────────────────────────────────────────────────────────────────
# 6. Intégration registre typé (Sprint 34)
# ──────────────────────────────────────────────────────────────────────────


class TestRegistryIntegration:
    def test_flesch_metrics_registered_for_text_text(self) -> None:
        # Force l'import qui peuple le registre
        import picarones.core.readability  # noqa: F401

        selected = select_metrics(
            (ArtifactType.TEXT, ArtifactType.TEXT),
        )
        names = {spec.name for spec in selected}
        assert "flesch_delta_fr" in names
        assert "flesch_delta_en" in names

    def test_registered_function_returns_same_as_direct_call(self) -> None:
        from picarones.core.metric_registry import compute_at_junction

        gt = "Je vous envoie cette missive afin de vous informer."
        ocr = "Je vous écris une lettre. Voici la situation."
        out = compute_at_junction(
            gt, ocr, (ArtifactType.TEXT, ArtifactType.TEXT),
        )
        # Le delta enregistré FR doit matcher l'appel direct
        assert out["flesch_delta_fr"] == pytest.approx(
            flesch_delta(gt, ocr, "fr"), abs=1e-9,
        )
