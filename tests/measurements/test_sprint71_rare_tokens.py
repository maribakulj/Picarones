"""Tests Sprint 71 — A.I.1 chantier 2 : rare-token recall.

Couvre :

1. ``tokenize`` : Unicode-aware, contractions (l'an, d'une),
   composés (peut-être, c'est-à-dire), apostrophe typographique
   ’, vide / None.
2. ``frequency_distribution`` : comptage corpus-wide, casse
   insensible par défaut, multi-doc.
3. ``extract_rare_tokens`` : hapax (max_freq=1), dis legomena
   (max_freq=2), ``max_freq < 1`` → ValueError.
4. ``compute_rare_token_recall`` :
   - cas standard : 5 rares en GT, 4 préservés
   - multiplicité : un rare présent 2× en GT, 1× en hyp → 0.5
   - hyp vide → 0.0, tous manqués
   - GT sans rare → 0.0, listes vides
   - case_sensitive
5. ``rare_token_recall`` raccourci.
6. **Cas réaliste** : registre d'état civil, noms propres rares
   discriminés.
"""

from __future__ import annotations

import pytest

from picarones.measurements.rare_tokens import (
    compute_rare_token_recall,
    extract_rare_tokens,
    frequency_distribution,
    rare_token_recall,
    tokenize,
)


# ──────────────────────────────────────────────────────────────────────────
# 1. tokenize
# ──────────────────────────────────────────────────────────────────────────


class TestTokenize:
    def test_basic_words(self) -> None:
        assert tokenize("hello world") == ["hello", "world"]

    def test_contraction_apostrophe_ascii(self) -> None:
        # L'an est un seul token
        assert tokenize("L'an") == ["L'an"]
        assert tokenize("d'une chose") == ["d'une", "chose"]

    def test_contraction_apostrophe_typographic(self) -> None:
        # ’ (U+2019) traité comme ' à l'intérieur du token
        assert tokenize("d’une") == ["d’une"]

    def test_compound_with_hyphen(self) -> None:
        assert tokenize("peut-être") == ["peut-être"]
        assert tokenize("c'est-à-dire") == ["c'est-à-dire"]

    def test_unicode_diacritics(self) -> None:
        assert tokenize("café à é ô") == ["café", "à", "é", "ô"]

    def test_punctuation_separates(self) -> None:
        assert tokenize("Marie, fille.") == ["Marie", "fille"]

    def test_numbers_are_tokens(self) -> None:
        assert tokenize("en 1789 et 1790") == ["en", "1789", "et", "1790"]

    def test_empty_input(self) -> None:
        assert tokenize("") == []
        assert tokenize(None) == []


# ──────────────────────────────────────────────────────────────────────────
# 2. frequency_distribution
# ──────────────────────────────────────────────────────────────────────────


class TestFrequencyDistribution:
    def test_single_document(self) -> None:
        freq = frequency_distribution(["hello hello world"])
        assert freq["hello"] == 2
        assert freq["world"] == 1

    def test_multi_document_summed(self) -> None:
        docs = ["hello world", "hello sun", "moon"]
        freq = frequency_distribution(docs)
        assert freq["hello"] == 2
        assert freq["world"] == 1
        assert freq["moon"] == 1

    def test_case_insensitive_default(self) -> None:
        freq = frequency_distribution(["Hello hello HELLO"])
        assert freq["hello"] == 3
        assert "Hello" not in freq

    def test_case_sensitive(self) -> None:
        freq = frequency_distribution(
            ["Hello hello"], case_sensitive=True,
        )
        assert freq["Hello"] == 1
        assert freq["hello"] == 1


# ──────────────────────────────────────────────────────────────────────────
# 3. extract_rare_tokens
# ──────────────────────────────────────────────────────────────────────────


class TestExtractRareTokens:
    def test_hapax_only(self) -> None:
        # max_freq=1 → uniquement les tokens uniques
        docs = ["a a b c"]
        rare = extract_rare_tokens(docs, max_freq=1)
        assert rare == frozenset({"b", "c"})

    def test_hapax_plus_dis_legomena_default(self) -> None:
        # max_freq=2 par défaut
        docs = ["a a a b b c"]
        rare = extract_rare_tokens(docs)
        # a (3) écarté, b (2) inclus, c (1) inclus
        assert rare == frozenset({"b", "c"})

    def test_invalid_max_freq(self) -> None:
        with pytest.raises(ValueError):
            extract_rare_tokens(["x"], max_freq=0)
        with pytest.raises(ValueError):
            extract_rare_tokens(["x"], max_freq=-1)

    def test_empty_corpus(self) -> None:
        assert extract_rare_tokens([]) == frozenset()


# ──────────────────────────────────────────────────────────────────────────
# 4. compute_rare_token_recall
# ──────────────────────────────────────────────────────────────────────────


class TestComputeRareTokenRecall:
    def test_full_recall(self) -> None:
        rare = {"alice", "bob"}
        m = compute_rare_token_recall(
            "alice et bob mangent", "alice et bob mangent", rare,
        )
        assert m["recall"] == 1.0
        assert m["n_rare_tokens_in_reference"] == 2
        assert m["n_rare_tokens_recalled"] == 2
        assert m["missed_tokens"] == []

    def test_partial_recall(self) -> None:
        rare = {"alice", "bob", "charlie"}
        m = compute_rare_token_recall(
            "alice bob charlie", "alice bob", rare,
        )
        assert m["n_rare_tokens_in_reference"] == 3
        assert m["n_rare_tokens_recalled"] == 2
        assert m["recall"] == pytest.approx(2 / 3)
        assert m["missed_tokens"] == ["charlie"]

    def test_zero_recall(self) -> None:
        rare = {"alice", "bob"}
        m = compute_rare_token_recall(
            "alice bob", "x y z", rare,
        )
        assert m["recall"] == 0.0
        assert sorted(m["missed_tokens"]) == ["alice", "bob"]

    def test_multiplicity(self) -> None:
        # Un token rare présent 2 fois en GT, 1 fois en hyp → 0.5
        rare = {"dupont"}
        m = compute_rare_token_recall(
            "Dupont et Dupont sont là", "Dupont arrive", rare,
        )
        assert m["n_rare_tokens_in_reference"] == 2
        assert m["n_rare_tokens_recalled"] == 1
        assert m["recall"] == 0.5
        assert m["missed_tokens"] == ["dupont"]

    def test_no_rare_in_gt(self) -> None:
        rare = {"alice"}
        m = compute_rare_token_recall("hello world", "hello world", rare)
        assert m["n_rare_tokens_in_reference"] == 0
        assert m["recall"] == 0.0
        assert m["missed_tokens"] == []

    def test_empty_hyp(self) -> None:
        rare = {"alice", "bob"}
        m = compute_rare_token_recall("alice bob", "", rare)
        assert m["recall"] == 0.0
        assert sorted(m["missed_tokens"]) == ["alice", "bob"]

    def test_none_inputs(self) -> None:
        rare = {"alice"}
        m = compute_rare_token_recall(None, None, rare)
        assert m["recall"] == 0.0
        assert m["n_rare_tokens_in_reference"] == 0

    def test_case_insensitive_default(self) -> None:
        rare = {"Alice"}  # passé en casse mixte
        m = compute_rare_token_recall("alice arrive", "alice", rare)
        # Casse-insensible par défaut : "Alice" → "alice", match
        assert m["recall"] == 1.0

    def test_case_sensitive(self) -> None:
        rare = {"Alice"}
        m = compute_rare_token_recall(
            "Alice arrive", "alice arrive", rare,
            case_sensitive=True,
        )
        # GT contient "Alice", hyp contient "alice" → pas de match
        # parce qu'on est sensible à la casse
        assert m["n_rare_tokens_in_reference"] == 1
        assert m["recall"] == 0.0


# ──────────────────────────────────────────────────────────────────────────
# 5. Raccourci
# ──────────────────────────────────────────────────────────────────────────


class TestShortcut:
    def test_shortcut_matches_full(self) -> None:
        rare = {"alice", "bob"}
        full = compute_rare_token_recall("alice bob", "alice", rare)
        assert rare_token_recall(
            "alice bob", "alice", rare,
        ) == pytest.approx(full["recall"])


# ──────────────────────────────────────────────────────────────────────────
# 6. Cas réaliste : registre d'état civil
# ──────────────────────────────────────────────────────────────────────────


class TestRealisticCivilRecord:
    def test_proper_nouns_discrimination(self) -> None:
        # 3 actes d'état civil avec noms propres uniques
        corpus = [
            "Marie Dupont, fille de Jean Dupont, baptisée 1789.",
            "Pierre Durand, fils de Catherine Bernard, né 1790.",
            "Jacques Martin, époux de Anne Lefèvre, décédé 1801.",
        ]
        rare = extract_rare_tokens(corpus, max_freq=2)
        # Tous les noms propres sont hapax (1 occurrence) sauf
        # « Dupont » (2 occurrences = dis legomenon). Tous restent
        # « rares » avec max_freq=2.
        assert "dupont" in rare
        assert "lefèvre" in rare
        assert "martin" in rare

        # OCR fautif qui rate les noms propres mais préserve les
        # mots fréquents
        gt = corpus[0]
        hyp_bad_proper = "Marie X, fille de Jean X, baptisée 1789."
        m = compute_rare_token_recall(gt, hyp_bad_proper, rare)
        # « Dupont » présent 2 fois en GT, 0 fois en hyp → 0/2
        # « Marie » et autres mots non rares → ignorés
        # « 1789 » est rare, présent 1 fois en GT, 1 fois en hyp → 1/1
        # « baptisée » est rare aussi
        assert m["n_rare_tokens_recalled"] < m["n_rare_tokens_in_reference"]
        # Au moins « dupont » manqué
        assert "dupont" in m["missed_tokens"]

    def test_proper_ocr_discriminates_more_than_cer(self) -> None:
        """Vérifie la conjecture du plan : un OCR qui préserve la
        structure mais rate les noms propres a un CER faible mais
        un rare-token recall plus dégradé.

        On compare deux OCR sur le même GT :
        - OCR_A : rate un nom propre rare (« Dupont »)
        - OCR_B : rate un mot fréquent (« le » présent ≥ 3× dans
          le corpus, donc PAS dans le set des rares)
        """
        # Corpus suffisamment grand pour que « le » soit fréquent
        # (≥ 3 occurrences) et donc non-rare.
        corpus = [
            "Marie Dupont arriva le matin chez le notaire.",
            "Pierre Durand le suivit dans le couloir.",
            "Catherine Bernard attendait le retour le soir.",
            "Jacques Martin écouta le récit de la journée.",
        ]
        rare = extract_rare_tokens(corpus, max_freq=2)
        # Sanité : « le » n'est PAS rare (apparaît 7 fois)
        assert "le" not in rare
        # « Dupont » est rare (1 occurrence)
        assert "dupont" in rare

        gt = corpus[0]
        hyp_a_proper_lost = "Marie X arriva le matin chez le notaire."
        hyp_b_freq_lost = "Marie Dupont arriva matin chez notaire."  # 2 « le » manquent
        m_a = compute_rare_token_recall(gt, hyp_a_proper_lost, rare)
        m_b = compute_rare_token_recall(gt, hyp_b_freq_lost, rare)
        # OCR_A perd un rare (« Dupont »), OCR_B n'en perd aucun
        # (« le » n'est pas rare donc sa perte n'affecte pas le recall)
        assert m_a["recall"] < m_b["recall"]
        assert m_b["recall"] == 1.0
