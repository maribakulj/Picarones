"""Sprint S5 — Tests d'entrées extrêmes pour ``compute_metrics``.

Robustesse face à :

- Texte 10 Mo
- Emoji multibyte (🎉🎊)
- RTL arabe
- NFC vs NFD (formes Unicode équivalentes mais bytes différents)
- Null bytes / whitespace seul
- Line / Paragraph separator U+2028 / U+2029

Pour chacun, on vérifie qu'aucune exception ne fuit hors de
``compute_metrics`` (le décorateur try/except interne doit retourner
un MetricsResult avec ``error`` non-None ou des métriques numériques
correctes).
"""

from __future__ import annotations

import unicodedata


# --------------------------------------------------------------------------
# 1. Texte de 10 Mo
# --------------------------------------------------------------------------


class TestExtremeLengthInputs:
    def test_10mb_text_does_not_crash(self):
        from picarones.evaluation.metrics.text_metrics import compute_metrics

        # 10 Mo de texte ASCII (caractère unique répété)
        big = "a" * (10 * 1024 * 1024)
        result = compute_metrics(big, big)

        # Identité parfaite : CER = 0.0
        # Si jiwer absent, error est non-None mais pas crash.
        if result.error is None:
            assert result.cer == 0.0
            assert result.cer_nfc == 0.0
        else:
            # Échec géré sans exception remontante
            assert isinstance(result.error, str)


# --------------------------------------------------------------------------
# 2. Emoji multibyte
# --------------------------------------------------------------------------


class TestEmojiInputs:
    def test_emoji_identity_is_zero_cer(self):
        from picarones.evaluation.metrics.text_metrics import compute_metrics

        ref = "Bonjour 🎉🎊 monde"
        hyp = "Bonjour 🎉🎊 monde"
        result = compute_metrics(ref, hyp)

        if result.error is None:
            assert result.cer == 0.0

    def test_emoji_substitution_yields_positive_cer(self):
        from picarones.evaluation.metrics.text_metrics import compute_metrics

        ref = "Bonjour 🎉🎊 monde"
        hyp = "Bonjour 🎯🎯 monde"
        result = compute_metrics(ref, hyp)

        # Soit erreur gérée, soit CER > 0
        if result.error is None:
            assert result.cer is not None
            assert result.cer > 0.0


# --------------------------------------------------------------------------
# 3. RTL arabe
# --------------------------------------------------------------------------


class TestRTLArabicInputs:
    def test_arabic_identity_zero_cer(self):
        from picarones.evaluation.metrics.text_metrics import compute_metrics

        ref = "السلام عليكم"
        hyp = "السلام عليكم"
        result = compute_metrics(ref, hyp)

        if result.error is None:
            assert result.cer == 0.0
            # Tous les caractères doivent être comptés
            assert result.reference_length == len(ref)

    def test_arabic_one_char_diff_cer_positive(self):
        from picarones.evaluation.metrics.text_metrics import compute_metrics

        ref = "السلام عليكم"
        hyp = "السلام عليك"  # un caractère manquant à la fin
        result = compute_metrics(ref, hyp)

        if result.error is None:
            assert result.cer is not None
            assert result.cer > 0.0


# --------------------------------------------------------------------------
# 4. NFC vs NFD : "é" en deux formes différentes
# --------------------------------------------------------------------------


class TestUnicodeNormalizationForms:
    def test_nfc_vs_nfd_same_apparent_content(self):
        """``é`` NFC = U+00E9 ; ``é`` NFD = U+0065 + U+0301.
        Le CER brut devrait être > 0 (bytes différents),
        mais le CER NFC = 0 (les deux formes sont normalisées)."""
        from picarones.evaluation.metrics.text_metrics import compute_metrics

        ref_nfc = unicodedata.normalize("NFC", "café")  # 4 chars
        ref_nfd = unicodedata.normalize("NFD", "café")  # 5 chars
        # Sanité : les deux représentations sont effectivement distinctes
        assert ref_nfc != ref_nfd
        assert len(ref_nfc) != len(ref_nfd)

        result = compute_metrics(ref_nfc, ref_nfd)

        if result.error is None:
            # Le CER normalisé NFC doit être 0
            assert result.cer_nfc == 0.0

    def test_pure_combining_chars_handled(self):
        """Texte composé uniquement de caractères combinants
        (par ex. accents seuls)."""
        from picarones.evaluation.metrics.text_metrics import compute_metrics

        # Combining grave + combining acute
        ref = "̀́̂"
        hyp = "̀́̂"
        result = compute_metrics(ref, hyp)
        # Soit error gérée, soit identité parfaite
        if result.error is None:
            assert result.cer == 0.0


# --------------------------------------------------------------------------
# 5. Null bytes / whitespace seulement
# --------------------------------------------------------------------------


class TestNullAndWhitespaceInputs:
    def test_null_bytes_only(self):
        """Texte uniquement composé de \\x00 — pas de crash."""
        from picarones.evaluation.metrics.text_metrics import compute_metrics

        ref = "\x00\x00\x00"
        hyp = "\x00\x00\x00"
        result = compute_metrics(ref, hyp)
        # Pas d'exception, comportement défini.
        assert result is not None

    def test_whitespace_only_strings(self):
        """Texte uniquement composé d'espaces — comportement défini."""
        from picarones.evaluation.metrics.text_metrics import compute_metrics

        ref = "   "
        hyp = "   "
        result = compute_metrics(ref, hyp)
        # Pas de crash. Le ``ref.strip()`` vide → la branche "ref vide"
        # ou bien CER = 0.
        assert result is not None

    def test_empty_string_both_sides(self):
        from picarones.evaluation.metrics.text_metrics import compute_metrics

        result = compute_metrics("", "")
        # Comportement défini : pas de crash, error éventuelle
        assert result is not None


# --------------------------------------------------------------------------
# 6. U+2028 / U+2029 (Line / Paragraph separator)
# --------------------------------------------------------------------------


class TestLineParagraphSeparators:
    def test_u2028_line_separator(self):
        """U+2028 : LINE SEPARATOR. Doit être traité comme un caractère
        normal par compute_metrics (jiwer travaille sur des codepoints)."""
        from picarones.evaluation.metrics.text_metrics import compute_metrics

        ref = "ligne 1 ligne 2"
        hyp = "ligne 1 ligne 2"
        result = compute_metrics(ref, hyp)
        if result.error is None:
            assert result.cer == 0.0

    def test_u2029_paragraph_separator(self):
        from picarones.evaluation.metrics.text_metrics import compute_metrics

        ref = "para 1 para 2"
        hyp = "para 1 para 2"
        result = compute_metrics(ref, hyp)
        if result.error is None:
            assert result.cer == 0.0


# --------------------------------------------------------------------------
# 7. Mélange de scripts
# --------------------------------------------------------------------------


class TestMixedScripts:
    def test_mixed_arabic_latin_emoji(self):
        from picarones.evaluation.metrics.text_metrics import compute_metrics

        ref = "Hello مرحبا 🌍 sweet world"
        hyp = "Hello مرحبا 🌍 sweet world"
        result = compute_metrics(ref, hyp)
        if result.error is None:
            assert result.cer == 0.0
            # On a bien des bytes / caractères tous comptés
            assert result.reference_length > 0


# --------------------------------------------------------------------------
# 8. Texte avec uniquement contrôles ASCII
# --------------------------------------------------------------------------


class TestControlCharacters:
    def test_only_control_chars(self):
        """Caractères de contrôle ASCII (BEL, BS, FF…)."""
        from picarones.evaluation.metrics.text_metrics import compute_metrics

        ref = "\x07\x08\x0c"
        hyp = "\x07\x08\x0c"
        result = compute_metrics(ref, hyp)
        # Pas de crash
        assert result is not None
