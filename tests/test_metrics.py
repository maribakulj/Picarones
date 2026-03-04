"""Tests unitaires pour le module picarones.core.metrics."""

import pytest

from picarones.core.metrics import aggregate_metrics, compute_metrics, MetricsResult


class TestComputeMetrics:
    """Tests de compute_metrics sur des cas connus."""

    def test_perfect_match(self):
        """CER et WER doivent être 0 quand référence == hypothèse."""
        result = compute_metrics("Bonjour le monde", "Bonjour le monde")
        assert result.cer == pytest.approx(0.0)
        assert result.wer == pytest.approx(0.0)
        assert result.error is None

    def test_complete_mismatch(self):
        """CER proche de 1 quand les textes sont totalement différents."""
        result = compute_metrics("abc", "xyz")
        assert result.cer > 0.0
        assert result.error is None

    def test_empty_reference(self):
        """Référence vide : CER = 1.0 si hypothèse non vide."""
        result = compute_metrics("", "quelque chose")
        assert result.cer == pytest.approx(1.0)

    def test_empty_both(self):
        """Référence et hypothèse vides : CER = 0.0."""
        result = compute_metrics("", "")
        assert result.cer == pytest.approx(0.0)

    def test_single_substitution(self):
        """Une seule substitution sur 4 chars → CER = 0.25."""
        result = compute_metrics("abcd", "abce")
        assert result.cer == pytest.approx(0.25)

    def test_case_insensitive_cer(self):
        """CER caseless ignore les différences de casse."""
        result = compute_metrics("Bonjour", "bonjour")
        assert result.cer_caseless == pytest.approx(0.0)
        # CER brut doit être > 0 (B ≠ b)
        assert result.cer > 0.0

    def test_nfc_normalization(self):
        """CER NFC normalise les séquences unicode équivalentes."""
        # é peut être encodé en forme composée (U+00E9) ou décomposée (e + U+0301)
        composed = "\u00e9"       # é (NFC)
        decomposed = "e\u0301"    # e + combining accent (NFD)
        result = compute_metrics(composed, decomposed)
        # Après NFC, les deux sont identiques → cer_nfc = 0
        assert result.cer_nfc == pytest.approx(0.0)

    def test_wer_one_word_wrong(self):
        """WER = 1/3 pour 1 mot faux sur 3."""
        result = compute_metrics("le chat dort", "le chien dort")
        assert result.wer == pytest.approx(1 / 3, rel=1e-2)

    def test_result_has_lengths(self):
        ref = "Texte de référence"
        result = compute_metrics(ref, "Texte différent")
        assert result.reference_length == len(ref)
        assert result.hypothesis_length > 0

    def test_metrics_result_as_dict(self):
        """as_dict() doit retourner toutes les clés attendues."""
        result = compute_metrics("abc", "abc")
        d = result.as_dict()
        for key in ["cer", "cer_nfc", "cer_caseless", "wer", "wer_normalized", "mer", "wil"]:
            assert key in d

    def test_cer_percent_property(self):
        result = compute_metrics("abcd", "abce")
        assert result.cer_percent == pytest.approx(25.0, rel=1e-2)


class TestAggregateMetrics:
    """Tests de aggregate_metrics."""

    def _make_result(self, cer: float) -> MetricsResult:
        return MetricsResult(
            cer=cer, cer_nfc=cer, cer_caseless=cer,
            wer=cer, wer_normalized=cer, mer=cer, wil=cer,
            reference_length=100,
            hypothesis_length=100,
        )

    def test_empty_list(self):
        assert aggregate_metrics([]) == {}

    def test_single_result(self):
        results = [self._make_result(0.1)]
        agg = aggregate_metrics(results)
        assert agg["cer"]["mean"] == pytest.approx(0.1)
        assert agg["cer"]["min"] == pytest.approx(0.1)
        assert agg["cer"]["max"] == pytest.approx(0.1)

    def test_multiple_results(self):
        results = [self._make_result(0.1), self._make_result(0.3)]
        agg = aggregate_metrics(results)
        assert agg["cer"]["mean"] == pytest.approx(0.2)
        assert agg["document_count"] == 2
        assert agg["failed_count"] == 0

    def test_failed_results_excluded(self):
        ok = self._make_result(0.1)
        failed = MetricsResult(
            cer=1.0, cer_nfc=1.0, cer_caseless=1.0,
            wer=1.0, wer_normalized=1.0, mer=1.0, wil=1.0,
            reference_length=50, hypothesis_length=0,
            error="Moteur en erreur",
        )
        agg = aggregate_metrics([ok, failed])
        # Les métriques agrégées n'incluent que les résultats sans erreur
        assert agg["cer"]["mean"] == pytest.approx(0.1)
        assert agg["failed_count"] == 1
