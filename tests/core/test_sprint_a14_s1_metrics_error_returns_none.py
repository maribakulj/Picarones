"""Sprint A14-S1 — A.I.0 P0 : compute_metrics retourne None en cas d'erreur.

Avant ce sprint, ``compute_metrics`` retournait des ``MetricsResult``
avec ``cer=0.0, wer=0.0, ...`` quand jiwer était indisponible ou qu'une
exception était levée.  Pour tout consommateur qui n'inspectait pas
``error``, ces zéros étaient indistinguables d'un score parfait — soit
l'inverse exact de la réalité (échec total = "100 % d'accord avec la
GT").

Désormais, en erreur, les champs métriques sont à ``None`` et ``error``
porte le message.  Un accès direct à ``result.cer`` sur un résultat en
erreur lèvera désormais ``TypeError`` lors d'opérations numériques
(``cer * 100``), ce qui est l'effet voulu : un crash explicite plutôt
qu'une valeur factice.
"""

from __future__ import annotations

from unittest import mock

import pytest

from picarones.core.metrics import MetricsResult, aggregate_metrics
from picarones.measurements import metrics as metrics_module
from picarones.measurements.metrics import compute_metrics


class TestComputeMetricsErrorPath:
    def test_jiwer_missing_returns_none_metrics(self) -> None:
        """Si jiwer absent, tous les champs sont None et error est set."""
        with mock.patch.object(metrics_module, "_JIWER_AVAILABLE", False):
            result = compute_metrics("référence", "hypothèse")
        assert result.cer is None
        assert result.cer_nfc is None
        assert result.cer_caseless is None
        assert result.wer is None
        assert result.wer_normalized is None
        assert result.mer is None
        assert result.wil is None
        assert result.error is not None
        assert "jiwer" in result.error.lower()

    def test_jiwer_exception_returns_none_metrics(self) -> None:
        """Si jiwer lève, on retombe dans le bloc except et on retourne None."""
        with mock.patch.object(
            metrics_module, "_cer_from_strings",
            side_effect=RuntimeError("simulated jiwer crash"),
        ):
            result = compute_metrics("a", "b")
        assert result.cer is None
        assert result.wer is None
        assert result.error is not None
        assert "simulated jiwer crash" in result.error

    def test_no_silent_zero_when_error_set(self) -> None:
        """Garde-fou : aucun champ ne doit être 0.0 si error est non-None.

        Verrouille le bug exact que ce sprint corrige (0.0 indistinguable
        d'un score parfait dans le JSON exporté).
        """
        with mock.patch.object(metrics_module, "_JIWER_AVAILABLE", False):
            result = compute_metrics("référence", "hypothèse")
        assert result.error is not None
        for field in ("cer", "cer_nfc", "cer_caseless", "wer",
                      "wer_normalized", "mer", "wil"):
            assert getattr(result, field) is None, (
                f"{field} = {getattr(result, field)!r} (devrait être None "
                "puisque error est non-None)"
            )


class TestMetricsResultPropertiesHandleNone:
    def test_cer_percent_handles_none(self) -> None:
        r = MetricsResult(error="boom")
        assert r.cer_percent is None

    def test_wer_percent_handles_none(self) -> None:
        r = MetricsResult(error="boom")
        assert r.wer_percent is None

    def test_as_dict_handles_none(self) -> None:
        r = MetricsResult(error="boom")
        d = r.as_dict()
        assert d["cer"] is None
        assert d["wer"] is None
        assert d["error"] == "boom"

    def test_as_dict_rounds_when_set(self) -> None:
        r = MetricsResult(cer=0.123456789, wer=0.456789, error=None)
        d = r.as_dict()
        assert d["cer"] == 0.123457  # 6 décimales
        assert d["wer"] == 0.456789


class TestAggregateMetricsFiltersNoneAndError:
    def test_aggregator_excludes_results_with_error(self) -> None:
        ok = MetricsResult(cer=0.1, wer=0.2, mer=0.15, wil=0.25, error=None)
        ko = MetricsResult(error="boom")  # cer/wer/etc tous None
        agg = aggregate_metrics([ok, ko])
        # Seul le résultat OK contribue à la moyenne.
        assert agg["cer"]["mean"] == 0.1
        assert agg["wer"]["mean"] == 0.2
        assert agg["failed_count"] == 1
        assert agg["document_count"] == 2

    def test_aggregator_robust_to_partial_none(self) -> None:
        """Défense en profondeur : un caller pourrait construire un
        MetricsResult avec des None sans avoir set ``error``.  On ne
        plante pas, on saute simplement les None."""
        partial = MetricsResult(cer=0.05, wer=None, mer=None, wil=None, error=None)
        agg = aggregate_metrics([partial])
        assert agg["cer"]["mean"] == 0.05
        # WER absent → stats vides plutôt que NaN.
        assert agg["wer"] == {}

    def test_aggregator_empty_when_all_errors(self) -> None:
        errs = [MetricsResult(error="x"), MetricsResult(error="y")]
        agg = aggregate_metrics(errs)
        assert agg["cer"] == {}
        assert agg["failed_count"] == 2
        assert agg["document_count"] == 2
