"""Tests Sprint 42 — exposition des token_confidences + câblage runner.

Le runner peut maintenant calculer des métriques de calibration
(ECE / MCE / reliability) dès qu'un moteur expose des
``token_confidences`` sur l'``EngineResult``.

Couvre :

1. ``EngineResult.token_confidences`` accepte ``None`` (rétrocompat
   stricte) ou une liste de dicts.
2. ``DocumentResult.calibration_metrics`` est sérialisé via ``as_dict``
   uniquement quand renseigné, libéré par ``compact()``.
3. ``EngineReport.aggregated_calibration`` apparaît dans ``as_dict``
   quand renseigné.
4. ``_calibration_from_engine_result`` :
   - Aligne en bag-of-words avec multiplicité (proxy oracle)
   - Normalise les confidences en pourcentage (>1) à [0, 1]
   - Ignore les confidences négatives (Tesseract -1 pour non-mots)
   - Retourne ``None`` sur entrée vide / ``None``
5. ``_aggregate_calibration`` :
   - Combine les bins de plusieurs documents en somme pondérée
   - Recalcule ECE/MCE micro à partir des sommes
   - Retourne ``None`` si aucun doc n'a de calibration
6. Rétrocompat : sans token_confidences sur l'EngineResult, aucun
   calcul calibration ; ``aggregated_calibration = None``.
"""

from __future__ import annotations

import pytest

from picarones.measurements.runner import (
    _aggregate_calibration,
    _calibration_from_engine_result,
)
from picarones.evaluation.benchmark_result import DocumentResult, EngineReport
from picarones.adapters.legacy_engines.base import EngineResult


# ──────────────────────────────────────────────────────────────────────────
# 1. EngineResult.token_confidences
# ──────────────────────────────────────────────────────────────────────────


class TestEngineResultExtension:
    def test_default_is_none(self) -> None:
        r = EngineResult("e", "/tmp/x.png", "hello", 1.0)
        assert r.token_confidences is None

    def test_accepts_list_of_dicts(self) -> None:
        confs = [{"token": "hello", "confidence": 0.95}]
        r = EngineResult("e", "/tmp/x.png", "hello", 1.0, token_confidences=confs)
        assert r.token_confidences == confs


# ──────────────────────────────────────────────────────────────────────────
# 2-3. Modèles : sérialisation et compact
# ──────────────────────────────────────────────────────────────────────────


def _make_dr(calibration_metrics: dict | None = None) -> DocumentResult:
    from picarones.measurements.metrics import MetricsResult

    return DocumentResult(
        doc_id="d1", image_path="/tmp/x.png",
        ground_truth="a b c", hypothesis="a b c",
        metrics=MetricsResult(
            cer=0.0, cer_nfc=0.0, cer_caseless=0.0,
            wer=0.0, wer_normalized=0.0, mer=0.0, wil=0.0,
            reference_length=5, hypothesis_length=5,
        ),
        duration_seconds=0.1,
        calibration_metrics=calibration_metrics,
    )


class TestModelsSerialization:
    def test_calibration_metrics_omitted_when_none(self) -> None:
        d = _make_dr(None).as_dict()
        assert "calibration_metrics" not in d

    def test_calibration_metrics_present_when_set(self) -> None:
        d = _make_dr({"ece": 0.05, "mce": 0.1}).as_dict()
        assert d["calibration_metrics"] == {"ece": 0.05, "mce": 0.1}

    def test_compact_clears_calibration(self) -> None:
        # Sprint A14-S1 — ``compact()`` est désormais opt-in.
        dr = _make_dr({"ece": 0.05})
        dr.compact(drop_analyses=True)
        assert dr.calibration_metrics is None

    def test_engine_report_aggregated_calibration_omitted_when_none(self) -> None:
        rep = EngineReport(
            engine_name="t", engine_version="1", engine_config={},
            document_results=[_make_dr()],
        )
        assert "aggregated_calibration" not in rep.as_dict()

    def test_engine_report_aggregated_calibration_included_when_set(self) -> None:
        rep = EngineReport(
            engine_name="t", engine_version="1", engine_config={},
            document_results=[_make_dr()],
            aggregated_calibration={"ece": 0.05, "n_predictions": 100},
        )
        assert rep.as_dict()["aggregated_calibration"] == {
            "ece": 0.05, "n_predictions": 100,
        }


# ──────────────────────────────────────────────────────────────────────────
# 4. Helper d'alignement
# ──────────────────────────────────────────────────────────────────────────


class TestCalibrationFromEngineResult:
    def test_returns_none_for_empty_inputs(self) -> None:
        assert _calibration_from_engine_result("text", None) is None
        assert _calibration_from_engine_result("text", []) is None

    def test_perfect_calibration_when_conf_matches_accuracy(self) -> None:
        gt = "a b c d e f g h i j"
        # 7 tokens dans la GT à conf=0.7, 3 hors de la GT à conf=0.7 → ECE = 0
        tcs = (
            [{"token": c, "confidence": 0.7} for c in "abcdefg"]
            + [{"token": c, "confidence": 0.7} for c in ["X", "Y", "Z"]]
        )
        m = _calibration_from_engine_result(gt, tcs)
        assert m is not None
        assert m["ece"] == pytest.approx(0.0, abs=1e-9)
        assert m["overall_accuracy"] == pytest.approx(0.7)
        assert m["n_predictions"] == 10

    def test_normalizes_percentage_confidences(self) -> None:
        """Conf > 1 est interprétée en pourcentage et divisée par 100."""
        m = _calibration_from_engine_result(
            "hello", [{"token": "hello", "confidence": 95.0}],
        )
        assert m is not None
        # 95/100 = 0.95
        assert m["overall_confidence"] == 0.95

    def test_skips_negative_confidences(self) -> None:
        """Tesseract met -1 pour les non-mots ; on les ignore."""
        m = _calibration_from_engine_result(
            "hello", [
                {"token": "hello", "confidence": 0.9},
                {"token": ".", "confidence": -1.0},
            ],
        )
        assert m is not None
        assert m["n_predictions"] == 1

    def test_bag_of_words_with_multiplicity(self) -> None:
        # GT contient deux 'le'. L'hypothèse en a trois → 2 corrects, 1 incorrect.
        gt = "le chat le chien"
        tcs = [
            {"token": "le", "confidence": 0.9},
            {"token": "le", "confidence": 0.9},
            {"token": "le", "confidence": 0.9},  # 3e 'le' : pas dans la GT
            {"token": "chat", "confidence": 0.9},
            {"token": "chien", "confidence": 0.9},
        ]
        m = _calibration_from_engine_result(gt, tcs)
        # 4 corrects sur 5
        assert m["overall_accuracy"] == 0.8
        assert m["n_predictions"] == 5

    def test_skips_invalid_entries(self) -> None:
        m = _calibration_from_engine_result(
            "hello", [
                "not a dict",
                {"no_token": True, "confidence": 0.5},
                {"token": "hello"},  # pas de confidence
                {"token": "hello", "confidence": "abc"},  # conf non numérique
                {"token": "hello", "confidence": 0.9},  # valide
            ],
        )
        assert m is not None
        assert m["n_predictions"] == 1


# ──────────────────────────────────────────────────────────────────────────
# 5. Agrégateur
# ──────────────────────────────────────────────────────────────────────────


class TestAggregateCalibration:
    def test_returns_none_when_no_doc_has_calibration(self) -> None:
        drs = [_make_dr(None), _make_dr(None)]
        assert _aggregate_calibration(drs) is None

    def test_combines_bins_across_docs(self) -> None:
        # Doc 1 : bin [0.5, 0.6) avec 10 prédictions, conf=0.55, acc=0.5
        # Doc 2 : bin [0.5, 0.6) avec 20 prédictions, conf=0.55, acc=0.7
        # Agrégat attendu : 30 prédictions dans ce bin, conf moy = 0.55,
        # acc moy pondérée = (10*0.5 + 20*0.7) / 30 = 19/30 ≈ 0.633
        empty_bin = lambda lo, hi: {  # noqa: E731
            "bin_low": lo, "bin_high": hi,
            "avg_confidence": None, "accuracy": None,
            "count": 0, "gap": None,
        }
        bins1 = [empty_bin(k / 10, (k + 1) / 10) for k in range(10)]
        bins1[5] = {
            "bin_low": 0.5, "bin_high": 0.6,
            "avg_confidence": 0.55, "accuracy": 0.5,
            "count": 10, "gap": 0.05,
        }
        m1 = {
            "ece": 0.05, "mce": 0.05, "n_bins": 10, "n_predictions": 10,
            "overall_accuracy": 0.5, "overall_confidence": 0.55, "bins": bins1,
        }
        bins2 = [empty_bin(k / 10, (k + 1) / 10) for k in range(10)]
        bins2[5] = {
            "bin_low": 0.5, "bin_high": 0.6,
            "avg_confidence": 0.55, "accuracy": 0.7,
            "count": 20, "gap": 0.15,
        }
        m2 = {
            "ece": 0.15, "mce": 0.15, "n_bins": 10, "n_predictions": 20,
            "overall_accuracy": 0.7, "overall_confidence": 0.55, "bins": bins2,
        }
        drs = [_make_dr(m1), _make_dr(m2)]
        agg = _aggregate_calibration(drs)
        assert agg is not None
        assert agg["n_predictions"] == 30
        assert agg["doc_count"] == 2
        # Accuracy combinée = (10*0.5 + 20*0.7) / 30
        assert agg["overall_accuracy"] == (10 * 0.5 + 20 * 0.7) / 30
        # Confidence combinée = 0.55 (constante)
        assert abs(agg["overall_confidence"] - 0.55) < 1e-9
        # ECE micro : seul bin non vide (bin 5), avec count=30,
        # avg_conf=0.55, accuracy=19/30 ≈ 0.633, gap = |0.55 - 0.633|
        expected_ece = abs(0.55 - 19 / 30)
        assert abs(agg["ece"] - expected_ece) < 1e-9
        assert agg["mce"] == agg["ece"]  # un seul bin non vide → MCE = ECE


# ──────────────────────────────────────────────────────────────────────────
# 6. Rétrocompat : sans token_confidences, rien ne change
# ──────────────────────────────────────────────────────────────────────────


class TestBackwardCompat:
    def test_engine_result_default_no_calibration(self) -> None:
        # Un EngineResult sans token_confidences → calibration_metrics
        # ne doit pas être calculée.
        from picarones.measurements.runner import _compute_document_result
        ocr = EngineResult(
            engine_name="e",
            image_path="/tmp/x.png",
            text="a b c",
            duration_seconds=0.1,
            token_confidences=None,
        )
        dr = _compute_document_result(
            doc_id="d1", image_path="/tmp/x.png",
            ground_truth="a b c",
            ocr_result=ocr,
            char_exclude=None,
        )
        assert dr.calibration_metrics is None

    def test_engine_result_with_confs_triggers_calibration(self) -> None:
        from picarones.measurements.runner import _compute_document_result
        ocr = EngineResult(
            engine_name="e",
            image_path="/tmp/x.png",
            text="a b c",
            duration_seconds=0.1,
            token_confidences=[
                {"token": "a", "confidence": 0.9},
                {"token": "b", "confidence": 0.9},
                {"token": "c", "confidence": 0.9},
            ],
        )
        dr = _compute_document_result(
            doc_id="d1", image_path="/tmp/x.png",
            ground_truth="a b c",
            ocr_result=ocr,
            char_exclude=None,
        )
        assert dr.calibration_metrics is not None
        # 3 tokens, tous corrects, conf 0.9 → accuracy = 1, conf = 0.9
        assert dr.calibration_metrics["overall_accuracy"] == 1.0
        assert dr.calibration_metrics["overall_confidence"] == 0.9
