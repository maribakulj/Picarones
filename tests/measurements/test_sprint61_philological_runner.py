"""Tests Sprint 61 — câblage backend des métriques philologiques.

Couvre :

1. Champs ``DocumentResult.philological_metrics`` et
   ``EngineReport.aggregated_philological`` posés.
2. Sérialisation conditionnelle dans ``as_dict``.
3. Libération par ``compact``.
4. ``compute_philological_metrics`` :
   - GT médiéval déclenche abbreviations + mufi
   - GT imprimé ancien déclenche early_modern
   - GT moderne déclenche modern_archives
   - GT avec numéraux romains déclenche roman_numerals
   - GT avec caractères hors Basic Latin déclenche unicode_blocks
   - GT en ASCII pur sans marqueur → ``None``
   - GT vide / None → ``None``
5. ``aggregate_philological_metrics`` :
   - Somme correcte des compteurs par module
   - Recalcul correct des scores globaux
   - Doc count cohérent
   - Aucun document avec signal → ``None``
6. Intégration runner end-to-end via fixture mock.
"""

from __future__ import annotations

from picarones.measurements.philological_hooks import (
    aggregate_philological_metrics,
    compute_philological_metrics,
)
from picarones.core.results import DocumentResult, EngineReport
from picarones.measurements.metrics import MetricsResult


def _make_doc(
    doc_id: str = "d1",
    gt: str = "",
    hyp: str = "",
    philological: dict | None = None,
) -> DocumentResult:
    """Helper : construit un DocumentResult minimal pour les tests."""
    return DocumentResult(
        doc_id=doc_id,
        image_path=f"/tmp/{doc_id}.png",
        ground_truth=gt,
        hypothesis=hyp,
        metrics=MetricsResult(
            cer=0.0, cer_nfc=0.0, cer_caseless=0.0,
            wer=0.0, wer_normalized=0.0, mer=0.0, wil=0.0,
            reference_length=len(gt), hypothesis_length=len(hyp),
        ),
        duration_seconds=0.1,
        philological_metrics=philological,
    )


# ──────────────────────────────────────────────────────────────────────────
# 1. Champs posés sur DocumentResult / EngineReport
# ──────────────────────────────────────────────────────────────────────────


class TestFields:
    def test_document_result_default_none(self) -> None:
        dr = _make_doc()
        assert dr.philological_metrics is None

    def test_document_result_accepts_dict(self) -> None:
        dr = _make_doc(philological={"mufi": {"coverage": 0.9}})
        assert dr.philological_metrics == {"mufi": {"coverage": 0.9}}

    def test_engine_report_default_none(self) -> None:
        report = EngineReport(
            engine_name="test", engine_version="1.0",
            engine_config={}, document_results=[],
        )
        assert report.aggregated_philological is None

    def test_engine_report_accepts_dict(self) -> None:
        report = EngineReport(
            engine_name="test", engine_version="1.0",
            engine_config={}, document_results=[],
            aggregated_philological={"mufi": {"coverage": 0.9}},
        )
        assert report.aggregated_philological == {"mufi": {"coverage": 0.9}}


# ──────────────────────────────────────────────────────────────────────────
# 2. Sérialisation as_dict
# ──────────────────────────────────────────────────────────────────────────


class TestSerialization:
    def test_as_dict_omits_none(self) -> None:
        dr = _make_doc()
        d = dr.as_dict()
        assert "philological_metrics" not in d

    def test_as_dict_includes_when_present(self) -> None:
        dr = _make_doc(philological={"mufi": {"coverage": 1.0}})
        d = dr.as_dict()
        assert d["philological_metrics"] == {"mufi": {"coverage": 1.0}}

    def test_engine_report_as_dict_omits_none(self) -> None:
        report = EngineReport(
            engine_name="t", engine_version="1", engine_config={},
            document_results=[],
        )
        assert "aggregated_philological" not in report.as_dict()

    def test_engine_report_as_dict_includes_when_present(self) -> None:
        report = EngineReport(
            engine_name="t", engine_version="1", engine_config={},
            document_results=[],
            aggregated_philological={"mufi": {"coverage": 0.5}},
        )
        d = report.as_dict()
        assert d["aggregated_philological"] == {"mufi": {"coverage": 0.5}}


# ──────────────────────────────────────────────────────────────────────────
# 3. Libération par compact()
# ──────────────────────────────────────────────────────────────────────────


class TestCompact:
    def test_compact_clears_philological(self) -> None:
        # Sprint A14-S1 — opt-in via drop_analyses=True.
        dr = _make_doc(philological={"mufi": {"coverage": 1.0}})
        dr.compact(drop_analyses=True)
        assert dr.philological_metrics is None


# ──────────────────────────────────────────────────────────────────────────
# 4. compute_philological_metrics — adaptive masking
# ──────────────────────────────────────────────────────────────────────────


class TestComputeAdaptive:
    def test_medieval_triggers_abbreviations_and_mufi(self) -> None:
        gt = "fait en lan ꝑ regem þæt"
        m = compute_philological_metrics(gt, gt)
        assert m is not None
        assert "abbreviations" in m
        assert "mufi" in m

    def test_early_modern_triggers_typography(self) -> None:
        gt = "le ſerpent ﬁnement & ã"
        m = compute_philological_metrics(gt, gt)
        assert m is not None
        assert "early_modern" in m

    def test_modern_archives_triggers_module(self) -> None:
        gt = "Mme Dupont au bd Voltaire vol. II"
        m = compute_philological_metrics(gt, gt)
        assert m is not None
        assert "modern_archives" in m

    def test_roman_numerals_triggers_module(self) -> None:
        gt = "Louis XIV mourut en MDCCXV"
        m = compute_philological_metrics(gt, gt)
        assert m is not None
        assert "roman_numerals" in m

    def test_unicode_blocks_triggered_only_outside_basic_latin(self) -> None:
        # ASCII pur sans marqueur → unicode_blocks omis (Basic Latin
        # uniquement, breakdown trivial).
        m = compute_philological_metrics("hello world", "hello world")
        assert m is None

    def test_unicode_blocks_triggered_with_diacritics(self) -> None:
        # Du Latin Extended → unicode_blocks inclus
        gt = "café à é ô"
        m = compute_philological_metrics(gt, gt)
        assert m is not None
        assert "unicode_blocks" in m

    def test_empty_returns_none(self) -> None:
        assert compute_philological_metrics("", "") is None
        assert compute_philological_metrics(None, None) is None

    def test_no_signal_returns_none(self) -> None:
        # Pure Basic Latin sans aucun marqueur philologique
        m = compute_philological_metrics("hello", "hello")
        assert m is None


# ──────────────────────────────────────────────────────────────────────────
# 5. aggregate_philological_metrics
# ──────────────────────────────────────────────────────────────────────────


class TestAggregation:
    def test_no_data_returns_none(self) -> None:
        assert aggregate_philological_metrics([]) is None
        assert aggregate_philological_metrics([None, None]) is None

    def test_aggregates_only_present_modules(self) -> None:
        # Doc 1 a mufi+abbr, Doc 2 a juste roman_numerals
        d1 = compute_philological_metrics("ꝑ ꝓ ꝗ", "per pro qui")
        d2 = compute_philological_metrics("Louis XIV", "Louis 14")
        agg = aggregate_philological_metrics([d1, d2])
        assert agg is not None
        # mufi présent (Doc1 le déclenchait avec ꝑ/ꝓ/ꝗ qui sont MUFI)
        assert "abbreviations" in agg
        assert "roman_numerals" in agg
        # doc_count par module
        assert agg["abbreviations"]["doc_count"] == 1
        assert agg["roman_numerals"]["doc_count"] == 1

    def test_aggregation_sums_counters(self) -> None:
        # 3 docs avec MUFI : "þæt ꝑ" = 3 caractères MUFI (þ, æ, ꝑ)
        gt = "þæt ꝑ"
        per_doc = [compute_philological_metrics(gt, gt) for _ in range(3)]
        agg = aggregate_philological_metrics(per_doc)
        assert agg is not None
        assert "mufi" in agg
        # 3 caractères × 3 docs = 9
        assert agg["mufi"]["n_mufi_chars_reference"] == 9
        assert agg["mufi"]["n_mufi_chars_preserved"] == 9
        assert agg["mufi"]["coverage"] == 1.0
        assert agg["mufi"]["doc_count"] == 3

    def test_aggregation_recomputes_global_score(self) -> None:
        # Doc1 préserve 100%, Doc2 préserve 0% → moyenne pondérée
        d1 = compute_philological_metrics("XIV", "XIV")
        d2 = compute_philological_metrics("V", "perdu")
        agg = aggregate_philological_metrics([d1, d2])
        roman = agg["roman_numerals"]
        # Doc1 : 1 strict_preserved (XIV)
        # Doc2 : 1 lost (V)
        # Total : 2 numéraux, 1 strict → 0.5
        assert roman["n_numerals_reference"] == 2
        assert roman["global_strict_score"] == 0.5

    def test_per_category_aggregation_modern_archives(self) -> None:
        # Deux docs avec modern_archives sur catégories différentes
        d1 = compute_philological_metrics("Mme bd", "Mme bd")
        d2 = compute_philological_metrics("vol. p.", "vol. p.")
        agg = aggregate_philological_metrics([d1, d2])
        per_cat = agg["modern_archives"]["per_category"]
        # Doc1 : civility_titles + address ; Doc2 : bibliographic
        assert "civility_titles" in per_cat
        assert "address" in per_cat
        assert "bibliographic" in per_cat
        for cat in per_cat.values():
            assert cat["strict_score"] == 1.0


# ──────────────────────────────────────────────────────────────────────────
# 6. Intégration end-to-end (mock léger sur le runner)
# ──────────────────────────────────────────────────────────────────────────


class TestRunnerIntegration:
    """Vérifie que ``_compute_document_result`` attache bien les
    ``philological_metrics`` quand la GT a du signal."""

    def test_runner_attaches_philological(self, tmp_path) -> None:
        from picarones.measurements.runner import _compute_document_result
        from picarones.evaluation.engines.base import EngineResult

        # Créer une image fictive (le module image_quality échouera
        # gracieusement, ce qui est OK pour le test).
        img = tmp_path / "doc.png"
        img.write_bytes(b"")  # vide ; on ignore le résultat image_quality

        gt = "ꝑ regem mcclxxxij"
        ocr_result = EngineResult(
            engine_name="mock", image_path=str(img),
            text=gt, duration_seconds=0.1, error=None,
        )
        dr = _compute_document_result(
            doc_id="d1",
            image_path=str(img),
            ground_truth=gt,
            ocr_result=ocr_result,
            char_exclude=None,
        )
        assert dr.philological_metrics is not None
        assert "abbreviations" in dr.philological_metrics
        assert "roman_numerals" in dr.philological_metrics

    def test_runner_omits_philological_on_plain_text(self, tmp_path) -> None:
        from picarones.measurements.runner import _compute_document_result
        from picarones.evaluation.engines.base import EngineResult

        img = tmp_path / "doc.png"
        img.write_bytes(b"")

        # Texte ASCII pur sans marqueur philologique
        gt = "hello world without any markers"
        ocr_result = EngineResult(
            engine_name="mock", image_path=str(img),
            text=gt, duration_seconds=0.1, error=None,
        )
        dr = _compute_document_result(
            doc_id="d1",
            image_path=str(img),
            ground_truth=gt,
            ocr_result=ocr_result,
            char_exclude=None,
        )
        assert dr.philological_metrics is None
