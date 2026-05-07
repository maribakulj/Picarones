"""Tests Sprint 16 — câblage line_metrics/hallucination + fondations du moteur narratif.

Couverture :
1. ``compute_document_result`` via le runner peuple bien ``line_metrics`` et
   ``hallucination_metrics`` sur un document réussi.
2. ``EngineReport`` expose ``aggregated_line_metrics`` et
   ``aggregated_hallucination`` après un benchmark.
3. Le modèle ``Fact`` et le ``DetectorRegistry`` fonctionnent.
4. Le registre par défaut est vide en Sprint 1 (les détecteurs seront activés
   progressivement dans les sprints suivants).
"""

from __future__ import annotations

import json
from pathlib import Path

from picarones.evaluation.corpus import Corpus, Document
from picarones.measurements.narrative import (
    DetectorRegistry,
    Fact,
    FactImportance,
    FactType,
    detect_all,
)
from picarones.measurements.runner import (
    _aggregate_hallucination,
    _aggregate_line_metrics,
    _compute_document_result,
    run_benchmark,
)
from picarones.adapters.legacy_engines.base import BaseOCREngine, EngineResult


class _FakeEngine(BaseOCREngine):
    """Moteur factice — renvoie un texte configurable, utile en test."""

    def __init__(self, output_text: str, name: str = "fake", config=None):
        super().__init__(config)
        self._output = output_text
        self._display_name = name

    @property
    def name(self) -> str:
        return self._display_name

    def version(self) -> str:
        return "test"

    def _run_ocr(self, image_path):
        return self._output, None

    def run(self, image_path) -> EngineResult:
        return EngineResult(
            engine_name=self.name,
            image_path=str(image_path),
            text=self._output,
            duration_seconds=0.01,
        )


# ---------------------------------------------------------------------------
# 1. Câblage line_metrics et hallucination par document
# ---------------------------------------------------------------------------

class TestDocumentResultWiring:
    """Vérifie que ``_compute_document_result`` peuple les nouveaux champs."""

    def test_line_metrics_populated_on_success(self, tmp_path: Path):
        image = tmp_path / "doc.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n")  # stub — image_quality loggera un warning

        ocr = EngineResult(
            engine_name="fake",
            image_path=str(image),
            text="ligne une\nligne deux\nligne trois",
            duration_seconds=0.1,
        )
        gt = "ligne une\nligne deux\nligne trois"

        result = _compute_document_result(
            doc_id="doc1",
            image_path=str(image),
            ground_truth=gt,
            ocr_result=ocr,
            char_exclude=None,
        )

        assert result.line_metrics is not None, "line_metrics doit être peuplé"
        assert "percentiles" in result.line_metrics
        assert "gini" in result.line_metrics
        assert result.line_metrics["line_count"] == 3

    def test_hallucination_metrics_populated_on_success(self, tmp_path: Path):
        image = tmp_path / "doc.png"
        image.write_bytes(b"")

        gt = "le chat est sur le tapis rouge et dort paisiblement"
        hyp = "le chat mange des bananes spatiales en orbite lunaire"

        ocr = EngineResult(
            engine_name="fake",
            image_path=str(image),
            text=hyp,
            duration_seconds=0.1,
        )

        result = _compute_document_result(
            doc_id="doc1",
            image_path=str(image),
            ground_truth=gt,
            ocr_result=ocr,
            char_exclude=None,
        )

        assert result.hallucination_metrics is not None
        assert "anchor_score" in result.hallucination_metrics
        assert "length_ratio" in result.hallucination_metrics
        assert "is_hallucinating" in result.hallucination_metrics

    def test_new_fields_empty_on_engine_failure(self, tmp_path: Path):
        """Si l'OCR échoue (success=False), pas de calcul line_metrics/hallucination."""
        image = tmp_path / "doc.png"
        image.write_bytes(b"")

        ocr = EngineResult(
            engine_name="fake",
            image_path=str(image),
            text="",
            duration_seconds=0.1,
            error="simulated failure",
        )
        result = _compute_document_result(
            doc_id="doc1",
            image_path=str(image),
            ground_truth="ground truth text",
            ocr_result=ocr,
            char_exclude=None,
        )

        assert result.line_metrics is None
        assert result.hallucination_metrics is None


# ---------------------------------------------------------------------------
# 2. Agrégation au niveau EngineReport
# ---------------------------------------------------------------------------

class TestAggregationWiring:
    """Vérifie que le benchmark complet produit les agrégations."""

    def test_aggregate_line_metrics_helper_with_empty_list(self):
        assert _aggregate_line_metrics([]) is None

    def test_aggregate_hallucination_helper_with_empty_list(self):
        assert _aggregate_hallucination([]) is None

    def test_benchmark_end_to_end_produces_aggregations(self, tmp_path: Path):
        img = tmp_path / "test.png"
        img.write_bytes(b"")

        corpus = Corpus(
            name="test",
            documents=[
                Document(
                    doc_id="d1",
                    image_path=img,
                    ground_truth="bonjour le monde\nligne deux\nfin",
                ),
                Document(
                    doc_id="d2",
                    image_path=img,
                    ground_truth="autre document test\navec deux lignes",
                ),
            ],
            source_path=str(tmp_path),
        )

        engine = _FakeEngine(
            output_text="bonjour le monde\nligne deux\nfin",
            name="fake_engine",
        )

        result = run_benchmark(
            corpus=corpus,
            engines=[engine],
            show_progress=False,
            max_workers=1,
            partial_dir=str(tmp_path / "partial"),
        )

        assert len(result.engine_reports) == 1
        report = result.engine_reports[0]

        assert report.aggregated_line_metrics is not None, (
            "aggregated_line_metrics doit être peuplé après benchmark"
        )
        assert "gini_mean" in report.aggregated_line_metrics
        assert "document_count" in report.aggregated_line_metrics
        assert report.aggregated_line_metrics["document_count"] == 2

        assert report.aggregated_hallucination is not None, (
            "aggregated_hallucination doit être peuplé après benchmark"
        )
        assert "anchor_score_mean" in report.aggregated_hallucination
        assert report.aggregated_hallucination["document_count"] == 2

    def test_json_export_includes_new_aggregations(self, tmp_path: Path):
        img = tmp_path / "t.png"
        img.write_bytes(b"")
        corpus = Corpus(
            name="test",
            documents=[
                Document(doc_id="d1", image_path=img, ground_truth="un\ndeux"),
            ],
            source_path=str(tmp_path),
        )
        engine = _FakeEngine(output_text="un\ndeux", name="fake")

        out = tmp_path / "bench.json"
        run_benchmark(
            corpus=corpus,
            engines=[engine],
            output_json=out,
            show_progress=False,
            max_workers=1,
            partial_dir=str(tmp_path / "partial"),
        )

        data = json.loads(out.read_text(encoding="utf-8"))
        report = data["engine_reports"][0]
        assert "aggregated_line_metrics" in report
        assert "aggregated_hallucination" in report


# ---------------------------------------------------------------------------
# 3. Modèle Fact et DetectorRegistry
# ---------------------------------------------------------------------------

class TestFactModel:
    def test_fact_is_serializable(self):
        fact = Fact(
            type=FactType.GLOBAL_LEADER_CER,
            importance=FactImportance.CRITICAL,
            payload={"engine": "tesseract", "cer": 0.042},
            engines_involved=("tesseract",),
        )
        d = fact.as_dict()
        assert d["type"] == "global_leader_cer"
        assert d["importance"] == 100
        assert d["payload"]["cer"] == 0.042
        assert d["engines_involved"] == ["tesseract"]

    def test_fact_importance_ordering(self):
        assert FactImportance.CRITICAL > FactImportance.HIGH
        assert FactImportance.HIGH > FactImportance.MEDIUM
        assert FactImportance.MEDIUM > FactImportance.LOW


class TestDetectorRegistry:
    def test_registry_starts_empty(self):
        registry = DetectorRegistry()
        assert registry.registered_types() == ()
        assert registry.run({}) == []

    def test_register_and_run(self):
        registry = DetectorRegistry()

        def dummy_detector(data: dict) -> list[Fact]:
            return [Fact(
                type=FactType.GLOBAL_LEADER_CER,
                importance=FactImportance.CRITICAL,
                payload={"engine": data.get("leader", "unknown")},
            )]

        registry.register(FactType.GLOBAL_LEADER_CER, dummy_detector)
        assert FactType.GLOBAL_LEADER_CER in registry.registered_types()

        facts = registry.run({"leader": "tesseract"})
        assert len(facts) == 1
        assert facts[0].payload["engine"] == "tesseract"

    def test_registry_swallows_detector_exceptions(self):
        """Un détecteur défaillant ne doit pas casser le pipeline narratif."""
        registry = DetectorRegistry()

        def broken_detector(data: dict) -> list[Fact]:
            raise RuntimeError("boom")

        def working_detector(data: dict) -> list[Fact]:
            return [Fact(
                type=FactType.SPEED_WINNER,
                importance=FactImportance.HIGH,
                payload={},
            )]

        registry.register(FactType.GLOBAL_LEADER_CER, broken_detector)
        registry.register(FactType.SPEED_WINNER, working_detector)

        facts = registry.run({})
        assert len(facts) == 1
        assert facts[0].type == FactType.SPEED_WINNER

    def test_default_registry_is_empty_in_sprint_1(self):
        """Sprint 1 = fondations uniquement. Aucun détecteur n'est activé
        par défaut — ils le seront au Sprint 4 avec leurs templates."""
        facts = detect_all({})
        assert facts == []
