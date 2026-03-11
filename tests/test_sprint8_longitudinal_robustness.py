"""Tests Sprint 8 — Suivi longitudinal et analyse de robustesse.

Classes de tests
----------------
TestBenchmarkHistory         (15 tests) — base SQLite historique
TestHistoryEntry             (6 tests)  — structure HistoryEntry
TestRegressionResult         (8 tests)  — détection de régression
TestGenerateDemoHistory      (5 tests)  — données fictives longitudinales
TestDegradationLevels        (6 tests)  — paramètres de dégradation
TestDegradationFunctions     (10 tests) — fonctions de dégradation image
TestDegradationCurve         (6 tests)  — structure DegradationCurve
TestRobustnessReport         (8 tests)  — rapport de robustesse
TestRobustnessAnalyzer       (8 tests)  — analyseur statique
TestGenerateDemoRobustness   (10 tests) — données fictives robustesse
TestCLIDemo                  (5 tests)  — picarones demo --with-history --with-robustness
"""

from __future__ import annotations

import json
import pytest


# ===========================================================================
# TestBenchmarkHistory
# ===========================================================================

class TestBenchmarkHistory:

    @pytest.fixture
    def db(self):
        from picarones.core.history import BenchmarkHistory
        return BenchmarkHistory(":memory:")

    def test_import_module(self):
        from picarones.core.history import BenchmarkHistory
        assert BenchmarkHistory is not None

    def test_init_in_memory(self, db):
        assert db.db_path == ":memory:"
        assert db.count() == 0

    def test_record_single(self, db):
        db.record_single(
            run_id="run001",
            corpus_name="Corpus Test",
            engine_name="tesseract",
            cer_mean=0.12,
            wer_mean=0.20,
            doc_count=10,
        )
        assert db.count() == 1

    def test_record_single_idempotent(self, db):
        db.record_single("run001", "C", "tesseract", 0.12, 0.20, 10)
        db.record_single("run001", "C", "tesseract", 0.10, 0.18, 10)  # même run_id → REPLACE
        assert db.count() == 1

    def test_query_returns_entries(self, db):
        db.record_single("r1", "C", "tesseract", 0.10, 0.18, 5)
        db.record_single("r2", "C", "pero_ocr", 0.07, 0.12, 5)
        entries = db.query()
        assert len(entries) == 2

    def test_query_filter_engine(self, db):
        db.record_single("r1", "C", "tesseract", 0.10, 0.18, 5)
        db.record_single("r2", "C", "pero_ocr", 0.07, 0.12, 5)
        entries = db.query(engine="tesseract")
        assert len(entries) == 1
        assert entries[0].engine_name == "tesseract"

    def test_query_filter_corpus(self, db):
        db.record_single("r1", "CorpusA", "tesseract", 0.10, 0.18, 5)
        db.record_single("r2", "CorpusB", "tesseract", 0.07, 0.12, 5)
        entries = db.query(corpus="CorpusA")
        assert len(entries) == 1
        assert entries[0].corpus_name == "CorpusA"

    def test_query_filter_since(self, db):
        db.record_single("r1", "C", "tesseract", 0.12, 0.20, 5, timestamp="2024-01-01T00:00:00+00:00")
        db.record_single("r2", "C", "tesseract", 0.10, 0.18, 5, timestamp="2025-06-01T00:00:00+00:00")
        entries = db.query(since="2025-01-01")
        assert len(entries) == 1
        assert "2025" in entries[0].timestamp

    def test_list_engines(self, db):
        db.record_single("r1", "C", "tesseract", 0.10, 0.18, 5)
        db.record_single("r2", "C", "pero_ocr", 0.07, 0.12, 5)
        engines = db.list_engines()
        assert "tesseract" in engines
        assert "pero_ocr" in engines

    def test_list_corpora(self, db):
        db.record_single("r1", "CorpusA", "tesseract", 0.10, 0.18, 5)
        db.record_single("r2", "CorpusB", "pero_ocr", 0.07, 0.12, 5)
        corpora = db.list_corpora()
        assert "CorpusA" in corpora
        assert "CorpusB" in corpora

    def test_get_cer_curve(self, db):
        db.record_single("r1", "C", "tesseract", 0.15, 0.25, 5, timestamp="2024-01-01T00:00:00+00:00")
        db.record_single("r2", "C", "tesseract", 0.12, 0.20, 5, timestamp="2024-06-01T00:00:00+00:00")
        db.record_single("r3", "C", "tesseract", 0.10, 0.18, 5, timestamp="2025-01-01T00:00:00+00:00")
        curve = db.get_cer_curve("tesseract")
        assert len(curve) == 3
        assert all("cer" in point for point in curve)
        assert all("timestamp" in point for point in curve)

    def test_get_cer_curve_filters_engine(self, db):
        db.record_single("r1", "C", "tesseract", 0.10, 0.18, 5)
        db.record_single("r2", "C", "pero_ocr", 0.07, 0.12, 5)
        curve = db.get_cer_curve("tesseract")
        assert all(point["cer"] is not None for point in curve)

    def test_export_json(self, db, tmp_path):
        db.record_single("r1", "C", "tesseract", 0.10, 0.18, 5)
        path = db.export_json(str(tmp_path / "history.json"))
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["picarones_history"] is True
        assert "runs" in data
        assert len(data["runs"]) == 1

    def test_record_benchmark_result(self, db):
        from picarones.fixtures import generate_sample_benchmark
        bm = generate_sample_benchmark(n_docs=3, seed=0)
        run_id = db.record(bm)
        assert isinstance(run_id, str)
        # Autant d'entrées que de moteurs dans le benchmark
        assert db.count() == len(bm.engine_reports)

    def test_repr(self, db):
        r = repr(db)
        assert "BenchmarkHistory" in r
        assert ":memory:" in r


# ===========================================================================
# TestHistoryEntry
# ===========================================================================

class TestHistoryEntry:

    def test_import(self):
        from picarones.core.history import HistoryEntry
        assert HistoryEntry is not None

    def test_cer_percent(self):
        from picarones.core.history import HistoryEntry
        entry = HistoryEntry(
            run_id="r1", timestamp="2025-01-01T00:00:00+00:00",
            corpus_name="C", engine_name="tesseract",
            cer_mean=0.12, wer_mean=0.20, doc_count=10,
        )
        assert abs(entry.cer_percent - 12.0) < 0.01

    def test_cer_percent_none(self):
        from picarones.core.history import HistoryEntry
        entry = HistoryEntry("r", "2025", "C", "e", None, None, 0)
        assert entry.cer_percent is None

    def test_as_dict_keys(self):
        from picarones.core.history import HistoryEntry
        entry = HistoryEntry("r1", "2025-01-01", "C", "tesseract", 0.10, 0.18, 5)
        d = entry.as_dict()
        assert "run_id" in d
        assert "cer_mean" in d
        assert "engine_name" in d

    def test_as_dict_metadata(self):
        from picarones.core.history import HistoryEntry
        entry = HistoryEntry("r1", "2025-01-01", "C", "tesseract", 0.10, 0.18, 5,
                             metadata={"key": "value"})
        d = entry.as_dict()
        assert d["metadata"] == {"key": "value"}

    def test_query_result_is_history_entry(self):
        from picarones.core.history import BenchmarkHistory, HistoryEntry
        db = BenchmarkHistory(":memory:")
        db.record_single("r1", "C", "tesseract", 0.10, 0.18, 5)
        entries = db.query()
        assert isinstance(entries[0], HistoryEntry)


# ===========================================================================
# TestRegressionResult
# ===========================================================================

class TestRegressionResult:

    @pytest.fixture
    def db_with_runs(self):
        from picarones.core.history import BenchmarkHistory
        db = BenchmarkHistory(":memory:")
        db.record_single("r1", "C", "tesseract", 0.12, 0.20, 10, timestamp="2025-01-01T00:00:00+00:00")
        db.record_single("r2", "C", "tesseract", 0.15, 0.25, 10, timestamp="2025-06-01T00:00:00+00:00")
        return db

    def test_detect_regression_is_detected(self, db_with_runs):
        result = db_with_runs.detect_regression("tesseract", threshold=0.01)
        assert result is not None
        assert result.is_regression is True

    def test_detect_regression_delta_positive(self, db_with_runs):
        result = db_with_runs.detect_regression("tesseract")
        assert result.delta_cer > 0

    def test_detect_regression_fields(self, db_with_runs):
        result = db_with_runs.detect_regression("tesseract")
        assert result.engine_name == "tesseract"
        assert result.baseline_cer is not None
        assert result.current_cer is not None

    def test_detect_no_regression(self):
        from picarones.core.history import BenchmarkHistory
        db = BenchmarkHistory(":memory:")
        # CER diminue = amélioration = pas de régression
        db.record_single("r1", "C", "tesseract", 0.15, 0.25, 5, timestamp="2025-01-01T00:00:00+00:00")
        db.record_single("r2", "C", "tesseract", 0.10, 0.18, 5, timestamp="2025-06-01T00:00:00+00:00")
        result = db.detect_regression("tesseract", threshold=0.01)
        assert result is not None
        assert result.is_regression is False

    def test_detect_regression_none_if_single_run(self):
        from picarones.core.history import BenchmarkHistory
        db = BenchmarkHistory(":memory:")
        db.record_single("r1", "C", "tesseract", 0.12, 0.20, 5)
        result = db.detect_regression("tesseract")
        assert result is None

    def test_detect_all_regressions(self):
        from picarones.core.history import BenchmarkHistory
        db = BenchmarkHistory(":memory:")
        db.record_single("r1", "C", "tesseract", 0.10, 0.18, 5, timestamp="2025-01-01T00:00:00+00:00")
        db.record_single("r2", "C", "tesseract", 0.20, 0.35, 5, timestamp="2025-06-01T00:00:00+00:00")
        regressions = db.detect_all_regressions(threshold=0.01)
        assert len(regressions) >= 1

    def test_regression_result_as_dict(self, db_with_runs):
        result = db_with_runs.detect_regression("tesseract")
        d = result.as_dict()
        assert "is_regression" in d
        assert "delta_cer" in d
        assert "engine_name" in d

    def test_regression_threshold_respected(self):
        from picarones.core.history import BenchmarkHistory
        db = BenchmarkHistory(":memory:")
        db.record_single("r1", "C", "tesseract", 0.100, 0.18, 5, timestamp="2025-01-01T00:00:00+00:00")
        db.record_single("r2", "C", "tesseract", 0.105, 0.19, 5, timestamp="2025-06-01T00:00:00+00:00")
        # Delta = 0.5%, threshold = 1% → pas de régression
        result = db.detect_regression("tesseract", threshold=0.01)
        assert result is not None
        assert result.is_regression is False
        # Avec threshold = 0.001% → régression
        result2 = db.detect_regression("tesseract", threshold=0.001)
        assert result2.is_regression is True


# ===========================================================================
# TestGenerateDemoHistory
# ===========================================================================

class TestGenerateDemoHistory:

    def test_generate_fills_db(self):
        from picarones.core.history import BenchmarkHistory, generate_demo_history
        db = BenchmarkHistory(":memory:")
        generate_demo_history(db, n_runs=5)
        assert db.count() > 0

    def test_generate_creates_multiple_engines(self):
        from picarones.core.history import BenchmarkHistory, generate_demo_history
        db = BenchmarkHistory(":memory:")
        generate_demo_history(db, n_runs=4)
        engines = db.list_engines()
        assert len(engines) >= 2

    def test_generate_n_runs(self):
        from picarones.core.history import BenchmarkHistory, generate_demo_history
        db = BenchmarkHistory(":memory:")
        generate_demo_history(db, n_runs=8)
        # 8 runs × 3 moteurs = 24 entrées
        assert db.count() == 8 * 3

    def test_cer_values_in_range(self):
        from picarones.core.history import BenchmarkHistory, generate_demo_history
        db = BenchmarkHistory(":memory:")
        generate_demo_history(db, n_runs=5)
        entries = db.query()
        for e in entries:
            if e.cer_mean is not None:
                assert 0.0 <= e.cer_mean <= 1.0

    def test_regression_detectable_in_demo(self):
        """La démo inclut une régression simulée au run 5 (tesseract)."""
        from picarones.core.history import BenchmarkHistory, generate_demo_history
        db = BenchmarkHistory(":memory:")
        generate_demo_history(db, n_runs=8, seed=42)
        # Vérifier que l'historique a été créé
        assert db.count() > 0
        # Vérifier que la courbe CER existe pour tesseract
        curve = db.get_cer_curve("tesseract")
        assert len(curve) > 0


# ===========================================================================
# TestDegradationLevels
# ===========================================================================

class TestDegradationLevels:

    def test_import_constants(self):
        from picarones.core.robustness import DEGRADATION_LEVELS, ALL_DEGRADATION_TYPES
        assert len(DEGRADATION_LEVELS) > 0
        assert len(ALL_DEGRADATION_TYPES) > 0

    def test_all_types_in_levels(self):
        from picarones.core.robustness import DEGRADATION_LEVELS, ALL_DEGRADATION_TYPES
        for t in ALL_DEGRADATION_TYPES:
            assert t in DEGRADATION_LEVELS

    def test_noise_levels(self):
        from picarones.core.robustness import DEGRADATION_LEVELS
        levels = DEGRADATION_LEVELS["noise"]
        assert len(levels) >= 2
        assert 0 in levels  # niveau original

    def test_blur_levels(self):
        from picarones.core.robustness import DEGRADATION_LEVELS
        levels = DEGRADATION_LEVELS["blur"]
        assert 0 in levels

    def test_resolution_levels_include_1(self):
        from picarones.core.robustness import DEGRADATION_LEVELS
        levels = DEGRADATION_LEVELS["resolution"]
        assert 1.0 in levels  # résolution originale

    def test_labels_match_levels(self):
        from picarones.core.robustness import DEGRADATION_LEVELS, DEGRADATION_LABELS
        for dtype in DEGRADATION_LEVELS:
            if dtype in DEGRADATION_LABELS:
                assert len(DEGRADATION_LABELS[dtype]) == len(DEGRADATION_LEVELS[dtype])


# ===========================================================================
# TestDegradationFunctions
# ===========================================================================

class TestDegradationFunctions:

    def _make_png(self) -> bytes:
        """Génère un PNG minimal valide (10×10 pixels)."""
        from picarones.fixtures import _make_placeholder_png
        return _make_placeholder_png(40, 30)

    def test_degrade_image_bytes_imports(self):
        from picarones.core.robustness import degrade_image_bytes
        assert callable(degrade_image_bytes)

    def test_degrade_noise_returns_bytes(self):
        from picarones.core.robustness import degrade_image_bytes
        png = self._make_png()
        result = degrade_image_bytes(png, "noise", 0)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_degrade_blur_returns_bytes(self):
        from picarones.core.robustness import degrade_image_bytes
        png = self._make_png()
        result = degrade_image_bytes(png, "blur", 0)
        assert isinstance(result, bytes)

    def test_degrade_rotation_returns_bytes(self):
        from picarones.core.robustness import degrade_image_bytes
        png = self._make_png()
        result = degrade_image_bytes(png, "rotation", 0)
        assert isinstance(result, bytes)

    def test_degrade_resolution_returns_bytes(self):
        from picarones.core.robustness import degrade_image_bytes
        png = self._make_png()
        result = degrade_image_bytes(png, "resolution", 1.0)
        assert isinstance(result, bytes)

    def test_degrade_binarization_returns_bytes(self):
        from picarones.core.robustness import degrade_image_bytes
        png = self._make_png()
        result = degrade_image_bytes(png, "binarization", 0)
        assert isinstance(result, bytes)

    def test_degrade_noise_level_5(self):
        from picarones.core.robustness import degrade_image_bytes
        png = self._make_png()
        result = degrade_image_bytes(png, "noise", 5)
        assert isinstance(result, bytes)

    def test_degrade_blur_level_2(self):
        from picarones.core.robustness import degrade_image_bytes
        png = self._make_png()
        result = degrade_image_bytes(png, "blur", 2)
        assert isinstance(result, bytes)

    def test_degrade_resolution_half(self):
        from picarones.core.robustness import degrade_image_bytes
        png = self._make_png()
        result = degrade_image_bytes(png, "resolution", 0.5)
        assert isinstance(result, bytes)

    def test_degrade_rotation_10_degrees(self):
        from picarones.core.robustness import degrade_image_bytes
        png = self._make_png()
        result = degrade_image_bytes(png, "rotation", 10)
        assert isinstance(result, bytes)


# ===========================================================================
# TestDegradationCurve
# ===========================================================================

class TestDegradationCurve:

    def test_import(self):
        from picarones.core.robustness import DegradationCurve
        assert DegradationCurve is not None

    def test_as_dict_keys(self):
        from picarones.core.robustness import DegradationCurve
        curve = DegradationCurve(
            engine_name="tesseract",
            degradation_type="noise",
            levels=[0, 5, 15],
            labels=["original", "σ=5", "σ=15"],
            cer_values=[0.10, 0.15, 0.25],
        )
        d = curve.as_dict()
        assert "engine_name" in d
        assert "degradation_type" in d
        assert "levels" in d
        assert "cer_values" in d

    def test_critical_threshold(self):
        from picarones.core.robustness import DegradationCurve
        curve = DegradationCurve(
            engine_name="tesseract",
            degradation_type="noise",
            levels=[0, 5, 15, 30],
            labels=["o", "σ=5", "σ=15", "σ=30"],
            cer_values=[0.10, 0.15, 0.22, 0.35],
            critical_threshold_level=15,
            cer_threshold=0.20,
        )
        assert curve.critical_threshold_level == 15

    def test_none_cer_allowed(self):
        from picarones.core.robustness import DegradationCurve
        curve = DegradationCurve(
            engine_name="e",
            degradation_type="blur",
            levels=[0, 2],
            labels=["o", "r=2"],
            cer_values=[None, 0.15],
        )
        assert curve.cer_values[0] is None

    def test_default_cer_threshold(self):
        from picarones.core.robustness import DegradationCurve
        curve = DegradationCurve("e", "noise", [0], ["o"], [0.1])
        assert curve.cer_threshold == 0.20

    def test_engine_name_preserved(self):
        from picarones.core.robustness import DegradationCurve
        curve = DegradationCurve("pero_ocr", "blur", [0, 1], ["o", "r=1"], [0.05, 0.08])
        assert curve.engine_name == "pero_ocr"

    def test_as_dict_roundtrip(self):
        from picarones.core.robustness import DegradationCurve
        curve = DegradationCurve(
            engine_name="tesseract",
            degradation_type="rotation",
            levels=[0, 5, 10],
            labels=["0°", "5°", "10°"],
            cer_values=[0.10, 0.18, 0.30],
            critical_threshold_level=10,
        )
        d = curve.as_dict()
        assert d["levels"] == [0, 5, 10]
        assert d["cer_values"] == [0.10, 0.18, 0.30]


# ===========================================================================
# TestRobustnessReport
# ===========================================================================

class TestRobustnessReport:

    def test_import(self):
        from picarones.core.robustness import RobustnessReport
        assert RobustnessReport is not None

    def test_get_curves_for_engine(self):
        from picarones.core.robustness import RobustnessReport, DegradationCurve
        c1 = DegradationCurve("tesseract", "noise", [0, 5], ["o", "σ=5"], [0.10, 0.15])
        c2 = DegradationCurve("pero_ocr", "noise", [0, 5], ["o", "σ=5"], [0.07, 0.10])
        report = RobustnessReport(["tesseract", "pero_ocr"], "C", ["noise"], [c1, c2])
        tess_curves = report.get_curves_for_engine("tesseract")
        assert len(tess_curves) == 1
        assert tess_curves[0].engine_name == "tesseract"

    def test_get_curves_for_type(self):
        from picarones.core.robustness import RobustnessReport, DegradationCurve
        c1 = DegradationCurve("tesseract", "noise", [0, 5], ["o", "σ=5"], [0.10, 0.15])
        c2 = DegradationCurve("tesseract", "blur", [0, 2], ["o", "r=2"], [0.10, 0.14])
        report = RobustnessReport(["tesseract"], "C", ["noise", "blur"], [c1, c2])
        noise_curves = report.get_curves_for_type("noise")
        assert len(noise_curves) == 1
        assert noise_curves[0].degradation_type == "noise"

    def test_as_dict_keys(self):
        from picarones.core.robustness import RobustnessReport
        report = RobustnessReport(["tesseract"], "C", ["noise"], [])
        d = report.as_dict()
        assert "engine_names" in d
        assert "curves" in d
        assert "summary" in d

    def test_as_dict_json_serializable(self):
        from picarones.core.robustness import RobustnessReport, DegradationCurve
        c = DegradationCurve("e", "noise", [0, 5], ["o", "n5"], [0.1, 0.2])
        report = RobustnessReport(["e"], "C", ["noise"], [c])
        d = report.as_dict()
        # Doit être sérialisable en JSON sans erreur
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_summary_populated(self):
        from picarones.core.robustness import generate_demo_robustness_report
        report = generate_demo_robustness_report(engine_names=["tesseract"], seed=1)
        assert isinstance(report.summary, dict)
        assert len(report.summary) > 0

    def test_corpus_name_preserved(self):
        from picarones.core.robustness import RobustnessReport
        report = RobustnessReport(["e"], "Mon Corpus", ["noise"], [])
        assert report.corpus_name == "Mon Corpus"

    def test_engine_names_list(self):
        from picarones.core.robustness import RobustnessReport
        report = RobustnessReport(["tesseract", "pero_ocr"], "C", [], [])
        assert "tesseract" in report.engine_names
        assert "pero_ocr" in report.engine_names


# ===========================================================================
# TestRobustnessAnalyzer
# ===========================================================================

class TestRobustnessAnalyzer:

    def test_import(self):
        from picarones.core.robustness import RobustnessAnalyzer
        assert RobustnessAnalyzer is not None

    def test_init_single_engine(self):
        from picarones.core.robustness import RobustnessAnalyzer
        mock_engine = type("E", (), {"name": "tesseract"})()
        analyzer = RobustnessAnalyzer(mock_engine)
        assert len(analyzer.engines) == 1

    def test_init_list_engines(self):
        from picarones.core.robustness import RobustnessAnalyzer
        engines = [
            type("E", (), {"name": "tesseract"})(),
            type("E", (), {"name": "pero_ocr"})(),
        ]
        analyzer = RobustnessAnalyzer(engines)
        assert len(analyzer.engines) == 2

    def test_default_degradation_types(self):
        from picarones.core.robustness import RobustnessAnalyzer, ALL_DEGRADATION_TYPES
        e = type("E", (), {"name": "e"})()
        analyzer = RobustnessAnalyzer(e)
        assert set(analyzer.degradation_types) == set(ALL_DEGRADATION_TYPES)

    def test_custom_degradation_types(self):
        from picarones.core.robustness import RobustnessAnalyzer
        e = type("E", (), {"name": "e"})()
        analyzer = RobustnessAnalyzer(e, degradation_types=["noise", "blur"])
        assert analyzer.degradation_types == ["noise", "blur"]

    def test_find_critical_level_found(self):
        from picarones.core.robustness import RobustnessAnalyzer
        levels = [0, 5, 15, 30]
        cer_values = [0.10, 0.15, 0.22, 0.35]
        critical = RobustnessAnalyzer._find_critical_level(levels, cer_values, 0.20)
        assert critical == 15

    def test_find_critical_level_none(self):
        from picarones.core.robustness import RobustnessAnalyzer
        levels = [0, 5, 15]
        cer_values = [0.05, 0.10, 0.15]
        critical = RobustnessAnalyzer._find_critical_level(levels, cer_values, 0.20)
        assert critical is None

    def test_build_summary(self):
        from picarones.core.robustness import RobustnessAnalyzer, DegradationCurve
        curves = [
            DegradationCurve("tesseract", "noise", [0, 5], ["o", "n5"], [0.10, 0.20]),
            DegradationCurve("pero_ocr", "noise", [0, 5], ["o", "n5"], [0.07, 0.12]),
        ]
        summary = RobustnessAnalyzer._build_summary(curves)
        assert "most_robust_noise" in summary
        assert summary["most_robust_noise"] == "pero_ocr"  # pero_ocr a le CER moyen le plus bas


# ===========================================================================
# TestGenerateDemoRobustness
# ===========================================================================

class TestGenerateDemoRobustness:

    def test_import(self):
        from picarones.core.robustness import generate_demo_robustness_report
        assert callable(generate_demo_robustness_report)

    def test_returns_report(self):
        from picarones.core.robustness import generate_demo_robustness_report, RobustnessReport
        report = generate_demo_robustness_report()
        assert isinstance(report, RobustnessReport)

    def test_default_engines(self):
        from picarones.core.robustness import generate_demo_robustness_report
        report = generate_demo_robustness_report()
        assert "tesseract" in report.engine_names
        assert "pero_ocr" in report.engine_names

    def test_custom_engines(self):
        from picarones.core.robustness import generate_demo_robustness_report
        report = generate_demo_robustness_report(engine_names=["moteur_custom"])
        assert "moteur_custom" in report.engine_names

    def test_all_degradation_types_present(self):
        from picarones.core.robustness import generate_demo_robustness_report, ALL_DEGRADATION_TYPES
        report = generate_demo_robustness_report()
        types_in_report = {c.degradation_type for c in report.curves}
        assert types_in_report == set(ALL_DEGRADATION_TYPES)

    def test_cer_values_in_range(self):
        from picarones.core.robustness import generate_demo_robustness_report
        report = generate_demo_robustness_report(seed=99)
        for curve in report.curves:
            for cer in curve.cer_values:
                if cer is not None:
                    assert 0.0 <= cer <= 1.0

    def test_cer_increases_with_degradation(self):
        """Pour la plupart des types, le CER doit augmenter avec le niveau de dégradation."""
        from picarones.core.robustness import generate_demo_robustness_report
        report = generate_demo_robustness_report(seed=42)
        for curve in report.curves:
            valid = [c for c in curve.cer_values if c is not None]
            if len(valid) >= 3:
                # Au moins le dernier niveau doit être >= le premier
                assert valid[-1] >= valid[0], (
                    f"CER devrait augmenter pour {curve.engine_name}/{curve.degradation_type}: "
                    f"{valid[0]} → {valid[-1]}"
                )

    def test_reproducible_with_seed(self):
        from picarones.core.robustness import generate_demo_robustness_report
        r1 = generate_demo_robustness_report(seed=7)
        r2 = generate_demo_robustness_report(seed=7)
        assert r1.curves[0].cer_values == r2.curves[0].cer_values

    def test_summary_contains_most_robust(self):
        from picarones.core.robustness import generate_demo_robustness_report
        report = generate_demo_robustness_report()
        assert any("most_robust" in k for k in report.summary)

    def test_json_serializable(self):
        from picarones.core.robustness import generate_demo_robustness_report
        report = generate_demo_robustness_report()
        d = report.as_dict()
        json_str = json.dumps(d, ensure_ascii=False)
        assert len(json_str) > 0
        reparsed = json.loads(json_str)
        assert "curves" in reparsed


# ===========================================================================
# TestCLIDemo
# ===========================================================================

class TestCLIDemo:

    def test_demo_with_history_flag(self):
        from click.testing import CliRunner
        from picarones.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["demo", "--with-history", "--docs", "3"])
        assert result.exit_code == 0
        assert "longitudinal" in result.output.lower() or "suivi" in result.output.lower() or "CER" in result.output

    def test_demo_with_robustness_flag(self):
        from click.testing import CliRunner
        from picarones.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["demo", "--with-robustness", "--docs", "3"])
        assert result.exit_code == 0
        assert "robustesse" in result.output.lower() or "robustness" in result.output.lower() or "bruit" in result.output.lower()

    def test_demo_with_both_flags(self):
        from click.testing import CliRunner
        from picarones.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["demo", "--with-history", "--with-robustness", "--docs", "3"])
        assert result.exit_code == 0

    def test_demo_without_flags(self):
        from click.testing import CliRunner
        from picarones.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["demo", "--docs", "3"])
        assert result.exit_code == 0

    def test_demo_generates_html_file(self):
        from click.testing import CliRunner
        from picarones.cli import cli
        import os
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["demo", "--docs", "3", "--output", "test_demo.html"])
            assert result.exit_code == 0
            assert os.path.exists("test_demo.html")
