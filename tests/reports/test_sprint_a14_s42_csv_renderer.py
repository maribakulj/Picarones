"""Sprint A14-S42 — ``CsvReportRenderer``."""

from __future__ import annotations

import csv
import io

from picarones.app.results import RunDocumentResult, RunResult
from picarones.domain import RunManifest, utcnow
from picarones.evaluation.views.base import ViewResult
from picarones.reports.csv import CsvReportRenderer


def _make_minimal_result(
    metric_values: dict | None = None,
    failed_metrics: dict | None = None,
    candidate_artifact_id: str = "doc01:tess:raw_text",
    pipeline_name: str = "tess",
) -> RunResult:
    started = utcnow()
    completed = utcnow()
    manifest = RunManifest(
        run_id="run_001",
        corpus_name="demo",
        n_documents=1,
        pipeline_names=(pipeline_name,),
        view_specs=(),
        code_version="1.0.0-s42",
        started_at=started,
        completed_at=completed,
    )
    view_result = ViewResult(
        view_name="text_final",
        pipeline_name=pipeline_name,
        candidate_artifact_id=candidate_artifact_id,
        ground_truth_artifact_id="doc01:gt",
        metric_values=metric_values or {},
        failed_metrics=failed_metrics or {},
    )
    return RunResult(
        manifest=manifest,
        document_results=(
            RunDocumentResult(
                document_id="doc01",
                pipeline_results=(),
                view_results=(view_result,),
            ),
        ),
    )


# ──────────────────────────────────────────────────────────────────────
# Renderer
# ──────────────────────────────────────────────────────────────────────


class TestCsvRendererHeader:
    def test_header_columns_in_order(self) -> None:
        result = _make_minimal_result()
        text = CsvReportRenderer().render(result)
        # Première ligne = header.
        first_line = text.splitlines()[0]
        cols = first_line.split(",")
        expected = list(CsvReportRenderer.HEADER)
        assert cols == expected


class TestCsvRendererSuccessfulMetrics:
    def test_successful_metric_emits_value_and_status_ok(self) -> None:
        result = _make_minimal_result(
            metric_values={"cer": 0.12, "wer": 0.25},
        )
        text = CsvReportRenderer().render(result)
        rows = list(csv.DictReader(io.StringIO(text)))
        assert len(rows) == 2
        cer_row = next(r for r in rows if r["metric_name"] == "cer")
        assert cer_row["status"] == "ok"
        assert cer_row["value"] == "0.120000"
        assert cer_row["pipeline_name"] == "tess"

    def test_value_formatted_to_6_decimals(self) -> None:
        result = _make_minimal_result(
            metric_values={"cer": 1.0 / 3.0},
        )
        text = CsvReportRenderer().render(result)
        rows = list(csv.DictReader(io.StringIO(text)))
        assert rows[0]["value"] == "0.333333"


class TestCsvRendererFailedMetrics:
    def test_failed_metric_emits_empty_value_and_status(self) -> None:
        result = _make_minimal_result(
            failed_metrics={"broken": "ValueError: x"},
        )
        text = CsvReportRenderer().render(result)
        rows = list(csv.DictReader(io.StringIO(text)))
        assert len(rows) == 1
        assert rows[0]["metric_name"] == "broken"
        assert rows[0]["status"] == "failed_metric"
        assert rows[0]["value"] == ""


class TestCsvRendererPipelineName:
    def test_pipeline_name_from_view_result_field(self) -> None:
        """``pipeline_name`` est lu directement depuis ``ViewResult.pipeline_name``,
        pas inféré par parsing de ``candidate_artifact_id``.
        """
        result = _make_minimal_result(
            metric_values={"cer": 0.0},
            pipeline_name="my_pipe",
            candidate_artifact_id="doc01:irrelevant_string:raw_text",
        )
        text = CsvReportRenderer().render(result)
        rows = list(csv.DictReader(io.StringIO(text)))
        assert rows[0]["pipeline_name"] == "my_pipe"

    def test_pipeline_name_independent_of_artifact_id(self) -> None:
        """Le ``candidate_artifact_id`` peut contenir n'importe quoi —
        ``pipeline_name`` reste celui du champ structurel.
        """
        result = _make_minimal_result(
            metric_values={"cer": 0.0},
            pipeline_name="real_pipeline",
            candidate_artifact_id="bad_id_no_separators",
        )
        text = CsvReportRenderer().render(result)
        rows = list(csv.DictReader(io.StringIO(text)))
        assert rows[0]["pipeline_name"] == "real_pipeline"


class TestCsvRendererDeterminism:
    def test_render_twice_yields_same_bytes(self) -> None:
        result = _make_minimal_result(
            metric_values={"cer": 0.1, "wer": 0.2, "mer": 0.15},
        )
        renderer = CsvReportRenderer()
        a = renderer.render(result)
        b = renderer.render(result)
        assert a == b
