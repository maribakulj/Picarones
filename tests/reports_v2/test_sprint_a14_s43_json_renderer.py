"""Sprint A14-S43 — ``JsonReportRenderer``."""

from __future__ import annotations

import json

from picarones.app.results import RunDocumentResult, RunResult
from picarones.domain import RunManifest, utcnow
from picarones.evaluation.views.base import ViewResult
from picarones.reports_v2.json import JsonReportRenderer


def _make_result(view_results: tuple[ViewResult, ...] = ()) -> RunResult:
    started = utcnow()
    completed = utcnow()
    manifest = RunManifest(
        run_id="run_001",
        corpus_name="demo",
        n_documents=1,
        pipeline_names=("tess",),
        view_specs=(),
        code_version="1.0.0-s43",
        started_at=started,
        completed_at=completed,
    )
    return RunResult(
        manifest=manifest,
        document_results=(
            RunDocumentResult(
                document_id="doc01",
                pipeline_results=(),
                view_results=view_results,
            ),
        ),
    )


# ──────────────────────────────────────────────────────────────────────
# Renderer
# ──────────────────────────────────────────────────────────────────────


class TestJsonRendererStructure:
    def test_includes_manifest_and_documents(self) -> None:
        result = _make_result()
        text = JsonReportRenderer().render(result)
        data = json.loads(text)
        assert "run_manifest" in data
        assert "documents" in data
        assert isinstance(data["documents"], list)
        assert len(data["documents"]) == 1

    def test_manifest_has_run_id(self) -> None:
        result = _make_result()
        text = JsonReportRenderer().render(result)
        data = json.loads(text)
        assert data["run_manifest"]["run_id"] == "run_001"
        assert data["run_manifest"]["corpus_name"] == "demo"

    def test_document_has_pipeline_and_view_results(self) -> None:
        view_result = ViewResult(
            view_name="text_final",
            pipeline_name="tess",
            candidate_artifact_id="doc01:tess:raw_text",
            ground_truth_artifact_id="doc01:gt",
            metric_values={"cer": 0.05},
        )
        result = _make_result(view_results=(view_result,))
        text = JsonReportRenderer().render(result)
        data = json.loads(text)
        doc = data["documents"][0]
        assert doc["document_id"] == "doc01"
        assert doc["pipeline_results"] == []
        assert len(doc["view_results"]) == 1
        assert doc["view_results"][0]["metric_values"] == {"cer": 0.05}


class TestJsonRendererDeterminism:
    def test_render_twice_yields_identical_bytes(self) -> None:
        result = _make_result()
        renderer = JsonReportRenderer()
        a = renderer.render(result)
        b = renderer.render(result)
        assert a == b

    def test_keys_sorted(self) -> None:
        result = _make_result()
        text = JsonReportRenderer().render(result)
        # Les clés top-level doivent apparaître triées : "documents"
        # avant "run_manifest" alphabétiquement.
        assert text.find('"documents"') < text.find('"run_manifest"')

    def test_unicode_preserved(self) -> None:
        view_result = ViewResult(
            view_name="text_final",
            pipeline_name="tess",
            candidate_artifact_id="doc01:tess:raw_text",
            ground_truth_artifact_id="doc01:gt",
            warnings=("français médiéval",),
        )
        result = _make_result(view_results=(view_result,))
        text = JsonReportRenderer().render(result)
        # Pas d'\u escapes (ensure_ascii=False).
        assert "français médiéval" in text


class TestJsonRendererIndentation:
    def test_uses_indent_2(self) -> None:
        result = _make_result()
        text = JsonReportRenderer().render(result)
        # indent=2 → des paires de spaces en début de ligne.
        assert "\n  \"" in text or "\n  \"" in text


class TestJsonRendererEmptyResult:
    def test_empty_documents_yields_empty_list(self) -> None:
        started = utcnow()
        manifest = RunManifest(
            run_id="run_empty",
            corpus_name="empty",
            n_documents=0,
            pipeline_names=(),
            view_specs=(),
            code_version="1.0.0-s43",
            started_at=started,
            completed_at=started,
        )
        result = RunResult(manifest=manifest, document_results=())
        text = JsonReportRenderer().render(result)
        data = json.loads(text)
        assert data["documents"] == []
        assert data["run_manifest"]["run_id"] == "run_empty"
