"""Sprint A14-S41 — ``artifacts_index.jsonl`` séparé.

Tests de la séparation introduite au S41 :

- ``BenchmarkService.persist`` produit un 4ᵉ fichier
  ``artifacts_index.jsonl`` distinct des ``pipeline_results.jsonl``.
- ``pipeline_results.jsonl`` ne contient plus la liste des artefacts.
- Round-trip via ``HtmlReportRenderer.load_run_result`` ré-attache
  les artefacts depuis l'index séparé.
- Compatibilité descendante : un run persisté sans
  ``artifacts_index.jsonl`` (legacy avant S41) reste lisible.
"""

from __future__ import annotations

import json
from pathlib import Path

from picarones.app.results import RunDocumentResult, RunResult
from picarones.domain import (
    Artifact,
    ArtifactType,
    ProvenanceRecord,
    RunManifest,
    utcnow,
)
from picarones.pipeline.types import PipelineResult, StepResult
from picarones.reports_v2.html.render import HtmlReportRenderer


def _make_run_result_with_artifacts() -> RunResult:
    """Construit un RunResult en mémoire avec quelques artefacts."""
    started = utcnow()
    completed = utcnow()
    manifest = RunManifest(
        run_id="run_001",
        corpus_name="demo",
        n_documents=2,
        pipeline_names=("ocr_only",),
        view_specs=(),
        code_version="1.0.0-s41-test",
        started_at=started,
        completed_at=completed,
    )
    artifact1 = Artifact(
        id="doc01:image",
        document_id="doc01",
        type=ArtifactType.IMAGE,
        content_hash="a" * 64,
    )
    artifact2 = Artifact(
        id="doc01:ocr_only:raw_text",
        document_id="doc01",
        type=ArtifactType.RAW_TEXT,
        content_hash="b" * 64,
        produced_by_step="ocr",
        provenance=ProvenanceRecord(
            code_version="1.0.0-s41-test",
            parameters_hash="c" * 64,
        ),
    )
    pr1 = PipelineResult(
        pipeline_name="ocr_only",
        document_id="doc01",
        step_results=(
            StepResult(
                step_id="ocr",
                succeeded=True,
                duration_seconds=0.5,
                produced_artifacts={"raw_text": "doc01:ocr_only:raw_text"},
            ),
        ),
        succeeded=True,
        duration_seconds=0.5,
        artifacts=(artifact1, artifact2),
    )
    return RunResult(
        manifest=manifest,
        document_results=(
            RunDocumentResult(
                document_id="doc01",
                pipeline_results=(pr1,),
                view_results=(),
            ),
        ),
    )


def _build_benchmark_service():
    """Crée un BenchmarkService minimal pour tester persist()."""
    from picarones.app.services.benchmark_service import BenchmarkService
    from picarones.evaluation.views.executor import (
        DefaultEvaluationViewExecutor,
    )
    from picarones.evaluation.registry import MetricRegistry
    from picarones.evaluation.projectors.registry import ProjectorRegistry
    from picarones.pipeline.executor import PipelineExecutor
    from picarones.pipeline.runner import CorpusRunner

    runner = CorpusRunner(
        PipelineExecutor(adapter_resolver=lambda n: None),
        max_in_flight=1,
        timeout_seconds_per_doc=1.0,
        poll_interval_seconds=0.001,
    )
    view_executor = DefaultEvaluationViewExecutor.from_registries(
        MetricRegistry(), ProjectorRegistry(), lambda art: "",
    )
    return BenchmarkService(
        corpus_runner=runner,
        view_executor=view_executor,
        code_version="1.0.0-s41-test",
    )


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────


class TestArtifactsIndexSeparation:
    def test_persist_writes_4_files(self, tmp_path: Path) -> None:
        """``persist`` doit retourner les 4 chemins (manifest +
        pipeline_results + artifacts_index + view_results)."""
        bench = _build_benchmark_service()
        result = _make_run_result_with_artifacts()
        paths = bench.persist(result, tmp_path)

        assert "manifest" in paths
        assert "pipeline_results" in paths
        assert "artifacts_index" in paths
        assert "view_results" in paths
        for kind, path in paths.items():
            assert path.exists(), f"{kind} non écrit"

    def test_artifacts_index_jsonl_format(self, tmp_path: Path) -> None:
        """Chaque ligne contient un artefact + document_id +
        pipeline_name."""
        bench = _build_benchmark_service()
        result = _make_run_result_with_artifacts()
        bench.persist(result, tmp_path)

        index_path = tmp_path / "artifacts_index.jsonl"
        lines = [
            line for line in index_path.read_text(
                encoding="utf-8",
            ).splitlines() if line.strip()
        ]
        assert len(lines) == 2  # 2 artefacts dans le RunResult

        for line in lines:
            rec = json.loads(line)
            assert "document_id" in rec
            assert "pipeline_name" in rec
            assert rec["document_id"] == "doc01"
            assert rec["pipeline_name"] == "ocr_only"
            assert "id" in rec
            assert "type" in rec

    def test_pipeline_results_jsonl_no_longer_contains_artifacts(
        self, tmp_path: Path,
    ) -> None:
        """``pipeline_results.jsonl`` ne porte plus la liste des
        artefacts (extraite vers l'index)."""
        bench = _build_benchmark_service()
        result = _make_run_result_with_artifacts()
        bench.persist(result, tmp_path)

        pipelines_path = tmp_path / "pipeline_results.jsonl"
        lines = [
            line for line in pipelines_path.read_text(
                encoding="utf-8",
            ).splitlines() if line.strip()
        ]
        assert len(lines) == 1
        rec = json.loads(lines[0])
        # Le champ artifacts ne doit pas apparaître (ou être vide).
        assert (
            "artifacts" not in rec
            or rec.get("artifacts") == []
            or rec.get("artifacts") is None
        )
        # Mais les autres champs (step_results, etc.) sont présents.
        assert rec["pipeline_name"] == "ocr_only"
        assert "step_results" in rec


class TestRoundTripWithIndex:
    def test_load_run_result_reattaches_artifacts(
        self, tmp_path: Path,
    ) -> None:
        """``load_run_result`` lit l'index séparé et ré-attache les
        artefacts à chaque PipelineResult."""
        bench = _build_benchmark_service()
        result = _make_run_result_with_artifacts()
        bench.persist(result, tmp_path)

        loaded = HtmlReportRenderer.load_run_result(tmp_path)
        assert len(loaded.document_results) == 1
        loaded_pr = loaded.document_results[0].pipeline_results[0]
        assert len(loaded_pr.artifacts) == 2
        # Les content_hash doivent être préservés.
        loaded_hashes = {a.content_hash for a in loaded_pr.artifacts}
        assert "a" * 64 in loaded_hashes
        assert "b" * 64 in loaded_hashes


class TestBackwardCompatNoIndex:
    def test_load_works_without_artifacts_index_file(
        self, tmp_path: Path,
    ) -> None:
        """Un run legacy persisté avant S41 (sans artifacts_index.jsonl)
        reste chargeable — les pipeline_results portent alors leurs
        artefacts directement (cas legacy)."""
        # Simule un run persisté à l'ancienne : pipeline_results
        # contient artifacts inline, pas de artifacts_index.jsonl.
        manifest = {
            "run_id": "legacy",
            "corpus_name": "demo",
            "n_documents": 1,
            "pipeline_names": ["ocr_only"],
            "view_specs": [],
            "code_version": "0.9.0-pre-s41",
            "started_at": "2026-05-06T10:00:00Z",
            "completed_at": "2026-05-06T10:01:00Z",
            "dependencies_lock": {},
            "metadata": {},
        }
        (tmp_path / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )

        legacy_pipeline_record = {
            "document_id": "doc01",
            "pipeline_name": "ocr_only",
            "step_results": [
                {
                    "step_id": "ocr",
                    "succeeded": True,
                    "duration_seconds": 0.5,
                    "produced_artifacts": {"raw_text": "doc01:ocr_only:raw_text"},
                    "error": None,
                },
            ],
            "succeeded": True,
            "duration_seconds": 0.5,
            "artifacts": [
                {
                    "id": "doc01:ocr_only:raw_text",
                    "document_id": "doc01",
                    "type": "raw_text",
                    "content_hash": "b" * 64,
                    "produced_by_step": "ocr",
                    "provenance": None,
                    "uri": None,
                },
            ],
        }
        (tmp_path / "pipeline_results.jsonl").write_text(
            json.dumps(legacy_pipeline_record) + "\n",
            encoding="utf-8",
        )
        (tmp_path / "view_results.jsonl").write_text(
            "", encoding="utf-8",
        )
        # Pas de artifacts_index.jsonl — legacy.

        loaded = HtmlReportRenderer.load_run_result(tmp_path)
        loaded_pr = loaded.document_results[0].pipeline_results[0]
        assert len(loaded_pr.artifacts) == 1
        assert loaded_pr.artifacts[0].content_hash == "b" * 64
