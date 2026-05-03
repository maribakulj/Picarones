"""Sprint A14-S5 — protocoles ``Projector`` et ``EvaluationViewExecutor``.

Vérifie qu'on peut implémenter une classe satisfaisant chaque
protocole sans erreur de typage runtime, et que ``ViewResult`` /
``ProjectionReport`` sont sérialisables JSON.

Pas de test sur l'exécuteur réel — c'est S13.  Ici on valide
seulement les contrats.
"""

from __future__ import annotations

import pytest

from picarones.domain import Artifact, ArtifactType, EvaluationView
from picarones.evaluation.projectors import ProjectionReport, Projector
from picarones.evaluation.views import EvaluationViewExecutor, ViewResult


# ──────────────────────────────────────────────────────────────────────
# ProjectionReport
# ──────────────────────────────────────────────────────────────────────


class TestProjectionReport:
    def test_minimal_report(self) -> None:
        r = ProjectionReport(
            source_artifact_id="a:b:c",
            source_type=ArtifactType.ALTO_XML,
            target_type=ArtifactType.RAW_TEXT,
            projector_name="alto_to_text",
        )
        assert r.lossy is True  # défaut
        assert r.ignored_dimensions == ()

    def test_with_ignored_dimensions(self) -> None:
        r = ProjectionReport(
            source_artifact_id="x",
            source_type=ArtifactType.ALTO_XML,
            target_type=ArtifactType.RAW_TEXT,
            projector_name="alto_to_text",
            lossy=True,
            ignored_dimensions=("geometry", "block_structure"),
            warnings=("ordre de lecture deviné",),
        )
        assert "geometry" in r.ignored_dimensions

    def test_identity_projection_not_lossy(self) -> None:
        r = ProjectionReport(
            source_artifact_id="x",
            source_type=ArtifactType.RAW_TEXT,
            target_type=ArtifactType.RAW_TEXT,
            projector_name="identity",
            lossy=False,
        )
        assert r.lossy is False

    def test_frozen(self) -> None:
        r = ProjectionReport(
            source_artifact_id="x",
            source_type=ArtifactType.RAW_TEXT,
            target_type=ArtifactType.RAW_TEXT,
            projector_name="identity",
        )
        with pytest.raises(Exception):
            r.lossy = False  # type: ignore[misc]

    def test_json_roundtrip(self) -> None:
        r = ProjectionReport(
            source_artifact_id="x",
            source_type=ArtifactType.ALTO_XML,
            target_type=ArtifactType.RAW_TEXT,
            projector_name="alto_to_text",
            ignored_dimensions=("geometry",),
            warnings=("w",),
        )
        r2 = ProjectionReport.model_validate_json(r.model_dump_json())
        assert r == r2


# ──────────────────────────────────────────────────────────────────────
# Projector — protocole satisfait par une classe minimale
# ──────────────────────────────────────────────────────────────────────


class _StubProjector:
    """Minimum pour satisfaire ``Projector``."""

    name = "stub_alto_to_text"
    source_type = ArtifactType.ALTO_XML
    target_type = ArtifactType.RAW_TEXT

    def project(
        self,
        artifact: Artifact,
        params: dict[str, str | int | float | bool],
    ) -> tuple[Artifact, ProjectionReport]:
        target = Artifact(
            id=artifact.id + ":projected",
            document_id=artifact.document_id,
            type=self.target_type,
        )
        report = ProjectionReport(
            source_artifact_id=artifact.id,
            source_type=self.source_type,
            target_type=self.target_type,
            projector_name=self.name,
        )
        return target, report


class TestProjectorProtocol:
    def test_stub_satisfies_protocol(self) -> None:
        p = _StubProjector()
        assert isinstance(p, Projector)

    def test_stub_can_project(self) -> None:
        src = Artifact(
            id="d1:ocr:alto",
            document_id="d1",
            type=ArtifactType.ALTO_XML,
        )
        tgt, report = _StubProjector().project(src, {})
        assert tgt.type == ArtifactType.RAW_TEXT
        assert report.source_artifact_id == "d1:ocr:alto"

    def test_non_conforming_object_does_not_satisfy(self) -> None:
        class _NotAProjector:
            pass
        assert not isinstance(_NotAProjector(), Projector)


# ──────────────────────────────────────────────────────────────────────
# ViewResult
# ──────────────────────────────────────────────────────────────────────


class TestViewResult:
    def test_minimal_result(self) -> None:
        r = ViewResult(
            view_name="text_final",
            candidate_artifact_id="d1:ocr:raw_text",
            ground_truth_artifact_id="d1:gt:raw_text",
        )
        assert r.metric_values == {}
        assert r.failed_metrics == {}
        assert r.projection_report is None

    def test_with_metrics_and_failures(self) -> None:
        r = ViewResult(
            view_name="text_final",
            candidate_artifact_id="x",
            ground_truth_artifact_id="y",
            metric_values={"cer": 0.05, "wer": 0.12},
            failed_metrics={"mufi_coverage": "GT vide, métrique inapplicable"},
            warnings=("normalisation diplomatique appliquée",),
        )
        assert r.metric_values["cer"] == 0.05
        assert "mufi_coverage" in r.failed_metrics

    def test_with_projection_report(self) -> None:
        report = ProjectionReport(
            source_artifact_id="src",
            source_type=ArtifactType.ALTO_XML,
            target_type=ArtifactType.RAW_TEXT,
            projector_name="alto_to_text",
        )
        r = ViewResult(
            view_name="text_final",
            candidate_artifact_id="src",
            ground_truth_artifact_id="gt",
            projection_report=report,
            ignored_dimensions=("geometry",),
        )
        assert r.projection_report is not None
        assert r.projection_report.projector_name == "alto_to_text"

    def test_frozen(self) -> None:
        r = ViewResult(
            view_name="x",
            candidate_artifact_id="a",
            ground_truth_artifact_id="b",
        )
        with pytest.raises(Exception):
            r.view_name = "y"  # type: ignore[misc]

    def test_json_roundtrip(self) -> None:
        r = ViewResult(
            view_name="text_final",
            candidate_artifact_id="x",
            ground_truth_artifact_id="y",
            metric_values={"cer": 0.05},
            failed_metrics={"wer": "boom"},
            warnings=("w",),
            ignored_dimensions=("geometry",),
        )
        r2 = ViewResult.model_validate_json(r.model_dump_json())
        assert r == r2


# ──────────────────────────────────────────────────────────────────────
# EvaluationViewExecutor — protocole satisfait par un stub minimal
# ──────────────────────────────────────────────────────────────────────


class _StubExecutor:
    """Implémentation triviale de ``EvaluationViewExecutor``.

    Ne fait aucun calcul réel — sert à vérifier qu'on peut écrire
    une classe satisfaisant le protocole.  Le vrai exécuteur arrive
    au S13.
    """

    def evaluate(
        self,
        view: EvaluationView,
        candidate: Artifact,
        ground_truth: Artifact,
    ) -> ViewResult:
        return ViewResult(
            view_name=view.name,
            candidate_artifact_id=candidate.id,
            ground_truth_artifact_id=ground_truth.id,
        )


class TestEvaluationViewExecutorProtocol:
    def test_stub_satisfies_protocol(self) -> None:
        ex = _StubExecutor()
        assert isinstance(ex, EvaluationViewExecutor)

    def test_stub_evaluate_returns_view_result(self) -> None:
        view = EvaluationView(
            name="text_final",
            candidate_types=frozenset({ArtifactType.RAW_TEXT}),
        )
        cand = Artifact(id="c", document_id="d", type=ArtifactType.RAW_TEXT)
        gt = Artifact(id="g", document_id="d", type=ArtifactType.RAW_TEXT)
        result = _StubExecutor().evaluate(view, cand, gt)
        assert result.view_name == "text_final"
        assert result.candidate_artifact_id == "c"
