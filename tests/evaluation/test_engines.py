"""Sprint A14-S27 — ``ProjectionEngine`` + ``EvaluationEngine`` séparés.

Tests des deux moteurs introduits par S27 pour découper le S13.
Couvre :

1. ``ProjectionEngine.project`` :
   - cas identité (spec None) → artefact tel quel, payload None,
     report None ;
   - spec identité (source == target) → idem ;
   - projection nominale → triplet complet (artefact target, payload,
     report) ;
   - projecteur introuvable → ProjectionError ;
   - projecteur qui lève → wrappé en ProjectionError ;
   - validation du constructeur (rejette non-registry).

2. ``EvaluationEngine.evaluate`` :
   - calcule chaque métrique, dispatch erreur dans failed_metrics ;
   - métrique inconnue → message explicite ;
   - métrique qui lève → message ``{type}: {msg}`` ;
   - ordre des résultats préservé ;
   - validation du constructeur ;
   - sucre ``evaluate_one`` ;
   - dataclass ``EvaluationResult`` (n_succeeded, n_failed,
     all_succeeded, with_global_failure).

3. Intégration : l'executor refondu (S27) délègue aux deux engines —
   les comportements existants du S13 sont préservés (couverture
   indirecte par ``test_sprint_a14_s13_view_executor.py``).
"""

from __future__ import annotations

import pytest

from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.domain.errors import ProjectionError
from picarones.domain.projection_spec import ProjectionSpec
from picarones.evaluation.evaluation_engine import (
    EvaluationEngine,
    EvaluationResult,
)
from picarones.evaluation.projection_engine import (
    ProjectionEngine,
    ProjectionResult,
)
from picarones.evaluation.projectors.base import ProjectionReport
from picarones.evaluation.projectors.registry import (
    ProjectorRegistry,
)
from picarones.evaluation.registry import MetricRegistry
from picarones.domain.evaluation_spec import MetricSpec


# ──────────────────────────────────────────────────────────────────────
# Stubs réutilisables
# ──────────────────────────────────────────────────────────────────────


class _StubProjector:
    name = "stub"
    source_type = ArtifactType.ALTO_XML
    target_type = ArtifactType.RAW_TEXT

    def __init__(self, payload: str = "projected") -> None:
        self._payload = payload

    def project(self, artifact, params):
        target = Artifact(
            id=f"{artifact.id}:projected",
            document_id=artifact.document_id,
            type=self.target_type,
        )
        report = ProjectionReport(
            source_artifact_id=artifact.id,
            source_type=self.source_type,
            target_type=self.target_type,
            projector_name=self.name,
            lossy=True,
            ignored_dimensions=("geometry",),
            warnings=("dim perdue",),
        )
        return target, self._payload, report


class _CrashingProjector:
    name = "crash"
    source_type = ArtifactType.ALTO_XML
    target_type = ArtifactType.RAW_TEXT

    def project(self, artifact, params):
        raise RuntimeError("boom interne")


# ──────────────────────────────────────────────────────────────────────
# ProjectionEngine
# ──────────────────────────────────────────────────────────────────────


class TestProjectionEngineConstructor:
    def test_rejects_non_registry(self) -> None:
        with pytest.raises(TypeError, match="projector_registry"):
            ProjectionEngine("nope")  # type: ignore[arg-type]

    def test_accepts_empty_registry(self) -> None:
        engine = ProjectionEngine(ProjectorRegistry())
        assert engine.projectors is not None


class TestProjectionEngineIdentity:
    def test_none_spec_returns_unchanged(self) -> None:
        engine = ProjectionEngine(ProjectorRegistry())
        artifact = Artifact(id="a", document_id="d", type=ArtifactType.RAW_TEXT)
        result = engine.project(artifact, None)
        assert result.artifact is artifact
        assert result.payload is None
        assert result.report is None
        assert result.has_projection is False

    def test_identity_spec_returns_unchanged(self) -> None:
        engine = ProjectionEngine(ProjectorRegistry())
        artifact = Artifact(id="a", document_id="d", type=ArtifactType.RAW_TEXT)
        spec = ProjectionSpec(
            source_type=ArtifactType.RAW_TEXT,
            target_type=ArtifactType.RAW_TEXT,
            projector_name="ignored_when_identity",
        )
        result = engine.project(artifact, spec)
        assert result.artifact is artifact
        assert result.payload is None
        assert result.report is None


class TestProjectionEngineNominal:
    def test_nominal_returns_triple(self) -> None:
        registry = ProjectorRegistry()
        registry.register(_StubProjector(payload="hello"))
        engine = ProjectionEngine(registry)
        artifact = Artifact(
            id="alto",
            document_id="d",
            type=ArtifactType.ALTO_XML,
        )
        spec = ProjectionSpec(
            source_type=ArtifactType.ALTO_XML,
            target_type=ArtifactType.RAW_TEXT,
            projector_name="stub",
        )
        result = engine.project(artifact, spec)
        assert result.artifact.type == ArtifactType.RAW_TEXT
        assert result.artifact.id == "alto:projected"
        assert result.payload == "hello"
        assert result.report is not None
        assert result.report.projector_name == "stub"
        assert result.has_projection is True


class TestProjectionEngineErrors:
    def test_unknown_projector_raises_projection_error(self) -> None:
        engine = ProjectionEngine(ProjectorRegistry())
        artifact = Artifact(id="a", document_id="d", type=ArtifactType.ALTO_XML)
        spec = ProjectionSpec(
            source_type=ArtifactType.ALTO_XML,
            target_type=ArtifactType.RAW_TEXT,
            projector_name="missing",
        )
        with pytest.raises(ProjectionError, match="introuvable"):
            engine.project(artifact, spec)

    def test_crashing_projector_wraps_in_projection_error(self) -> None:
        registry = ProjectorRegistry()
        registry.register(_CrashingProjector())
        engine = ProjectionEngine(registry)
        artifact = Artifact(id="a", document_id="d", type=ArtifactType.ALTO_XML)
        spec = ProjectionSpec(
            source_type=ArtifactType.ALTO_XML,
            target_type=ArtifactType.RAW_TEXT,
            projector_name="crash",
        )
        with pytest.raises(ProjectionError, match="boom interne"):
            engine.project(artifact, spec)

    def test_native_projection_error_propagated_unwrapped(self) -> None:
        """Si le projecteur lève déjà un ``ProjectionError``, on ne le
        wrappe pas dans un nouveau (préservation de la sémantique)."""
        class _NativeProjErrProjector:
            name = "native_err"
            source_type = ArtifactType.ALTO_XML
            target_type = ArtifactType.RAW_TEXT

            def project(self, artifact, params):
                raise ProjectionError("erreur native")

        registry = ProjectorRegistry()
        registry.register(_NativeProjErrProjector())
        engine = ProjectionEngine(registry)
        artifact = Artifact(id="a", document_id="d", type=ArtifactType.ALTO_XML)
        spec = ProjectionSpec(
            source_type=ArtifactType.ALTO_XML,
            target_type=ArtifactType.RAW_TEXT,
            projector_name="native_err",
        )
        with pytest.raises(ProjectionError, match="erreur native"):
            engine.project(artifact, spec)


# ──────────────────────────────────────────────────────────────────────
# EvaluationEngine
# ──────────────────────────────────────────────────────────────────────


def _build_metric_registry(extra: dict = None) -> MetricRegistry:
    reg = MetricRegistry()
    reg.register(
        MetricSpec(
            name="cer",
            input_types=(ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT),
        ),
        lambda r, h: 0.0 if r == h else 1.0,
    )
    reg.register(
        MetricSpec(
            name="wer",
            input_types=(ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT),
        ),
        lambda r, h: 0.0 if r == h else 0.5,
    )
    if extra:
        for name, fn in extra.items():
            reg.register(
                MetricSpec(
                    name=name,
                    input_types=(ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT),
                ),
                fn,
            )
    return reg


class TestEvaluationEngineConstructor:
    def test_rejects_non_registry(self) -> None:
        with pytest.raises(TypeError, match="metric_registry"):
            EvaluationEngine("nope")  # type: ignore[arg-type]

    def test_accepts_empty_registry(self) -> None:
        engine = EvaluationEngine(MetricRegistry())
        assert engine.metrics is not None


class TestEvaluationEngineNominal:
    def test_all_metrics_succeed(self) -> None:
        engine = EvaluationEngine(_build_metric_registry())
        result = engine.evaluate(("cer", "wer"), "x", "x")
        assert result.metric_values == {"cer": 0.0, "wer": 0.0}
        assert result.failed_metrics == {}
        assert result.n_succeeded == 2
        assert result.n_failed == 0
        assert result.all_succeeded is True

    def test_metric_returning_nonzero(self) -> None:
        engine = EvaluationEngine(_build_metric_registry())
        result = engine.evaluate(("cer", "wer"), "abc", "xyz")
        assert result.metric_values["cer"] == 1.0
        assert result.metric_values["wer"] == 0.5

    def test_evaluate_one_sugar(self) -> None:
        engine = EvaluationEngine(_build_metric_registry())
        result = engine.evaluate_one("cer", "x", "x")
        assert result.metric_values == {"cer": 0.0}
        assert result.failed_metrics == {}

    def test_order_preserved(self) -> None:
        engine = EvaluationEngine(_build_metric_registry())
        result = engine.evaluate(("wer", "cer"), "x", "x")
        # dict préserve l'ordre d'insertion (Python 3.7+).
        assert list(result.metric_values.keys()) == ["wer", "cer"]


class TestEvaluationEngineFailures:
    def test_unknown_metric_goes_to_failed(self) -> None:
        engine = EvaluationEngine(_build_metric_registry())
        result = engine.evaluate(("cer", "missing"), "x", "x")
        assert "cer" in result.metric_values
        assert "missing" in result.failed_metrics
        assert "non enregistrée" in result.failed_metrics["missing"]

    def test_metric_that_raises_goes_to_failed(self) -> None:
        def _broken(r, h):
            raise ValueError("metric crashed")

        engine = EvaluationEngine(_build_metric_registry({"broken": _broken}))
        result = engine.evaluate(("cer", "broken", "wer"), "x", "x")
        assert "cer" in result.metric_values
        assert "wer" in result.metric_values
        assert "broken" in result.failed_metrics
        assert "ValueError" in result.failed_metrics["broken"]
        assert "metric crashed" in result.failed_metrics["broken"]
        assert result.n_succeeded == 2
        assert result.n_failed == 1
        assert result.all_succeeded is False

    def test_empty_metric_list_returns_empty_result(self) -> None:
        engine = EvaluationEngine(_build_metric_registry())
        result = engine.evaluate((), "x", "x")
        assert result.metric_values == {}
        assert result.failed_metrics == {}
        assert result.all_succeeded is True


class TestEvaluationResultDataclass:
    def test_with_global_failure_marks_all(self) -> None:
        engine = EvaluationEngine(_build_metric_registry())
        result = engine.evaluate(("cer", "wer"), "x", "x")
        failed_all = result.with_global_failure("loader crashed")
        assert failed_all.metric_values == {}
        assert failed_all.failed_metrics == {
            "cer": "loader crashed",
            "wer": "loader crashed",
        }

    def test_dataclass_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        result = EvaluationResult(metric_values={"cer": 0.0})
        with pytest.raises(FrozenInstanceError):
            result.metric_values = {}  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────────────
# ProjectionResult dataclass
# ──────────────────────────────────────────────────────────────────────


class TestProjectionResultDataclass:
    def test_has_projection_property(self) -> None:
        artifact = Artifact(id="a", document_id="d", type=ArtifactType.RAW_TEXT)
        no_proj = ProjectionResult(artifact=artifact, payload=None, report=None)
        assert no_proj.has_projection is False

        report = ProjectionReport(
            source_artifact_id="a",
            source_type=ArtifactType.ALTO_XML,
            target_type=ArtifactType.RAW_TEXT,
            projector_name="x",
        )
        with_proj = ProjectionResult(
            artifact=artifact, payload="text", report=report,
        )
        assert with_proj.has_projection is True

    def test_dataclass_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        artifact = Artifact(id="a", document_id="d", type=ArtifactType.RAW_TEXT)
        result = ProjectionResult(artifact=artifact, payload=None, report=None)
        with pytest.raises(FrozenInstanceError):
            result.payload = "modified"  # type: ignore[misc]
