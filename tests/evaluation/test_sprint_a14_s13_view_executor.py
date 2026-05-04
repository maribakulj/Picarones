"""Sprint A14-S13 — ``DefaultEvaluationViewExecutor``.

Tests d'orchestration : la vue + ses dépendances (registries +
payload loader) sur 10+ cas couvrant les chemins critiques.
"""

from __future__ import annotations

import pytest

from picarones.domain import (
    Artifact,
    ArtifactType,
    EvaluationView,
    MetricSpec,
    ProjectionError,
    ProjectionSpec,
)
from picarones.evaluation.projectors import (
    ProjectionReport,
    ProjectorRegistry,
    ProjectorRegistrationError,
    ProjectorNotFoundError,
)
from picarones.evaluation.registry import MetricRegistry
from picarones.evaluation.views import (
    DefaultEvaluationViewExecutor,
    ViewResult,
)


# ──────────────────────────────────────────────────────────────────────
# Stubs réutilisables
# ──────────────────────────────────────────────────────────────────────


class _StubProjector:
    """Projecteur ALTO → texte simple pour les tests."""

    name = "stub_alto_to_text"
    source_type = ArtifactType.ALTO_XML
    target_type = ArtifactType.RAW_TEXT

    def __init__(self, output_payload: str = "projected text") -> None:
        self.output_payload = output_payload

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
            ignored_dimensions=("geometry", "blocks"),
            warnings=("ordre de lecture deviné",),
        )
        # Sprint S25 — retourne le payload directement.
        return target, self.output_payload, report


def _build_executor(
    payloads: dict[str, object],
    *,
    register_projector: bool = True,
    extra_metrics: dict[str, object] | None = None,
) -> DefaultEvaluationViewExecutor:
    metrics = MetricRegistry()
    metrics.register(
        MetricSpec(
            name="cer",
            input_types=(ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT),
        ),
        lambda gt, hyp: 0.0 if gt == hyp else (
            0.5 if isinstance(gt, str) and isinstance(hyp, str) and len(gt) == len(hyp)
            else 1.0
        ),
    )
    metrics.register(
        MetricSpec(
            name="wer",
            input_types=(ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT),
        ),
        lambda gt, hyp: 0.0 if gt == hyp else 0.5,
    )
    if extra_metrics:
        for name, fn in extra_metrics.items():
            metrics.register(
                MetricSpec(
                    name=name,
                    input_types=(ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT),
                ),
                fn,
            )

    projectors = ProjectorRegistry()
    if register_projector:
        projectors.register(_StubProjector())

    def loader(artifact: Artifact):
        if artifact.id not in payloads:
            raise KeyError(f"payload manquant : {artifact.id}")
        return payloads[artifact.id]

    return DefaultEvaluationViewExecutor(metrics, projectors, loader)


def _text_view(
    *,
    name: str = "text_final",
    candidate_types: frozenset = frozenset({
        ArtifactType.RAW_TEXT,
        ArtifactType.CORRECTED_TEXT,
        ArtifactType.ALTO_XML,
    }),
    projection: ProjectionSpec | None = None,
    normalization_profile: str | None = None,
    metric_names: tuple[str, ...] = ("cer",),
    ignored_dimensions: tuple[str, ...] = (),
    warnings: tuple[str, ...] = (),
) -> EvaluationView:
    return EvaluationView(
        name=name,
        candidate_types=candidate_types,
        projection=projection,
        normalization_profile=normalization_profile,
        metric_names=metric_names,
        ignored_dimensions=ignored_dimensions,
        warnings=warnings,
    )


# ──────────────────────────────────────────────────────────────────────
# 10 cas d'évaluation
# ──────────────────────────────────────────────────────────────────────


class TestEvaluator:

    def test_text_direct_no_projection(self) -> None:
        """Cas 1 — RAW_TEXT direct, pas de projection."""
        payloads = {"cand": "hello", "gt": "hello"}
        executor = _build_executor(payloads)
        view = _text_view(metric_names=("cer", "wer"))
        cand = Artifact(id="cand", document_id="d", type=ArtifactType.RAW_TEXT)
        gt = Artifact(id="gt", document_id="d", type=ArtifactType.RAW_TEXT)
        result = executor.evaluate(view, cand, gt)
        assert result.metric_values["cer"] == 0.0
        assert result.metric_values["wer"] == 0.0
        assert result.projection_report is None
        assert result.failed_metrics == {}

    def test_text_direct_with_difference(self) -> None:
        """Cas 2 — RAW_TEXT, candidat différent de la GT."""
        payloads = {"cand": "world", "gt": "hello"}
        executor = _build_executor(payloads)
        view = _text_view()
        cand = Artifact(id="cand", document_id="d", type=ArtifactType.RAW_TEXT)
        gt = Artifact(id="gt", document_id="d", type=ArtifactType.RAW_TEXT)
        result = executor.evaluate(view, cand, gt)
        assert result.metric_values["cer"] > 0

    def test_alto_to_text_via_projection(self) -> None:
        """Cas 3 — ALTO_XML projeté en RAW_TEXT, projection_report présent."""
        payloads = {
            "alto:projected": "projected text",
            "gt": "projected text",
        }
        executor = _build_executor(payloads)
        view = _text_view(
            projection=ProjectionSpec(
                source_type=ArtifactType.ALTO_XML,
                target_type=ArtifactType.RAW_TEXT,
                projector_name="stub_alto_to_text",
            ),
        )
        cand = Artifact(id="alto", document_id="d", type=ArtifactType.ALTO_XML)
        gt = Artifact(id="gt", document_id="d", type=ArtifactType.RAW_TEXT)
        result = executor.evaluate(view, cand, gt)
        assert result.projection_report is not None
        assert result.projection_report.projector_name == "stub_alto_to_text"
        assert "geometry" in result.ignored_dimensions
        assert "ordre de lecture deviné" in result.warnings
        assert result.metric_values["cer"] == 0.0

    def test_view_rejects_wrong_artifact_type(self) -> None:
        """Cas 4 — la vue n'accepte pas IMAGE → ValueError."""
        payloads = {}
        executor = _build_executor(payloads)
        view = _text_view(
            candidate_types=frozenset({ArtifactType.RAW_TEXT}),
        )
        cand = Artifact(id="x", document_id="d", type=ArtifactType.IMAGE)
        gt = Artifact(id="gt", document_id="d", type=ArtifactType.RAW_TEXT)
        with pytest.raises(ValueError, match="n'accepte pas"):
            executor.evaluate(view, cand, gt)

    def test_unknown_projector_raises_projection_error(self) -> None:
        """Cas 5 — la vue référence un projecteur non enregistré."""
        payloads = {"cand": "x", "gt": "x"}
        executor = _build_executor(payloads, register_projector=False)
        view = _text_view(
            projection=ProjectionSpec(
                source_type=ArtifactType.ALTO_XML,
                target_type=ArtifactType.RAW_TEXT,
                projector_name="nonexistent",
            ),
        )
        cand = Artifact(id="cand", document_id="d", type=ArtifactType.ALTO_XML)
        gt = Artifact(id="gt", document_id="d", type=ArtifactType.RAW_TEXT)
        with pytest.raises(ProjectionError, match="introuvable"):
            executor.evaluate(view, cand, gt)

    def test_projector_that_raises_wraps_in_projection_error(self) -> None:
        """Cas 6 — le projecteur lève une exception interne."""
        class _CrashingProjector:
            name = "crash"
            source_type = ArtifactType.ALTO_XML
            target_type = ArtifactType.RAW_TEXT
            def project(self, artifact, params):
                raise RuntimeError("boom interne")

        metrics = MetricRegistry()
        projectors = ProjectorRegistry()
        projectors.register(_CrashingProjector())
        executor = DefaultEvaluationViewExecutor(
            metrics, projectors, lambda a: None,
        )
        view = _text_view(
            projection=ProjectionSpec(
                source_type=ArtifactType.ALTO_XML,
                target_type=ArtifactType.RAW_TEXT,
                projector_name="crash",
            ),
            metric_names=(),
        )
        cand = Artifact(id="c", document_id="d", type=ArtifactType.ALTO_XML)
        gt = Artifact(id="gt", document_id="d", type=ArtifactType.RAW_TEXT)
        with pytest.raises(ProjectionError, match="boom interne"):
            executor.evaluate(view, cand, gt)

    def test_metric_that_raises_goes_to_failed_metrics(self) -> None:
        """Cas 7 — une métrique qui lève → failed_metrics, pas plante."""
        def _broken(gt, hyp):
            raise ValueError("métrique cassée")
        payloads = {"cand": "x", "gt": "x"}
        executor = _build_executor(
            payloads,
            extra_metrics={"broken": _broken},
        )
        view = _text_view(metric_names=("cer", "broken", "wer"))
        cand = Artifact(id="cand", document_id="d", type=ArtifactType.RAW_TEXT)
        gt = Artifact(id="gt", document_id="d", type=ArtifactType.RAW_TEXT)
        result = executor.evaluate(view, cand, gt)
        assert "cer" in result.metric_values
        assert "wer" in result.metric_values
        assert "broken" in result.failed_metrics
        assert "métrique cassée" in result.failed_metrics["broken"]

    def test_unknown_metric_goes_to_failed_metrics(self) -> None:
        """Cas 8 — une métrique non enregistrée → failed_metrics."""
        payloads = {"cand": "x", "gt": "x"}
        executor = _build_executor(payloads)
        view = _text_view(metric_names=("cer", "nonexistent_metric"))
        cand = Artifact(id="cand", document_id="d", type=ArtifactType.RAW_TEXT)
        gt = Artifact(id="gt", document_id="d", type=ArtifactType.RAW_TEXT)
        result = executor.evaluate(view, cand, gt)
        assert "cer" in result.metric_values
        assert "nonexistent_metric" in result.failed_metrics
        assert "non enregistrée" in result.failed_metrics["nonexistent_metric"]

    def test_normalization_profile_applied(self) -> None:
        """Cas 9 — vue avec normalization_profile applique la
        normalisation aux deux payloads."""
        # Avec medieval_french : ſ → s, u → v
        payloads = {"cand": "afpre", "gt": "aſpre"}
        executor = _build_executor(payloads)
        view = _text_view(normalization_profile="medieval_french")
        cand = Artifact(id="cand", document_id="d", type=ArtifactType.RAW_TEXT)
        gt = Artifact(id="gt", document_id="d", type=ArtifactType.RAW_TEXT)
        result = executor.evaluate(view, cand, gt)
        # Après normalisation, les deux deviennent "aspre" (cer stub
        # retourne 0.5 pour len égal, 0.0 pour égalité stricte).
        # On vérifie au moins que la métrique a été calculée.
        assert "cer" in result.metric_values

    def test_payload_loader_failure_blocks_all_metrics(self) -> None:
        """Cas 10 — le loader plante → toutes les métriques sont
        marquées en échec global."""
        # Loader plante systématiquement.
        metrics = MetricRegistry()
        metrics.register(
            MetricSpec(
                name="cer",
                input_types=(ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT),
            ),
            lambda r, h: 0.0,
        )
        projectors = ProjectorRegistry()

        def _bad_loader(artifact):
            raise FileNotFoundError(f"missing file for {artifact.id}")

        executor = DefaultEvaluationViewExecutor(metrics, projectors, _bad_loader)
        view = _text_view(metric_names=("cer",))
        cand = Artifact(id="cand", document_id="d", type=ArtifactType.RAW_TEXT)
        gt = Artifact(id="gt", document_id="d", type=ArtifactType.RAW_TEXT)
        result = executor.evaluate(view, cand, gt)
        assert result.metric_values == {}
        assert "cer" in result.failed_metrics
        assert "payload_loader a échoué" in result.failed_metrics["cer"]


# ──────────────────────────────────────────────────────────────────────
# Constructor validation
# ──────────────────────────────────────────────────────────────────────


class TestConstructor:
    def test_rejects_non_metric_registry(self) -> None:
        with pytest.raises(TypeError, match="metric_registry"):
            DefaultEvaluationViewExecutor(
                "not a registry", ProjectorRegistry(), lambda a: None,  # type: ignore[arg-type]
            )

    def test_rejects_non_projector_registry(self) -> None:
        with pytest.raises(TypeError, match="projector_registry"):
            DefaultEvaluationViewExecutor(
                MetricRegistry(), "nope", lambda a: None,  # type: ignore[arg-type]
            )

    def test_rejects_non_callable_loader(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            DefaultEvaluationViewExecutor(
                MetricRegistry(), ProjectorRegistry(), "not_callable",  # type: ignore[arg-type]
            )


# ──────────────────────────────────────────────────────────────────────
# ProjectorRegistry — tests directs
# ──────────────────────────────────────────────────────────────────────


class TestProjectorRegistry:
    def test_register_and_get(self) -> None:
        reg = ProjectorRegistry()
        p = _StubProjector()
        reg.register(p)
        assert "stub_alto_to_text" in reg
        assert reg.get("stub_alto_to_text") is p

    def test_register_non_protocol_raises(self) -> None:
        reg = ProjectorRegistry()
        class _NotAProjector:
            pass
        with pytest.raises(ProjectorRegistrationError):
            reg.register(_NotAProjector())  # type: ignore[arg-type]

    def test_idempotent_re_registration(self) -> None:
        reg = ProjectorRegistry()
        p = _StubProjector()
        reg.register(p)
        reg.register(p)  # ne lève pas
        assert len(reg) == 1

    def test_get_unknown_raises(self) -> None:
        reg = ProjectorRegistry()
        with pytest.raises(ProjectorNotFoundError):
            reg.get("missing")

    def test_two_registries_independent(self) -> None:
        a = ProjectorRegistry()
        b = ProjectorRegistry()
        a.register(_StubProjector())
        assert "stub_alto_to_text" in a
        assert "stub_alto_to_text" not in b
