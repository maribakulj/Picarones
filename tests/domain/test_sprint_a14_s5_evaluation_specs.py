"""Sprint A14-S5 — contrats déclaratifs des vues d'évaluation.

Tests de ``MetricSpec``, ``EvaluationView``, ``EvaluationSpec``,
``ProjectionSpec``.  Pas de logique métier — juste les invariants
des dataclasses pydantic.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from picarones.domain import (
    ArtifactType,
    EvaluationSpec,
    EvaluationView,
    MetricSpec,
    ProjectionSpec,
)


# ──────────────────────────────────────────────────────────────────────
# MetricSpec
# ──────────────────────────────────────────────────────────────────────


class TestMetricSpec:
    def test_minimal_spec(self) -> None:
        spec = MetricSpec(
            name="cer",
            input_types=(ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT),
        )
        assert spec.name == "cer"
        assert spec.description == ""
        assert spec.higher_is_better is False
        assert spec.tags == frozenset()

    def test_higher_is_better_for_quality_metrics(self) -> None:
        spec = MetricSpec(
            name="ner_f1",
            input_types=(ArtifactType.ENTITIES, ArtifactType.ENTITIES),
            description="F1 micro sur entités nommées",
            higher_is_better=True,
            tags=frozenset({"ner", "icdar"}),
        )
        assert spec.higher_is_better is True
        assert "ner" in spec.tags

    def test_frozen(self) -> None:
        spec = MetricSpec(
            name="cer",
            input_types=(ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT),
        )
        with pytest.raises(ValidationError):
            spec.name = "wer"  # type: ignore[misc]

    def test_no_callable_field(self) -> None:
        """Différence avec l'ancien core.metric_registry.MetricSpec :
        pas de ``func`` ici (le callable vit dans MetricRegistry)."""
        spec = MetricSpec(
            name="cer",
            input_types=(ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT),
        )
        assert not hasattr(spec, "func")

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MetricSpec(  # type: ignore[call-arg]
                name="cer",
                input_types=(ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT),
                bogus=42,
            )


# ──────────────────────────────────────────────────────────────────────
# ProjectionSpec
# ──────────────────────────────────────────────────────────────────────


class TestProjectionSpec:
    def test_alto_to_text(self) -> None:
        p = ProjectionSpec(
            source_type=ArtifactType.ALTO_XML,
            target_type=ArtifactType.RAW_TEXT,
            projector_name="alto_to_text",
        )
        assert p.source_type == ArtifactType.ALTO_XML
        assert p.target_type == ArtifactType.RAW_TEXT
        assert p.params == {}
        assert p.is_identity is False

    def test_identity_projection(self) -> None:
        p = ProjectionSpec(
            source_type=ArtifactType.RAW_TEXT,
            target_type=ArtifactType.RAW_TEXT,
            projector_name="identity",
        )
        assert p.is_identity is True

    def test_with_params(self) -> None:
        p = ProjectionSpec(
            source_type=ArtifactType.ALTO_XML,
            target_type=ArtifactType.RAW_TEXT,
            projector_name="alto_to_text",
            params={"reading_order": "natural", "preserve_hyphens": True},
        )
        assert p.params["reading_order"] == "natural"
        assert p.params["preserve_hyphens"] is True

    def test_frozen(self) -> None:
        p = ProjectionSpec(
            source_type=ArtifactType.ALTO_XML,
            target_type=ArtifactType.RAW_TEXT,
            projector_name="alto_to_text",
        )
        with pytest.raises(ValidationError):
            p.projector_name = "other"  # type: ignore[misc]

    def test_json_roundtrip(self) -> None:
        p = ProjectionSpec(
            source_type=ArtifactType.ALTO_XML,
            target_type=ArtifactType.RAW_TEXT,
            projector_name="alto_to_text",
            params={"reading_order": "natural"},
        )
        p2 = ProjectionSpec.model_validate_json(p.model_dump_json())
        assert p == p2


# ──────────────────────────────────────────────────────────────────────
# EvaluationView — la pièce centrale du S5
# ──────────────────────────────────────────────────────────────────────


class TestEvaluationView:
    def test_text_final_view_canonical_shape(self) -> None:
        """Définition de done du S5 : tu peux instancier
        ``EvaluationView(name="text_final", projection_spec=..., metrics=...)``
        sans rien d'autre."""
        view = EvaluationView(
            name="text_final",
            description="Compare les sorties textuelles finales.",
            candidate_types=frozenset({
                ArtifactType.RAW_TEXT,
                ArtifactType.CORRECTED_TEXT,
                ArtifactType.ALTO_XML,
            }),
            projection=ProjectionSpec(
                source_type=ArtifactType.ALTO_XML,
                target_type=ArtifactType.RAW_TEXT,
                projector_name="alto_to_text",
            ),
            metric_names=("cer", "wer"),
            ignored_dimensions=("geometry", "block_structure"),
            warnings=("Cette vue ignore la structure spatiale.",),
        )
        assert view.name == "text_final"
        assert view.accepts(ArtifactType.RAW_TEXT)
        assert view.accepts(ArtifactType.ALTO_XML)
        assert not view.accepts(ArtifactType.IMAGE)

    def test_alto_view_no_projection(self) -> None:
        """Une vue qui n'a pas besoin de projection (compare l'ALTO
        tel quel)."""
        view = EvaluationView(
            name="alto_documentary",
            candidate_types=frozenset({ArtifactType.ALTO_XML}),
            projection=None,
            metric_names=("alto_validity", "line_alignment_f1"),
        )
        assert view.projection is None

    def test_search_view_text_only(self) -> None:
        view = EvaluationView(
            name="searchability",
            candidate_types=frozenset({
                ArtifactType.RAW_TEXT, ArtifactType.CORRECTED_TEXT,
            }),
            metric_names=("rare_token_recall", "numerical_sequences"),
        )
        assert view.accepts(ArtifactType.RAW_TEXT)
        assert not view.accepts(ArtifactType.ALTO_XML)

    def test_view_with_normalization_profile(self) -> None:
        view = EvaluationView(
            name="text_diplomatic",
            candidate_types=frozenset({ArtifactType.RAW_TEXT}),
            normalization_profile="medieval_french",
            metric_names=("cer",),
        )
        assert view.normalization_profile == "medieval_french"

    def test_empty_candidate_types_is_valid_but_useless(self) -> None:
        """Pas de validation à la construction : un caller peut
        construire une vue inutile (qui n'accepte rien) ; à
        l'EvaluationViewExecutor de la signaler runtime."""
        view = EvaluationView(
            name="useless",
            candidate_types=frozenset(),
        )
        assert not view.accepts(ArtifactType.RAW_TEXT)

    def test_frozen(self) -> None:
        view = EvaluationView(
            name="x",
            candidate_types=frozenset({ArtifactType.RAW_TEXT}),
        )
        with pytest.raises(ValidationError):
            view.name = "y"  # type: ignore[misc]

    def test_json_roundtrip(self) -> None:
        view = EvaluationView(
            name="text_final",
            description="x",
            candidate_types=frozenset({ArtifactType.RAW_TEXT}),
            projection=ProjectionSpec(
                source_type=ArtifactType.ALTO_XML,
                target_type=ArtifactType.RAW_TEXT,
                projector_name="alto_to_text",
            ),
            normalization_profile="nfc",
            metric_names=("cer",),
            ignored_dimensions=("geometry",),
            warnings=("avertissement",),
        )
        v2 = EvaluationView.model_validate_json(view.model_dump_json())
        assert view == v2


# ──────────────────────────────────────────────────────────────────────
# EvaluationSpec
# ──────────────────────────────────────────────────────────────────────


class TestEvaluationSpec:
    def test_empty_spec(self) -> None:
        s = EvaluationSpec()
        assert s.views == ()

    def test_multi_view_spec(self) -> None:
        s = EvaluationSpec(
            views=(
                EvaluationView(
                    name="text",
                    candidate_types=frozenset({ArtifactType.RAW_TEXT}),
                ),
                EvaluationView(
                    name="alto",
                    candidate_types=frozenset({ArtifactType.ALTO_XML}),
                ),
            ),
        )
        assert len(s.views) == 2
        assert s.view_by_name("text") is not None
        assert s.view_by_name("alto") is not None
        assert s.view_by_name("missing") is None

    def test_frozen(self) -> None:
        s = EvaluationSpec()
        with pytest.raises(ValidationError):
            s.views = ()  # type: ignore[misc]
