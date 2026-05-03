"""Sprint A14-S6 — ``PipelineStep``, ``PipelineSpec`` (déclaratifs)."""

from __future__ import annotations

import pytest

from picarones.domain import ArtifactType, PicaronesError
from picarones.pipeline import INITIAL_STEP_ID, PipelineSpec, PipelineStep


# ──────────────────────────────────────────────────────────────────────
# PipelineStep — validation des id et champs
# ──────────────────────────────────────────────────────────────────────


class TestPipelineStep:
    def test_minimal_step(self) -> None:
        s = PipelineStep(
            id="ocr",
            kind="ocr",
            adapter_name="tesseract",
            input_types=(ArtifactType.IMAGE,),
            output_types=(ArtifactType.RAW_TEXT,),
        )
        assert s.id == "ocr"
        assert s.params == {}
        assert s.inputs_from == {}

    def test_step_with_inputs_from(self) -> None:
        s = PipelineStep(
            id="correction",
            kind="post_correction",
            adapter_name="openai:gpt-4o",
            input_types=(ArtifactType.RAW_TEXT,),
            output_types=(ArtifactType.CORRECTED_TEXT,),
            inputs_from={ArtifactType.RAW_TEXT: "ocr"},
        )
        assert s.inputs_from[ArtifactType.RAW_TEXT] == "ocr"

    def test_step_with_params(self) -> None:
        s = PipelineStep(
            id="ocr",
            kind="ocr",
            adapter_name="tesseract",
            params={"lang": "fra", "psm": 6, "preserve_interword_spaces": True},
        )
        assert s.params["lang"] == "fra"
        assert s.params["psm"] == 6

    def test_id_validation_rejects_space(self) -> None:
        with pytest.raises(PicaronesError, match="step id invalide"):
            PipelineStep(id="bad id", kind="x", adapter_name="y")

    def test_id_validation_rejects_dot(self) -> None:
        with pytest.raises(PicaronesError, match="step id invalide"):
            PipelineStep(id="bad.id", kind="x", adapter_name="y")

    def test_id_validation_rejects_initial_sentinel(self) -> None:
        """``__initial__`` est réservé pour désigner les entrées
        initiales du runner — un step ne peut pas porter ce nom."""
        with pytest.raises(PicaronesError, match="réservé"):
            PipelineStep(id=INITIAL_STEP_ID, kind="x", adapter_name="y")

    def test_id_accepts_alphanum_underscore_dash(self) -> None:
        s = PipelineStep(id="step_1-final", kind="x", adapter_name="y")
        assert s.id == "step_1-final"

    def test_frozen(self) -> None:
        s = PipelineStep(id="a", kind="b", adapter_name="c")
        with pytest.raises(Exception):
            s.id = "d"  # type: ignore[misc]

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(Exception):
            PipelineStep(  # type: ignore[call-arg]
                id="a", kind="b", adapter_name="c", bogus=42,
            )


# ──────────────────────────────────────────────────────────────────────
# PipelineSpec
# ──────────────────────────────────────────────────────────────────────


class TestPipelineSpec:
    def test_minimal_spec(self) -> None:
        s = PipelineSpec(name="empty")
        assert s.name == "empty"
        assert s.steps == ()
        assert s.initial_inputs == ()

    def test_spec_with_steps(self) -> None:
        s = PipelineSpec(
            name="ocr_only",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(
                PipelineStep(
                    id="ocr",
                    kind="ocr",
                    adapter_name="tesseract",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(ArtifactType.RAW_TEXT,),
                ),
            ),
        )
        assert len(s.steps) == 1
        assert s.step_by_id("ocr") is not None
        assert s.step_by_id("missing") is None

    def test_frozen(self) -> None:
        s = PipelineSpec(name="x")
        with pytest.raises(Exception):
            s.name = "y"  # type: ignore[misc]
