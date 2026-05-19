"""Sprint A14-S39 — RunSpec étendu (inputs_from + preferred_text_output).

Tests des nouvelles fonctionnalités YAML introduites au S39 :

- ``StepSpec.inputs_from`` : DAG branchant via mapping symbolique
  ``ArtifactType → step_id``.
- ``PipelineSpecYaml.preferred_text_output`` : référence symbolique
  ``step_id.output_type`` pour désigner la sortie préférée.

Les tests existants S24 ne sont pas modifiés — l'extension est
purement additive.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from picarones.app.schemas.run_spec import (
    PipelineSpecYaml,
    RunSpec,
    RunSpecLoadError,
    StepSpec,
    load_run_spec_from_yaml,
)
from picarones.domain.artifacts import ArtifactType
from picarones.domain.pipeline_spec import INITIAL_STEP_ID


# ──────────────────────────────────────────────────────────────────────
# StepSpec.inputs_from
# ──────────────────────────────────────────────────────────────────────


class TestStepSpecInputsFrom:
    def test_default_empty(self) -> None:
        step = StepSpec(
            id="ocr",
            adapter_class="my.AdapterClass",
            input_types=(ArtifactType.IMAGE,),
            output_types=(ArtifactType.RAW_TEXT,),
        )
        assert step.inputs_from == {}

    def test_explicit_inputs_from(self) -> None:
        step = StepSpec(
            id="corrector",
            adapter_class="my.LLM",
            input_types=(ArtifactType.RAW_TEXT,),
            output_types=(ArtifactType.CORRECTED_TEXT,),
            inputs_from={ArtifactType.RAW_TEXT: "ocr"},
        )
        assert step.inputs_from[ArtifactType.RAW_TEXT] == "ocr"


# ──────────────────────────────────────────────────────────────────────
# PipelineSpecYaml.preferred_text_output
# ──────────────────────────────────────────────────────────────────────


class TestPreferredTextOutput:
    def test_none_by_default(self) -> None:
        pipe = PipelineSpecYaml(
            name="basic",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(StepSpec(
                id="ocr",
                adapter_class="my.A",
                input_types=(ArtifactType.IMAGE,),
                output_types=(ArtifactType.RAW_TEXT,),
            ),),
        )
        assert pipe.preferred_text_output is None

    def test_valid_reference_accepted(self) -> None:
        pipe = PipelineSpecYaml(
            name="ocr_then_correct",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(
                StepSpec(
                    id="ocr",
                    adapter_class="my.OCR",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(ArtifactType.RAW_TEXT,),
                ),
                StepSpec(
                    id="corrector",
                    adapter_class="my.LLM",
                    input_types=(ArtifactType.RAW_TEXT,),
                    output_types=(ArtifactType.CORRECTED_TEXT,),
                ),
            ),
            preferred_text_output="corrector.corrected_text",
        )
        assert pipe.preferred_text_output == "corrector.corrected_text"

    def test_rejects_missing_dot(self) -> None:
        with pytest.raises(ValidationError, match="format"):
            PipelineSpecYaml(
                name="bad",
                initial_inputs=(ArtifactType.IMAGE,),
                steps=(StepSpec(
                    id="ocr",
                    adapter_class="my.A",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(ArtifactType.RAW_TEXT,),
                ),),
                preferred_text_output="just_a_step_id",
            )

    def test_rejects_unknown_step(self) -> None:
        with pytest.raises(ValidationError, match="introuvable"):
            PipelineSpecYaml(
                name="bad",
                initial_inputs=(ArtifactType.IMAGE,),
                steps=(StepSpec(
                    id="ocr",
                    adapter_class="my.A",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(ArtifactType.RAW_TEXT,),
                ),),
                preferred_text_output="missing_step.raw_text",
            )

    def test_rejects_step_not_producing_type(self) -> None:
        with pytest.raises(ValidationError, match="ne produit pas"):
            PipelineSpecYaml(
                name="bad",
                initial_inputs=(ArtifactType.IMAGE,),
                steps=(StepSpec(
                    id="ocr",
                    adapter_class="my.A",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(ArtifactType.RAW_TEXT,),
                ),),
                # ocr ne produit pas alto_xml — devrait lever
                preferred_text_output="ocr.alto_xml",
            )

    def test_rejects_unknown_artifact_type(self) -> None:
        with pytest.raises(ValidationError, match="output_type"):
            PipelineSpecYaml(
                name="bad",
                initial_inputs=(ArtifactType.IMAGE,),
                steps=(StepSpec(
                    id="ocr",
                    adapter_class="my.A",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(ArtifactType.RAW_TEXT,),
                ),),
                preferred_text_output="ocr.totally_unknown_type",
            )


# ──────────────────────────────────────────────────────────────────────
# Validation inputs_from au niveau pipeline
# ──────────────────────────────────────────────────────────────────────


class TestInputsFromValidation:
    def test_initial_step_id_valid(self) -> None:
        # `__initial__` doit être valide quand le type est bien dans initial_inputs.
        PipelineSpecYaml(
            name="ok",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(StepSpec(
                id="ocr",
                adapter_class="my.A",
                input_types=(ArtifactType.IMAGE,),
                output_types=(ArtifactType.RAW_TEXT,),
                inputs_from={ArtifactType.IMAGE: INITIAL_STEP_ID},
            ),),
        )

    def test_initial_step_id_rejects_unknown_initial_input(self) -> None:
        # `__initial__` mais le type n'est pas dans initial_inputs → erreur.
        with pytest.raises(ValidationError, match="initial_inputs"):
            PipelineSpecYaml(
                name="bad",
                initial_inputs=(ArtifactType.IMAGE,),
                steps=(StepSpec(
                    id="ocr",
                    adapter_class="my.A",
                    input_types=(ArtifactType.IMAGE, ArtifactType.RAW_TEXT),
                    output_types=(ArtifactType.RAW_TEXT,),
                    # raw_text n'est pas dans initial_inputs.
                    inputs_from={ArtifactType.RAW_TEXT: INITIAL_STEP_ID},
                ),),
            )

    def test_explicit_step_reference_valid(self) -> None:
        PipelineSpecYaml(
            name="dag",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(
                StepSpec(
                    id="ocr_a",
                    adapter_class="my.A",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(ArtifactType.RAW_TEXT,),
                ),
                StepSpec(
                    id="ocr_b",
                    adapter_class="my.B",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(ArtifactType.RAW_TEXT,),
                ),
                StepSpec(
                    id="corrector",
                    adapter_class="my.LLM",
                    input_types=(ArtifactType.RAW_TEXT,),
                    output_types=(ArtifactType.CORRECTED_TEXT,),
                    # On choisit explicitement ocr_a (pas ocr_b
                    # qui serait le « dernier producteur »).
                    inputs_from={ArtifactType.RAW_TEXT: "ocr_a"},
                ),
            ),
        )

    def test_rejects_forward_reference(self) -> None:
        # Un step ne peut pas référencer un step en aval de lui.
        with pytest.raises(ValidationError, match="antérieure"):
            PipelineSpecYaml(
                name="bad",
                initial_inputs=(ArtifactType.IMAGE,),
                steps=(
                    StepSpec(
                        id="step1",
                        adapter_class="my.A",
                        input_types=(ArtifactType.IMAGE, ArtifactType.RAW_TEXT),
                        output_types=(ArtifactType.RAW_TEXT,),
                        # Référence step2 qui vient APRÈS — invalide.
                        inputs_from={ArtifactType.RAW_TEXT: "step2"},
                    ),
                    StepSpec(
                        id="step2",
                        adapter_class="my.B",
                        input_types=(ArtifactType.IMAGE,),
                        output_types=(ArtifactType.RAW_TEXT,),
                    ),
                ),
            )

    def test_rejects_step_not_producing_referenced_type(self) -> None:
        with pytest.raises(ValidationError, match="ne produit pas"):
            PipelineSpecYaml(
                name="bad",
                initial_inputs=(ArtifactType.IMAGE,),
                steps=(
                    StepSpec(
                        id="ocr",
                        adapter_class="my.A",
                        input_types=(ArtifactType.IMAGE,),
                        output_types=(ArtifactType.RAW_TEXT,),
                    ),
                    StepSpec(
                        id="alto_remap",
                        adapter_class="my.B",
                        input_types=(ArtifactType.RAW_TEXT, ArtifactType.ALTO_XML),
                        output_types=(ArtifactType.ALTO_XML,),
                        # ocr ne produit pas ALTO_XML mais on le réclame.
                        inputs_from={ArtifactType.ALTO_XML: "ocr"},
                    ),
                ),
            )


# ──────────────────────────────────────────────────────────────────────
# Round-trip YAML
# ──────────────────────────────────────────────────────────────────────


class TestYamlRoundTrip:
    def test_yaml_with_inputs_from_loads_correctly(self) -> None:
        yaml_text = """
corpus_dir: /tmp/corpus
output_dir: /tmp/out
pipelines:
  - name: ocr_then_correct
    initial_inputs: [image]
    preferred_text_output: corrector.corrected_text
    steps:
      - id: ocr
        adapter_class: my_pkg.OCR
        input_types: [image]
        output_types: [raw_text]
      - id: corrector
        adapter_class: my_pkg.LLM
        input_types: [raw_text]
        output_types: [corrected_text]
        inputs_from:
          raw_text: ocr
views: [text_final]
        """.strip()
        spec = load_run_spec_from_yaml(yaml_text)
        assert isinstance(spec, RunSpec)
        assert spec.pipelines[0].preferred_text_output == "corrector.corrected_text"
        corrector = spec.pipelines[0].steps[1]
        assert corrector.inputs_from[ArtifactType.RAW_TEXT] == "ocr"

    def test_yaml_invalid_preferred_text_raises_load_error(self) -> None:
        yaml_text = """
corpus_dir: /tmp/corpus
output_dir: /tmp/out
pipelines:
  - name: ocr
    initial_inputs: [image]
    preferred_text_output: missing_step.raw_text
    steps:
      - id: ocr
        adapter_class: my_pkg.OCR
        input_types: [image]
        output_types: [raw_text]
views: [text_final]
        """.strip()
        with pytest.raises(RunSpecLoadError, match="introuvable"):
            load_run_spec_from_yaml(yaml_text)

    def test_yaml_invalid_inputs_from_raises_load_error(self) -> None:
        yaml_text = """
corpus_dir: /tmp/corpus
output_dir: /tmp/out
pipelines:
  - name: bad
    initial_inputs: [image]
    steps:
      - id: ocr
        adapter_class: my_pkg.OCR
        input_types: [image, raw_text]
        output_types: [raw_text]
        inputs_from:
          raw_text: __initial__
views: [text_final]
        """.strip()
        # raw_text n'est pas dans initial_inputs → erreur.
        with pytest.raises(RunSpecLoadError, match="initial_inputs"):
            load_run_spec_from_yaml(yaml_text)
