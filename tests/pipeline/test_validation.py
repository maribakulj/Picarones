"""Sprint A14-S6 — ``validate_spec``.

Couvre les ~12 cas typiques : chaîne valide, type manquant,
adapter inconnu, fork avec ``inputs_from``, références invalides,
DAG vide, IDs dupliqués.

Aucun ``StepExecutor`` instancié — la validation est purement
statique sur la spec.
"""

from __future__ import annotations

from picarones.domain import ArtifactType
from picarones.pipeline import (
    INITIAL_STEP_ID,
    PipelineSpec,
    PipelineStep,
    validate_spec,
)


# ──────────────────────────────────────────────────────────────────────
# Cas valides
# ──────────────────────────────────────────────────────────────────────


class TestValidSpecs:
    def test_simple_ocr_pipeline(self) -> None:
        spec = PipelineSpec(
            name="ocr_only",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(
                PipelineStep(
                    id="ocr", kind="ocr", adapter_name="tesseract",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(ArtifactType.RAW_TEXT,),
                ),
            ),
        )
        assert validate_spec(spec) == []

    def test_ocr_then_llm(self) -> None:
        spec = PipelineSpec(
            name="ocr_llm",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(
                PipelineStep(
                    id="ocr", kind="ocr", adapter_name="tesseract",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(ArtifactType.RAW_TEXT,),
                ),
                PipelineStep(
                    id="correct", kind="post_correction",
                    adapter_name="openai:gpt-4o",
                    input_types=(ArtifactType.RAW_TEXT,),
                    output_types=(ArtifactType.CORRECTED_TEXT,),
                ),
            ),
        )
        assert validate_spec(spec) == []

    def test_def_of_done_tesseract_llm_alto_remap(self) -> None:
        """Définition de done du S6 : valider le YAML cible BnF."""
        spec = PipelineSpec(
            name="tesseract_llm_alto_remap",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(
                PipelineStep(
                    id="ocr", kind="ocr", adapter_name="tesseract",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(ArtifactType.RAW_TEXT, ArtifactType.ALTO_XML),
                ),
                PipelineStep(
                    id="correction", kind="post_correction",
                    adapter_name="openai:gpt-4o",
                    input_types=(ArtifactType.RAW_TEXT,),
                    output_types=(ArtifactType.CORRECTED_TEXT,),
                    inputs_from={ArtifactType.RAW_TEXT: "ocr"},
                ),
                PipelineStep(
                    id="alto_remap", kind="alto_remapping",
                    adapter_name="picarones-contrib:line_remapper",
                    input_types=(
                        ArtifactType.CORRECTED_TEXT, ArtifactType.ALTO_XML,
                    ),
                    output_types=(ArtifactType.ALTO_XML,),
                    inputs_from={
                        ArtifactType.CORRECTED_TEXT: "correction",
                        ArtifactType.ALTO_XML: "ocr",
                    },
                ),
            ),
        )
        assert validate_spec(spec) == []

    def test_inputs_from_initial_explicit(self) -> None:
        """Une étape peut référencer explicitement les entrées
        initiales via ``__initial__``."""
        spec = PipelineSpec(
            name="explicit_initial",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(
                PipelineStep(
                    id="ocr", kind="ocr", adapter_name="tesseract",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(ArtifactType.RAW_TEXT,),
                    inputs_from={ArtifactType.IMAGE: INITIAL_STEP_ID},
                ),
            ),
        )
        assert validate_spec(spec) == []


# ──────────────────────────────────────────────────────────────────────
# Cas invalides
# ──────────────────────────────────────────────────────────────────────


class TestInvalidSpecs:
    def test_empty_pipeline(self) -> None:
        spec = PipelineSpec(name="empty")
        errors = validate_spec(spec)
        assert len(errors) == 1
        assert errors[0].code == "empty_pipeline"

    def test_missing_input_no_initial(self) -> None:
        """Une étape qui demande IMAGE mais initial_inputs vide."""
        spec = PipelineSpec(
            name="missing_image",
            initial_inputs=(),
            steps=(
                PipelineStep(
                    id="ocr", kind="ocr", adapter_name="tesseract",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(ArtifactType.RAW_TEXT,),
                ),
            ),
        )
        errors = validate_spec(spec)
        codes = [e.code for e in errors]
        assert "missing_input" in codes

    def test_missing_input_step_order_wrong(self) -> None:
        """L'étape de correction est avant l'OCR — le RAW_TEXT n'existe
        pas encore."""
        spec = PipelineSpec(
            name="wrong_order",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(
                PipelineStep(
                    id="correct", kind="post_correction",
                    adapter_name="openai",
                    input_types=(ArtifactType.RAW_TEXT,),
                    output_types=(ArtifactType.CORRECTED_TEXT,),
                ),
                PipelineStep(
                    id="ocr", kind="ocr", adapter_name="tesseract",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(ArtifactType.RAW_TEXT,),
                ),
            ),
        )
        errors = validate_spec(spec)
        codes = [e.code for e in errors]
        assert "missing_input" in codes
        # La première étape (correct) doit être le step_id signalé.
        missing = [e for e in errors if e.code == "missing_input"]
        assert any(e.step_id == "correct" for e in missing)

    def test_duplicate_step_id(self) -> None:
        spec = PipelineSpec(
            name="dup",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(
                PipelineStep(
                    id="step", kind="ocr", adapter_name="a",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(ArtifactType.RAW_TEXT,),
                ),
                PipelineStep(
                    id="step", kind="post_correction", adapter_name="b",
                    input_types=(ArtifactType.RAW_TEXT,),
                    output_types=(ArtifactType.CORRECTED_TEXT,),
                ),
            ),
        )
        errors = validate_spec(spec)
        codes = [e.code for e in errors]
        assert "duplicate_id" in codes

    def test_unknown_adapter_when_registry_provided(self) -> None:
        spec = PipelineSpec(
            name="unknown",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(
                PipelineStep(
                    id="ocr", kind="ocr", adapter_name="not_in_registry",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(ArtifactType.RAW_TEXT,),
                ),
            ),
        )
        errors = validate_spec(spec, available_adapters={"tesseract"})
        codes = [e.code for e in errors]
        assert "unknown_adapter" in codes

    def test_no_adapter_check_when_registry_none(self) -> None:
        """Si available_adapters=None, on ne vérifie pas les adapters."""
        spec = PipelineSpec(
            name="x",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(
                PipelineStep(
                    id="ocr", kind="ocr", adapter_name="not_registered_anywhere",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(ArtifactType.RAW_TEXT,),
                ),
            ),
        )
        errors = validate_spec(spec)  # registry=None
        codes = [e.code for e in errors]
        assert "unknown_adapter" not in codes

    def test_inputs_from_unused_type(self) -> None:
        """Une étape déclare ``inputs_from[X]`` mais X n'est pas dans
        son ``input_types``."""
        spec = PipelineSpec(
            name="x",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(
                PipelineStep(
                    id="ocr", kind="ocr", adapter_name="tess",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(ArtifactType.RAW_TEXT,),
                    inputs_from={ArtifactType.ALTO_XML: INITIAL_STEP_ID},
                ),
            ),
        )
        errors = validate_spec(spec)
        codes = [e.code for e in errors]
        assert "inputs_from_unused" in codes

    def test_unknown_input_source(self) -> None:
        """``inputs_from[type] = "ghost"`` mais ``ghost`` n'existe pas."""
        spec = PipelineSpec(
            name="x",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(
                PipelineStep(
                    id="ocr", kind="ocr", adapter_name="tess",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(ArtifactType.RAW_TEXT,),
                    inputs_from={ArtifactType.IMAGE: "ghost"},
                ),
            ),
        )
        errors = validate_spec(spec)
        codes = [e.code for e in errors]
        assert "unknown_input_source" in codes

    def test_source_does_not_produce_type(self) -> None:
        """``inputs_from[ALTO_XML] = "ocr"`` mais ``ocr`` ne produit que
        ``RAW_TEXT``."""
        spec = PipelineSpec(
            name="x",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(
                PipelineStep(
                    id="ocr", kind="ocr", adapter_name="tess",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(ArtifactType.RAW_TEXT,),
                ),
                PipelineStep(
                    id="alto_consumer", kind="x", adapter_name="y",
                    input_types=(ArtifactType.ALTO_XML,),
                    output_types=(ArtifactType.ALTO_XML,),
                    inputs_from={ArtifactType.ALTO_XML: "ocr"},
                ),
            ),
        )
        errors = validate_spec(spec)
        codes = [e.code for e in errors]
        assert "source_does_not_produce_type" in codes
        # En plus, ALTO_XML n'est pas disponible dans le bag → missing_input
        # peut aussi être levé.

    def test_multiple_errors_at_once(self) -> None:
        """``validate_spec`` ne s'arrête pas à la première erreur."""
        spec = PipelineSpec(
            name="multi_errors",
            initial_inputs=(),
            steps=(
                PipelineStep(
                    id="dup", kind="x", adapter_name="a",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(),
                ),
                PipelineStep(
                    id="dup", kind="y", adapter_name="b",
                    input_types=(ArtifactType.RAW_TEXT,),
                    output_types=(),
                ),
            ),
        )
        errors = validate_spec(spec)
        codes = [e.code for e in errors]
        assert "duplicate_id" in codes
        assert "missing_input" in codes  # IMAGE et RAW_TEXT manquants
