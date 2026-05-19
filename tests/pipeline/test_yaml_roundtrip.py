"""Sprint A14-S6 — round-trip YAML d'une ``PipelineSpec``.

Garantit que ``dump_spec_to_yaml(spec)`` produit du YAML qui se
recharge en une spec strictement égale.  C'est la propriété qui
permet de versionner les pipelines en git de façon
human-readable + machine-actionable.
"""

from __future__ import annotations

import pytest

from picarones.domain import ArtifactType, PicaronesError
from picarones.pipeline import (
    PipelineSpec,
    PipelineStep,
    dump_spec_to_yaml,
    load_spec_from_yaml,
)


def _ocr_only_spec() -> PipelineSpec:
    return PipelineSpec(
        name="ocr_only",
        description="Tesseract sur image patrimoniale.",
        initial_inputs=(ArtifactType.IMAGE,),
        steps=(
            PipelineStep(
                id="ocr",
                kind="ocr",
                adapter_name="tesseract",
                params={"lang": "fra", "psm": 6},
                input_types=(ArtifactType.IMAGE,),
                output_types=(ArtifactType.RAW_TEXT,),
            ),
        ),
    )


def _full_pipeline_spec() -> PipelineSpec:
    return PipelineSpec(
        name="tesseract_llm_alto_remap",
        description="OCR + LLM + remapping ALTO (cas BnF central).",
        initial_inputs=(ArtifactType.IMAGE,),
        steps=(
            PipelineStep(
                id="ocr",
                kind="ocr",
                adapter_name="tesseract",
                params={"lang": "fra"},
                input_types=(ArtifactType.IMAGE,),
                output_types=(ArtifactType.RAW_TEXT, ArtifactType.ALTO_XML),
            ),
            PipelineStep(
                id="correction",
                kind="post_correction",
                adapter_name="openai:gpt-4o",
                params={"temperature": 0.0, "max_tokens": 4096},
                input_types=(ArtifactType.RAW_TEXT,),
                output_types=(ArtifactType.CORRECTED_TEXT,),
                inputs_from={ArtifactType.RAW_TEXT: "ocr"},
            ),
            PipelineStep(
                id="alto_remap",
                kind="alto_remapping",
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


class TestYAMLRoundtrip:
    @pytest.mark.parametrize("spec_factory", [_ocr_only_spec, _full_pipeline_spec])
    def test_roundtrip_preserves_equality(self, spec_factory) -> None:
        spec = spec_factory()
        yml = dump_spec_to_yaml(spec)
        spec2 = load_spec_from_yaml(yml)
        assert spec == spec2

    def test_roundtrip_is_idempotent(self) -> None:
        """Dump → Load → Dump produit le même YAML byte-pour-byte."""
        spec = _full_pipeline_spec()
        yml1 = dump_spec_to_yaml(spec)
        spec2 = load_spec_from_yaml(yml1)
        yml2 = dump_spec_to_yaml(spec2)
        assert yml1 == yml2

    def test_yaml_is_human_readable(self) -> None:
        """Le YAML produit doit utiliser le style 'block' (un champ
        par ligne), pas le style 'flow' (JSON-like)."""
        yml = dump_spec_to_yaml(_full_pipeline_spec())
        assert "name: tesseract_llm_alto_remap" in yml
        assert "steps:" in yml
        # Pas de "{" pour signaler le style block.
        # Les ``params`` peuvent encore contenir des ``{}`` quand le
        # dict est vide ; on vérifie juste que le format général
        # est lisible.
        assert "- id: ocr" in yml

    def test_empty_yaml_raises(self) -> None:
        with pytest.raises(PicaronesError, match="vide"):
            load_spec_from_yaml("")

    def test_yaml_ordered_fields(self) -> None:
        """``sort_keys=False`` doit être respecté."""
        yml = dump_spec_to_yaml(_ocr_only_spec())
        # Dans la spec, ``name`` apparaît avant ``description``,
        # ``initial_inputs`` avant ``steps``.
        i_name = yml.index("name:")
        i_desc = yml.index("description:")
        i_init = yml.index("initial_inputs:")
        i_steps = yml.index("steps:")
        assert i_name < i_desc < i_init < i_steps

    def test_invalid_yaml_raises(self) -> None:
        """Un YAML qui ne respecte pas le schéma de PipelineSpec
        lève une ValidationError pydantic."""
        from pydantic import ValidationError

        bad = "name: x\nsteps:\n  - id: ocr\n    kind: ocr\n    adapter_name: x\n    input_types: [bogus_type]\n"
        with pytest.raises(ValidationError):
            load_spec_from_yaml(bad)
