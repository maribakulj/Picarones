"""Phase 6 volet 2 — ``make_ocr_llm_pipeline_spec``.

Vérifie que les 3 modes historiques de
``picarones.pipelines.base.OCRLLMPipeline`` (text_only,
text_and_image, zero_shot) se traduisent en ``PipelineSpec``
canoniques exécutables par ``PipelineExecutor``.

Ces tests valident la **structure** de la spec produite ; ils ne
lancent pas de vraie exécution OCR/LLM (les adapters concrets sont
testés ailleurs).  Le smoke test d'exécution end-to-end passe par
le runner de fixtures et vit dans
``tests/integration/test_pipeline_executor_smoke.py`` (S8 / S9).
"""

from __future__ import annotations

import pytest

from picarones.domain import ArtifactType, PicaronesError
from picarones.domain.pipeline_spec import INITIAL_STEP_ID
from picarones.pipeline.llm_pipeline_builder import make_ocr_llm_pipeline_spec
from picarones.pipeline.validation import validate_spec


# ──────────────────────────────────────────────────────────────────────
# Mode text_only — OCR + LLM (texte seul)
# ──────────────────────────────────────────────────────────────────────


class TestTextOnlyMode:
    def test_two_steps_ocr_then_llm(self) -> None:
        spec = make_ocr_llm_pipeline_spec(
            mode="text_only",
            ocr_adapter_name="tesseract",
            llm_adapter_name="openai:gpt-4o",
        )
        assert len(spec.steps) == 2
        assert spec.steps[0].kind == "ocr"
        assert spec.steps[0].adapter_name == "tesseract"
        assert spec.steps[1].kind == "post_correction"
        assert spec.steps[1].adapter_name == "openai:gpt-4o"

    def test_initial_input_is_image(self) -> None:
        spec = make_ocr_llm_pipeline_spec(
            mode="text_only",
            ocr_adapter_name="tesseract",
            llm_adapter_name="openai:gpt-4o",
        )
        assert spec.initial_inputs == (ArtifactType.IMAGE,)

    def test_ocr_consumes_image_produces_raw_text(self) -> None:
        spec = make_ocr_llm_pipeline_spec(
            mode="text_only",
            ocr_adapter_name="tesseract",
            llm_adapter_name="mistral:large",
        )
        ocr = spec.steps[0]
        assert ArtifactType.IMAGE in ocr.input_types
        assert ArtifactType.RAW_TEXT in ocr.output_types
        assert ocr.inputs_from[ArtifactType.IMAGE] == INITIAL_STEP_ID

    def test_llm_reads_text_from_ocr_step(self) -> None:
        spec = make_ocr_llm_pipeline_spec(
            mode="text_only",
            ocr_adapter_name="tesseract",
            llm_adapter_name="mistral:large",
        )
        llm = spec.steps[1]
        assert ArtifactType.RAW_TEXT in llm.input_types
        # Crucial : le LLM tire son RAW_TEXT du step OCR (et non des
        # initial inputs) — c'est la chaîne de production.
        assert llm.inputs_from[ArtifactType.RAW_TEXT] == "ocr"

    def test_llm_produces_corrected_text(self) -> None:
        spec = make_ocr_llm_pipeline_spec(
            mode="text_only",
            ocr_adapter_name="tesseract",
            llm_adapter_name="anthropic:claude-3-5-sonnet",
        )
        llm = spec.steps[1]
        assert ArtifactType.CORRECTED_TEXT in llm.output_types

    def test_llm_does_not_see_image_in_text_only(self) -> None:
        """En mode text_only, le LLM ne consomme pas d'IMAGE."""
        spec = make_ocr_llm_pipeline_spec(
            mode="text_only",
            ocr_adapter_name="tesseract",
            llm_adapter_name="ollama:llama3",
        )
        llm = spec.steps[1]
        assert ArtifactType.IMAGE not in llm.input_types
        assert ArtifactType.IMAGE not in llm.inputs_from


# ──────────────────────────────────────────────────────────────────────
# Mode text_and_image — OCR + LLM multimodal
# ──────────────────────────────────────────────────────────────────────


class TestTextAndImageMode:
    def test_two_steps_like_text_only(self) -> None:
        spec = make_ocr_llm_pipeline_spec(
            mode="text_and_image",
            ocr_adapter_name="tesseract",
            llm_adapter_name="openai:gpt-4o",
        )
        assert len(spec.steps) == 2

    def test_llm_consumes_both_text_and_image(self) -> None:
        spec = make_ocr_llm_pipeline_spec(
            mode="text_and_image",
            ocr_adapter_name="tesseract",
            llm_adapter_name="openai:gpt-4o",
        )
        llm = spec.steps[1]
        assert ArtifactType.RAW_TEXT in llm.input_types
        assert ArtifactType.IMAGE in llm.input_types
        # Le RAW_TEXT vient de l'OCR, l'IMAGE vient des inputs initiaux.
        assert llm.inputs_from[ArtifactType.RAW_TEXT] == "ocr"
        assert llm.inputs_from[ArtifactType.IMAGE] == INITIAL_STEP_ID

    def test_llm_still_produces_corrected_text(self) -> None:
        spec = make_ocr_llm_pipeline_spec(
            mode="text_and_image",
            ocr_adapter_name="precomputed",
            llm_adapter_name="mistral:large",
        )
        assert ArtifactType.CORRECTED_TEXT in spec.steps[1].output_types


# ──────────────────────────────────────────────────────────────────────
# Mode zero_shot — VLM seul (pas d'OCR amont)
# ──────────────────────────────────────────────────────────────────────


class TestZeroShotMode:
    def test_single_step(self) -> None:
        spec = make_ocr_llm_pipeline_spec(
            mode="zero_shot",
            llm_adapter_name="anthropic:claude-3-5-sonnet",
        )
        assert len(spec.steps) == 1

    def test_vlm_consumes_image_directly(self) -> None:
        spec = make_ocr_llm_pipeline_spec(
            mode="zero_shot",
            llm_adapter_name="openai:gpt-4o",
        )
        vlm = spec.steps[0]
        assert ArtifactType.IMAGE in vlm.input_types
        assert vlm.inputs_from[ArtifactType.IMAGE] == INITIAL_STEP_ID

    def test_vlm_produces_raw_text_not_corrected(self) -> None:
        """En zero_shot, le VLM transcrit — il produit RAW_TEXT
        (transcription primaire) et non CORRECTED_TEXT (qui implique
        la correction d'un texte préexistant)."""
        spec = make_ocr_llm_pipeline_spec(
            mode="zero_shot",
            llm_adapter_name="anthropic:claude-3-5-sonnet",
        )
        vlm = spec.steps[0]
        assert ArtifactType.RAW_TEXT in vlm.output_types
        assert ArtifactType.CORRECTED_TEXT not in vlm.output_types

    def test_kind_is_zero_shot_transcription(self) -> None:
        spec = make_ocr_llm_pipeline_spec(
            mode="zero_shot",
            llm_adapter_name="mistral:pixtral",
        )
        assert spec.steps[0].kind == "zero_shot_transcription"

    def test_zero_shot_rejects_ocr_adapter(self) -> None:
        """Combinaison incohérente : on ne fournit pas d'OCR amont
        en zero-shot — le VLM consomme directement l'image."""
        with pytest.raises(PicaronesError, match="zero_shot.*incompatible"):
            make_ocr_llm_pipeline_spec(
                mode="zero_shot",
                ocr_adapter_name="tesseract",
                llm_adapter_name="anthropic:claude-3-5-sonnet",
            )


# ──────────────────────────────────────────────────────────────────────
# Validation — les specs produites passent ``validate_spec``
# ──────────────────────────────────────────────────────────────────────


class TestSpecsArevalid:
    @pytest.mark.parametrize(
        "mode,ocr_name",
        [
            ("text_only", "tesseract"),
            ("text_and_image", "tesseract"),
            ("zero_shot", None),
        ],
    )
    def test_spec_passes_validation(self, mode: str, ocr_name: str | None) -> None:
        """Les 3 modes produisent une spec valide ``validate_spec``."""
        spec = make_ocr_llm_pipeline_spec(
            mode=mode,
            ocr_adapter_name=ocr_name,
            llm_adapter_name="openai:gpt-4o",
        )
        # Passer des adapters fictifs disponibles — on teste juste
        # la structure du DAG, pas la résolution runtime.
        validate_spec(
            spec,
            available_adapters={"tesseract", "openai:gpt-4o"},
        )


# ──────────────────────────────────────────────────────────────────────
# Erreurs — combinaisons invalides
# ──────────────────────────────────────────────────────────────────────


class TestErrorPaths:
    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(PicaronesError, match="mode OCR.LLM inconnu"):
            make_ocr_llm_pipeline_spec(
                mode="invalid_mode",  # type: ignore[arg-type]
                ocr_adapter_name="tesseract",
                llm_adapter_name="openai:gpt-4o",
            )

    def test_text_only_requires_ocr(self) -> None:
        with pytest.raises(PicaronesError, match="requiert ocr_adapter_name"):
            make_ocr_llm_pipeline_spec(
                mode="text_only",
                llm_adapter_name="openai:gpt-4o",
            )

    def test_text_and_image_requires_ocr(self) -> None:
        with pytest.raises(PicaronesError, match="requiert ocr_adapter_name"):
            make_ocr_llm_pipeline_spec(
                mode="text_and_image",
                llm_adapter_name="openai:gpt-4o",
            )


# ──────────────────────────────────────────────────────────────────────
# Auto-naming
# ──────────────────────────────────────────────────────────────────────


class TestAutoNaming:
    def test_auto_name_text_only(self) -> None:
        spec = make_ocr_llm_pipeline_spec(
            mode="text_only",
            ocr_adapter_name="tesseract",
            llm_adapter_name="openai:gpt-4o",
        )
        assert "text_only" in spec.name
        assert "tesseract" in spec.name
        # Les ``:`` du nom d'adapter LLM sont remplacés par ``_``.
        assert ":" not in spec.name
        assert "openai_gpt_4o" in spec.name

    def test_explicit_name_overrides_auto(self) -> None:
        spec = make_ocr_llm_pipeline_spec(
            mode="text_only",
            ocr_adapter_name="tesseract",
            llm_adapter_name="openai:gpt-4o",
            name="my_custom_pipeline",
        )
        assert spec.name == "my_custom_pipeline"

    def test_auto_name_zero_shot(self) -> None:
        spec = make_ocr_llm_pipeline_spec(
            mode="zero_shot",
            llm_adapter_name="anthropic:claude-3-5-sonnet",
        )
        assert spec.name.startswith("vlm_zero_shot_")
        assert "claude_3_5_sonnet" in spec.name


# ──────────────────────────────────────────────────────────────────────
# YAML round-trip (réutilise l'infra Sprint S6)
# ──────────────────────────────────────────────────────────────────────


class TestYamlRoundtrip:
    @pytest.mark.parametrize(
        "mode,ocr_name",
        [
            ("text_only", "tesseract"),
            ("text_and_image", "tesseract"),
            ("zero_shot", None),
        ],
    )
    def test_round_trip_through_yaml(self, mode: str, ocr_name: str | None) -> None:
        """Une spec produite par le builder doit faire l'aller-retour
        complet vers YAML sans perte d'information."""
        from picarones.pipeline.yaml_io import (
            dump_spec_to_yaml,
            load_spec_from_yaml,
        )

        original = make_ocr_llm_pipeline_spec(
            mode=mode,
            ocr_adapter_name=ocr_name,
            llm_adapter_name="openai:gpt-4o",
        )
        yaml_text = dump_spec_to_yaml(original)
        reloaded = load_spec_from_yaml(yaml_text)
        assert reloaded == original
