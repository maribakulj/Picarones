"""Builder de ``PipelineSpec`` pour les chaînes OCR + LLM (Phase 6 volet 2).

Ce module fournit la convergence entre les 3 modes historiques de
``picarones.pipelines.base.OCRLLMPipeline`` (legacy) et la
``PipelineSpec`` canonique exécutable par ``PipelineExecutor``.

Mapping mode legacy → spec canonique
------------------------------------

================ ============= =========== ================================
Mode legacy      Initial input Steps       Output final
================ ============= =========== ================================
``text_only``    IMAGE         OCR + LLM   ``CORRECTED_TEXT``
``text_and_image`` IMAGE       OCR + LLM   ``CORRECTED_TEXT`` (LLM voit aussi IMAGE)
``zero_shot``    IMAGE         VLM seul    ``RAW_TEXT``
================ ============= =========== ================================

Les 3 modes correspondent aux contrats ``StepExecutor`` :

- ``BaseLLMAdapter`` (texte → texte corrigé) — couvre ``text_only``
  et ``text_and_image`` car son ``execute()`` lit l'image
  optionnellement présente dans le bag d'inputs.
- ``BaseVLMAdapter`` (image → texte) — couvre ``zero_shot``.

L'adapter OCR amont (Tesseract, Pero, Mistral OCR, Google Vision,
Azure DI, ou ``precomputed`` quand le corpus porte déjà l'OCR) est
quelconque tant qu'il déclare ``output_types ⊇ {RAW_TEXT}``.

Exemple de migration
--------------------
Code legacy ::

    from picarones.pipelines import OCRLLMPipeline, PipelineMode
    from picarones.adapters.legacy_engines.tesseract import TesseractEngine
    from picarones.adapters.llm import OpenAIAdapter

    pipeline = OCRLLMPipeline(
        ocr_engine=TesseractEngine({"lang": "fra"}),
        llm_adapter=OpenAIAdapter(model="gpt-4o"),
        mode=PipelineMode.TEXT_ONLY,
    )
    result = pipeline.run("scan.jpg")  # → EngineResult

Code canonique équivalent ::

    from picarones.pipeline import PipelineExecutor
    from picarones.pipeline.llm_pipeline_builder import (
        make_ocr_llm_pipeline_spec,
    )

    spec = make_ocr_llm_pipeline_spec(
        mode="text_only",
        ocr_adapter_name="tesseract",
        llm_adapter_name="openai:gpt-4o",
    )
    executor = PipelineExecutor(adapter_resolver=resolver, ...)
    result = executor.run(spec, document, initial_inputs={IMAGE: ...}, context=...)

Le runtime résout les ``adapter_name`` en instances via le
``adapter_resolver`` du caller (cf. ``picarones.app.services.run_orchestrator``).
"""

from __future__ import annotations

from typing import Literal

from picarones.domain.artifacts import ArtifactType
from picarones.domain.errors import PicaronesError
from picarones.domain.pipeline_spec import (
    INITIAL_STEP_ID,
    PipelineSpec,
    PipelineStep,
)


#: Modes supportés — alignés sur ``picarones.pipelines.base.PipelineMode``.
OCRLLMPipelineMode = Literal["text_only", "text_and_image", "zero_shot"]


def make_ocr_llm_pipeline_spec(
    mode: OCRLLMPipelineMode,
    *,
    ocr_adapter_name: str | None = None,
    llm_adapter_name: str,
    name: str | None = None,
    description: str = "",
    ocr_step_id: str = "ocr",
    llm_step_id: str = "llm",
    ocr_params: dict[str, str | int | float | bool] | None = None,
    llm_params: dict[str, str | int | float | bool] | None = None,
) -> PipelineSpec:
    """Construit la ``PipelineSpec`` correspondant à un mode OCR+LLM.

    Parameters
    ----------
    mode:
        ``"text_only"`` (OCR → LLM texte) | ``"text_and_image"`` (OCR
        → LLM texte+image) | ``"zero_shot"`` (VLM image → texte).
    ocr_adapter_name:
        Nom de l'adapter OCR amont (ex. ``"tesseract"``,
        ``"precomputed"``).  **Requis** pour ``text_only`` et
        ``text_and_image`` ; **interdit** pour ``zero_shot``.
    llm_adapter_name:
        Nom de l'adapter LLM ou VLM (ex. ``"openai:gpt-4o"``,
        ``"anthropic:claude-3-5-sonnet"``).  Pour ``zero_shot``,
        doit pointer sur un VLM adapter.
    name:
        Nom court de la pipeline (snake_case).  Auto-généré depuis
        ``mode`` + adapters si non fourni.
    description:
        Phrase courte pour le rapport.  Vide par défaut.
    ocr_step_id, llm_step_id:
        Identifiants des étapes (utiles pour les ``inputs_from``
        cross-pipeline).  Défauts : ``"ocr"`` et ``"llm"``.
    ocr_params:
        Paramètres dynamiques passés au step OCR au runtime
        (Sprint B du plan v2.0).  Typiquement vide — la
        configuration de l'adapter passe par son constructeur.
        Format scalaire (``str``, ``int``, ``float``, ``bool``).
    llm_params:
        Paramètres dynamiques passés au step LLM/VLM au runtime.
        Cas typique (Sprint B du plan v2.0) :
        ``{"prompt_template": "Corrige : {ocr_output}"}`` permet à
        un caller de spécifier un template legacy ou rewrite sans
        toucher à la config de l'adapter.

    Returns
    -------
    PipelineSpec
        Spec immutable prête à être exécutée par ``PipelineExecutor``.

    Raises
    ------
    PicaronesError
        Si la combinaison mode/adapters est incohérente
        (ex. ``zero_shot`` avec ``ocr_adapter_name`` fourni).
    """
    if mode == "zero_shot":
        if ocr_adapter_name is not None:
            raise PicaronesError(
                "mode 'zero_shot' incompatible avec ocr_adapter_name : "
                "le VLM consomme directement l'image, pas d'OCR amont."
            )
        return _make_zero_shot_spec(
            llm_adapter_name=llm_adapter_name,
            name=name or f"vlm_zero_shot_{_safe_name(llm_adapter_name)}",
            description=description,
            llm_step_id=llm_step_id,
            llm_params=llm_params,
        )

    if mode not in ("text_only", "text_and_image"):
        raise PicaronesError(
            f"mode OCR+LLM inconnu : {mode!r}.  "
            "Attendu : text_only | text_and_image | zero_shot."
        )

    if not ocr_adapter_name:
        raise PicaronesError(
            f"mode {mode!r} requiert ocr_adapter_name (un adapter "
            "produisant RAW_TEXT en amont du LLM)."
        )

    return _make_ocr_plus_llm_spec(
        mode=mode,
        ocr_adapter_name=ocr_adapter_name,
        llm_adapter_name=llm_adapter_name,
        name=name or (
            f"ocr_llm_{mode}_"
            f"{_safe_name(ocr_adapter_name)}_to_{_safe_name(llm_adapter_name)}"
        ),
        description=description,
        ocr_step_id=ocr_step_id,
        llm_step_id=llm_step_id,
        ocr_params=ocr_params,
        llm_params=llm_params,
    )


def _make_zero_shot_spec(
    *,
    llm_adapter_name: str,
    name: str,
    description: str,
    llm_step_id: str,
    llm_params: dict[str, str | int | float | bool] | None = None,
) -> PipelineSpec:
    """Spec ``zero_shot`` : un seul step VLM IMAGE → RAW_TEXT."""
    return PipelineSpec(
        name=name,
        description=description,
        initial_inputs=(ArtifactType.IMAGE,),
        steps=(
            PipelineStep(
                id=llm_step_id,
                kind="zero_shot_transcription",
                adapter_name=llm_adapter_name,
                params=llm_params or {},
                input_types=(ArtifactType.IMAGE,),
                output_types=(ArtifactType.RAW_TEXT,),
                inputs_from={ArtifactType.IMAGE: INITIAL_STEP_ID},
            ),
        ),
    )


def _make_ocr_plus_llm_spec(
    *,
    mode: str,
    ocr_adapter_name: str,
    llm_adapter_name: str,
    name: str,
    description: str,
    ocr_step_id: str,
    llm_step_id: str,
    ocr_params: dict[str, str | int | float | bool] | None = None,
    llm_params: dict[str, str | int | float | bool] | None = None,
) -> PipelineSpec:
    """Spec à 2 steps : OCR (IMAGE → RAW_TEXT) + LLM (RAW_TEXT → CORRECTED_TEXT)."""
    llm_inputs_from: dict[ArtifactType, str] = {
        ArtifactType.RAW_TEXT: ocr_step_id,
    }
    llm_input_types: list[ArtifactType] = [ArtifactType.RAW_TEXT]
    if mode == "text_and_image":
        # Le LLM voit aussi l'image initiale (mode multimodal).
        llm_inputs_from[ArtifactType.IMAGE] = INITIAL_STEP_ID
        llm_input_types.append(ArtifactType.IMAGE)

    return PipelineSpec(
        name=name,
        description=description,
        initial_inputs=(ArtifactType.IMAGE,),
        steps=(
            PipelineStep(
                id=ocr_step_id,
                kind="ocr",
                adapter_name=ocr_adapter_name,
                params=ocr_params or {},
                input_types=(ArtifactType.IMAGE,),
                output_types=(ArtifactType.RAW_TEXT,),
                inputs_from={ArtifactType.IMAGE: INITIAL_STEP_ID},
            ),
            PipelineStep(
                id=llm_step_id,
                kind="post_correction",
                adapter_name=llm_adapter_name,
                params=llm_params or {},
                input_types=tuple(llm_input_types),
                output_types=(ArtifactType.CORRECTED_TEXT,),
                inputs_from=llm_inputs_from,
            ),
        ),
    )


def _safe_name(adapter_name: str) -> str:
    """Convertit un ``adapter_name`` (qui peut contenir ``:``, ``/``,
    etc.) en suffixe ``snake_case`` valide pour un step id."""
    return (
        adapter_name
        .replace(":", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
        .lower()
    )


__all__ = [
    "OCRLLMPipelineMode",
    "make_ocr_llm_pipeline_spec",
]
