"""``OCRLLMPipelineConfig`` — container canonique pour pipelines OCR+LLM.

Sprint H.2.b/c du plan v2.0 — équivalent canonique de
``picarones.adapters.legacy_pipelines.base.OCRLLMPipeline``.

Pourquoi
--------
``OCRLLMPipeline`` (legacy) :

- hérite de ``BaseOCREngine`` (legacy),
- expose une méthode ``run(image_path) → EngineResult``,
- mélange contrat d'exécution et configuration.

Cette config canonique :

- est un container *pur* (immutable, pas de logique d'exécution),
- accepte un ``BaseOCRAdapter`` (canonique) au lieu d'un
  ``BaseOCREngine`` (legacy) pour le step OCR amont,
- ne dépend pas du legacy.

L'exécution effective passe par ``PipelineExecutor`` qui consomme
une ``PipelineSpec`` construite via ``make_ocr_llm_pipeline_spec``.

Duck-typing compat
------------------
Pour faciliter la migration progressive,
``OCRLLMPipelineConfig`` expose les mêmes attributs/propriétés
que ``OCRLLMPipeline`` legacy :

- ``is_pipeline = True``,
- ``ocr_engine`` (alias de ``ocr_adapter`` côté canonique),
- ``llm_adapter``,
- ``mode`` (string, pas enum — tolérance ajoutée dans
  ``_ocr_llm_pipeline_to_spec``),
- ``prompt_template``,
- ``name``.

Les helpers
``picarones.app.services._legacy_runner_adapter.engine_to_pipeline_spec``
et ``build_adapter_resolver`` traitent donc indifféremment les
deux types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

OCRLLMMode = Literal["text_only", "text_and_image", "zero_shot"]


@dataclass(frozen=True)
class OCRLLMPipelineConfig:
    """Configuration canonique pour une pipeline OCR + LLM.

    Parameters
    ----------
    llm_adapter:
        Instance ``BaseLLMAdapter`` (canonique, déjà
        ``StepExecutor`` natif depuis Sprint A14-S44).
    mode:
        ``"text_only"`` (LLM corrige le texte OCR pur),
        ``"text_and_image"`` (VLM corrige texte + image),
        ``"zero_shot"`` (VLM transcrit sans OCR amont).
    ocr_adapter:
        Instance ``BaseOCRAdapter`` (canonique).  ``None`` pour
        ``zero_shot``.
    prompt_template:
        Template de prompt passé au LLM.  Vide → l'adapter LLM
        utilise son prompt par défaut.
    pipeline_name:
        Nom lisible affiché dans les rapports.  Si vide, dérivé
        des composants.

    Examples
    --------
    >>> from picarones.adapters.ocr import ocr_adapter_from_name
    >>> from picarones.adapters.llm.openai_adapter import OpenAIAdapter
    >>> config = OCRLLMPipelineConfig(
    ...     ocr_adapter=ocr_adapter_from_name("tesseract"),
    ...     llm_adapter=OpenAIAdapter(model="gpt-4o"),
    ...     mode="text_only",
    ...     prompt_template="Corrige les erreurs OCR :",
    ... )
    >>> config.is_pipeline
    True
    >>> config.name
    'tesseract → gpt-4o'
    """

    llm_adapter: Any
    mode: OCRLLMMode
    ocr_adapter: Any | None = None
    prompt_template: str = ""
    pipeline_name: str = ""

    #: Marker duck-typing pour les helpers existants qui distinguent
    #: les pipelines composées des engines simples via ce flag.
    is_pipeline: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        if self.mode not in ("text_only", "text_and_image", "zero_shot"):
            raise ValueError(
                f"OCRLLMPipelineConfig : mode invalide {self.mode!r}.  "
                "Valeurs valides : text_only, text_and_image, zero_shot.",
            )
        if self.mode != "zero_shot" and self.ocr_adapter is None:
            raise ValueError(
                f"OCRLLMPipelineConfig : mode {self.mode!r} requiert "
                "un ``ocr_adapter`` non-None (l'OCR amont alimente le LLM).",
            )
        if self.mode == "zero_shot" and self.ocr_adapter is not None:
            raise ValueError(
                "OCRLLMPipelineConfig : mode 'zero_shot' ne doit pas "
                "avoir d'``ocr_adapter`` (le VLM lit l'image directement).",
            )

    @property
    def name(self) -> str:
        """Nom lisible — défini ou dérivé."""
        if self.pipeline_name:
            return self.pipeline_name
        if self.mode == "zero_shot":
            return f"{self.llm_adapter.model} (zero-shot)"
        if self.ocr_adapter is not None:
            return f"{self.ocr_adapter.name} → {self.llm_adapter.model}"
        return f"pipeline → {self.llm_adapter.model}"

    @property
    def ocr_engine(self) -> Any | None:
        """Compat duck-typing avec ``OCRLLMPipeline`` legacy.

        Les helpers ``_ocr_llm_pipeline_to_spec`` et
        ``build_adapter_resolver`` accèdent à ``pipeline.ocr_engine``
        — on expose ``ocr_adapter`` sous ce nom pour la
        rétro-compatibilité du wiring existant.
        """
        return self.ocr_adapter


__all__ = ["OCRLLMMode", "OCRLLMPipelineConfig"]
