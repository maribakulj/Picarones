"""``OCRLLMPipelineConfig`` — container pour pipelines OCR+LLM.

Container *pur* (immutable, pas de logique d'exécution) qui décrit
un pipeline composé OCR amont + LLM aval.  L'exécution effective
passe par ``PipelineExecutor`` qui consomme une ``PipelineSpec``
construite via ``make_ocr_llm_pipeline_spec``.

Attributs exposés
-----------------
- ``is_pipeline = True`` — marker consommé par ``benchmark_runner``
  pour distinguer un pipeline composé d'un OCR seul.
- ``ocr_engine`` (alias de ``ocr_adapter``) — adapter OCR amont.
- ``llm_adapter`` — adapter LLM aval.
- ``mode`` — string parmi ``text_only`` / ``text_and_image`` /
  ``zero_shot``.
- ``prompt_template`` — template de prompt pour le LLM.
- ``name`` — nom du pipeline pour l'identification dans le rapport.
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
        # Garde-fou Sprint S9 — anti-régression du bug "filename passé
        # à la place du contenu" (cf. ``tests/web/test_s9_prompt_loading.py``).
        # ``prompt_template`` est censé être le contenu BRUT du prompt
        # (déjà lu depuis disque par le caller), pas un identifiant de
        # ressource.  Un template sans aucun placeholder de substitution
        # est sémantiquement invalide : le LLM recevrait une string fixe
        # qui ignore le texte OCR et halluciinerait une réponse plausible.
        # Le filename ``correction_*.txt`` était précisément ce cas.
        if self.prompt_template:
            has_placeholder = any(
                marker in self.prompt_template
                for marker in ("{ocr_output}", "{text}", "{image_b64}")
            )
            if not has_placeholder:
                # Heuristique pour aider au diagnostic : si la string
                # ressemble à un filename, on le dit explicitement.
                looks_like_filename = (
                    self.prompt_template.endswith(".txt")
                    and "\n" not in self.prompt_template
                    and len(self.prompt_template) < 256
                )
                hint = (
                    " (la string ressemble à un nom de fichier — "
                    "as-tu oublié de charger le contenu via "
                    "``Path(prompts_dir / filename).read_text()`` "
                    "avant de l'injecter ?)"
                    if looks_like_filename else ""
                )
                raise ValueError(
                    "OCRLLMPipelineConfig : ``prompt_template`` ne "
                    "contient aucun placeholder substituable "
                    "(``{ocr_output}``, ``{text}`` ou ``{image_b64}``). "
                    "Le LLM recevrait une string fixe et ignorerait "
                    f"le texte OCR.{hint}",
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
        """Alias historique de ``ocr_adapter``.

        Les helpers ``_ocr_llm_pipeline_to_spec`` et
        ``build_adapter_resolver`` accèdent à ``pipeline.ocr_engine`` ;
        on expose ``ocr_adapter`` sous ce nom pour préserver leur
        wiring.
        """
        return self.ocr_adapter


__all__ = ["OCRLLMMode", "OCRLLMPipelineConfig"]
