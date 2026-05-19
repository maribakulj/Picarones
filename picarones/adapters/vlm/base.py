"""``BaseVLMAdapter``

Adapter VLM (Vision-Language Model) qui hérite de ``BaseLLMAdapter``
et surcharge le contrat StepExecutor pour consommer ``IMAGE`` au
lieu de ``RAW_TEXT`` et produire ``RAW_TEXT`` (transcription
directe par un VLM).

Pas un shim sur les LLM adapters : c'est un mode d'usage différent
de la même API LLM (texte vs image) — le contrat StepExecutor diffère.

Différences avec ``BaseOCRAdapter`` (S26)
-----------------------------------------
- Un OCR (Tesseract, Pero, Mistral OCR, Google Vision, Azure DI)
  utilise des modèles dédiés OCR avec layout structuré, confidences
  natives, etc.
- Un VLM (Anthropic Claude, GPT-4-Vision, Pixtral, LLaVA) fait de la
  transcription via un modèle généraliste prompt+image.

Les deux peuvent produire RAW_TEXT et être comparés en TextView ;
la projection report explicitera ce qu'on perd côté VLM (pas de
coordonnées spatiales nativement).

Convention output : RAW_TEXT (transcription plate).  Une sous-classe
qui produit du markdown structuré (ex. ``CANONICAL_DOCUMENT``) peut
surcharger ``output_types``.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

from picarones.adapters.llm.base import BaseLLMAdapter
from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.domain.errors import AdapterStepError

logger = logging.getLogger(__name__)


class VLMAdapterError(AdapterStepError):
    """Erreur typée pour un échec d'adapter VLM.

    Hérite de ``AdapterStepError`` — racine commune avec les erreurs
    OCR et LLM, ce qui permet à un orchestrateur d'attraper toutes
    les erreurs d'adapter sans connaître le type concret.
    """


class BaseVLMAdapter(BaseLLMAdapter):
    """Adapter VLM qui transcrit une IMAGE en RAW_TEXT.

    Hérite de ``BaseLLMAdapter`` et surcharge le contrat
    ``StepExecutor`` pour consommer ``IMAGE`` au lieu de ``RAW_TEXT``.

    Parameters
    ----------
    model:
        Modèle VLM (cf. sous-classes pour les défauts).
    config:
        Config dict ; supporte
        ``config["transcription_prompt"]`` pour personnaliser le
        prompt de transcription.

    Garde-fou MRO
    -------------
    Les VLM concrets utilisent l'héritage multiple :

    ::

        class AnthropicVLMAdapter(BaseVLMAdapter, AnthropicAdapter)

    L'ordre est critique : ``BaseVLMAdapter`` doit venir d'ABORD
    pour que ``input_types``, ``output_types``, ``execute``, et
    ``DEFAULT_TRANSCRIPTION_PROMPTS`` soient résolus depuis lui (et
    pas depuis le LLM sibling qui aurait des output_types =
    {CORRECTED_TEXT}).

    ``__init_subclass__`` valide cet ordre à la définition de la
    classe.  Si le développeur swap accidentellement les parents
    par habitude alphabétique, la définition de classe lève une
    ``TypeError`` immédiate au lieu d'un comportement silencieusement
    différent (output_types incorrect au runtime).
    """

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # Garde-fou : BaseVLMAdapter doit être le premier parent
        # *non-trivial* dans l'ordre de la déclaration (pour gagner
        # le MRO sur les attributs surchargés).
        bases = cls.__bases__
        if len(bases) <= 1:
            # Sous-classe directe simple — pas de MRO multiple, OK.
            return
        # On parcourt les bases dans l'ordre déclaré.
        try:
            vlm_idx = next(
                i for i, b in enumerate(bases)
                if issubclass(b, BaseVLMAdapter)
            )
        except StopIteration:
            return  # ne devrait pas arriver, vlm subclass DOIT inclure VLM
        # Toutes les bases AVANT BaseVLMAdapter doivent être
        # neutres (mixins sans surcharge des output_types).
        for prev in bases[:vlm_idx]:
            if issubclass(prev, BaseLLMAdapter) and not issubclass(
                prev, BaseVLMAdapter,
            ):
                raise TypeError(
                    f"{cls.__name__} : ordre MRO incorrect — "
                    f"BaseVLMAdapter doit précéder {prev.__name__} "
                    "dans la liste des parents pour que les "
                    "output_types VLM ({IMAGE} → {RAW_TEXT}) "
                    "soient résolus correctement (et pas écrasés "
                    "par les output_types LLM = {CORRECTED_TEXT}). "
                    f"Corrigez : `class {cls.__name__}(BaseVLMAdapter, "
                    f"{prev.__name__})`.",
                )

    @property
    def input_types(self) -> "frozenset":
        return frozenset({ArtifactType.IMAGE})

    @property
    def output_types(self) -> "frozenset":
        return frozenset({ArtifactType.RAW_TEXT})

    #: Prompts de transcription VLM par défaut, indexés par code
    #: langue ISO 639-1 (``fr``, ``en``, ``la``).
    DEFAULT_TRANSCRIPTION_PROMPTS: dict[str, str] = {
        "fr": (
            "Transcris fidèlement le texte visible sur cette image "
            "de document historique. Conserve l'orthographe "
            "historique, les abréviations, et la ponctuation. "
            "Retourne uniquement le texte transcrit, sans commentaire."
        ),
        "en": (
            "Faithfully transcribe the text visible in this image of "
            "a historical document. Preserve the historical "
            "spelling, abbreviations, and punctuation. Return only "
            "the transcribed text, with no commentary."
        ),
        "la": (
            "Fideliter transcribe textum in hac imagine documenti "
            "historici visibilem. Serva orthographiam historicam, "
            "abbreviationes, et interpunctionem. Redde solum textum "
            "transcriptum, sine ulla glossa."
        ),
    }

    def execute(
        self,
        inputs: dict,
        params: dict,
        context: Any,
    ) -> dict:
        """Exécute la transcription VLM.

        Lit ``inputs[IMAGE]`` (URI), encode en base64, appelle
        ``self.complete(prompt, image_b64)``, écrit le résultat
        dans ``<stem>.<name>.txt`` à côté de l'image, et retourne
        ``{RAW_TEXT: Artifact}``.
        """
        if ArtifactType.IMAGE not in inputs:
            raise VLMAdapterError(
                f"{self.name} : input IMAGE manquant.",
            )
        image_artifact = inputs[ArtifactType.IMAGE]
        if image_artifact.uri is None:
            raise VLMAdapterError(
                f"{self.name} : artefact image "
                f"{image_artifact.id!r} sans URI.",
            )
        image_path = Path(image_artifact.uri)
        if not image_path.exists():
            raise VLMAdapterError(
                f"{self.name} : image introuvable {image_path!r}.",
            )

        image_b64 = base64.b64encode(
            image_path.read_bytes(),
        ).decode("ascii")

        # Override explicite > prompt par langue > FR (fallback).
        custom = self.config.get("transcription_prompt")
        if custom is not None:
            prompt = custom
        else:
            lang = (self.config.get("lang") or "fr").lower()
            if lang not in self.DEFAULT_TRANSCRIPTION_PROMPTS:
                logger.warning(
                    "[%s] lang=%r non supportée par "
                    "DEFAULT_TRANSCRIPTION_PROMPTS (%s) — fallback FR. "
                    "Pour un corpus dans cette langue, fournir "
                    "config['transcription_prompt'] explicite.",
                    self.name, lang,
                    sorted(self.DEFAULT_TRANSCRIPTION_PROMPTS.keys()),
                )
            prompt = self.DEFAULT_TRANSCRIPTION_PROMPTS.get(
                lang, self.DEFAULT_TRANSCRIPTION_PROMPTS["fr"],
            )

        result = self.complete(prompt, image_b64=image_b64)
        if not result.success:
            raise VLMAdapterError(
                f"{self.name} : VLM a échoué ({result.error}).",
            )

        from picarones.adapters.output_paths import resolve_output_path
        out_path = resolve_output_path(
            input_path=image_path,
            adapter_name=self.name,
            suffix="txt",
            context=context,
        )
        out_path.write_text(result.text, encoding="utf-8")

        return {
            ArtifactType.RAW_TEXT: Artifact(
                id=f"{context.document_id}:{self.name}:raw_text",
                document_id=context.document_id,
                type=ArtifactType.RAW_TEXT,
                produced_by_step="vlm_transcription",
                uri=str(out_path),
            ),
        }


__all__ = ["BaseVLMAdapter", "VLMAdapterError"]
