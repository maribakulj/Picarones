"""``BaseVLMAdapter`` — Sprint A14-S45.

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
from picarones.adapters.ocr.base import OCRAdapterError
from picarones.domain.artifacts import Artifact, ArtifactType

logger = logging.getLogger(__name__)


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
    """

    @property
    def input_types(self) -> "frozenset":
        return frozenset({ArtifactType.IMAGE})

    @property
    def output_types(self) -> "frozenset":
        return frozenset({ArtifactType.RAW_TEXT})

    DEFAULT_TRANSCRIPTION_PROMPT: str = (
        "Transcris fidèlement le texte visible sur cette image de "
        "document historique. Conserve l'orthographe historique, les "
        "abréviations, et la ponctuation. Retourne uniquement le "
        "texte transcrit, sans commentaire."
    )

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
            raise OCRAdapterError(
                f"{self.name} : input IMAGE manquant.",
            )
        image_artifact = inputs[ArtifactType.IMAGE]
        if image_artifact.uri is None:
            raise OCRAdapterError(
                f"{self.name} : artefact image "
                f"{image_artifact.id!r} sans URI.",
            )
        image_path = Path(image_artifact.uri)
        if not image_path.exists():
            raise OCRAdapterError(
                f"{self.name} : image introuvable {image_path!r}.",
            )

        image_b64 = base64.b64encode(
            image_path.read_bytes(),
        ).decode("ascii")

        prompt = self.config.get(
            "transcription_prompt", self.DEFAULT_TRANSCRIPTION_PROMPT,
        )

        result = self.complete(prompt, image_b64=image_b64)
        if not result.success:
            raise OCRAdapterError(
                f"{self.name} : VLM a échoué ({result.error}).",
            )

        out_path = (
            image_path.parent / f"{image_path.stem}.{self.name}.txt"
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


__all__ = ["BaseVLMAdapter"]
