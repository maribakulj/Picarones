"""``OpenAIVLMAdapter`` — GPT-4-Vision / GPT-4o (vision).

Sprint A14-S45.  Délègue à ``OpenAIAdapter`` qui supporte déjà la
vision via les modèles ``gpt-4o``, ``gpt-4-turbo``,
``gpt-4-vision-preview``.
"""

from __future__ import annotations

from picarones.adapters.llm.openai_adapter import OpenAIAdapter
from picarones.adapters.vlm.base import BaseVLMAdapter


class OpenAIVLMAdapter(BaseVLMAdapter, OpenAIAdapter):
    """VLM OpenAI (gpt-4o, gpt-4-turbo, gpt-4-vision-preview)."""

    @property
    def name(self) -> str:
        return "openai_vlm"


__all__ = ["OpenAIVLMAdapter"]
