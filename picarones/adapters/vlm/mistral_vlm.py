"""``MistralVLMAdapter`` — Pixtral 12b/Large (vision Mistral).

Sprint A14-S45.  Délègue à ``MistralAdapter`` qui supporte la
vision via les modèles ``pixtral-12b-2409``, ``pixtral-large-latest``.
"""

from __future__ import annotations

from picarones.adapters.llm.mistral_adapter import MistralAdapter
from picarones.adapters.vlm.base import BaseVLMAdapter


class MistralVLMAdapter(BaseVLMAdapter, MistralAdapter):
    """VLM Mistral (pixtral-12b-2409, pixtral-large-latest)."""

    @property
    def name(self) -> str:
        return "mistral_vlm"

    @property
    def default_model(self) -> str:
        # Ré-définit le défaut pour pointer vers un modèle vision.
        return "pixtral-12b-2409"


__all__ = ["MistralVLMAdapter"]
