"""``AnthropicVLMAdapter`` — Claude Sonnet/Opus en mode vision.

Sprint A14-S45.  Délègue l'appel API au mécanisme de
``AnthropicAdapter`` (qui supporte déjà la vision via le SDK
anthropic) en surchargeant le contrat StepExecutor pour consommer
IMAGE au lieu de RAW_TEXT.
"""

from __future__ import annotations

from picarones.adapters.llm.anthropic_adapter import AnthropicAdapter
from picarones.adapters.vlm.base import BaseVLMAdapter


class AnthropicVLMAdapter(BaseVLMAdapter, AnthropicAdapter):
    """VLM Claude (Sonnet/Opus avec vision).

    L'ordre du MRO est important : ``BaseVLMAdapter`` d'abord pour
    surcharger ``input_types``/``output_types``/``execute``, puis
    ``AnthropicAdapter`` pour ``_call``/``default_model``/``name``/
    retry/validation API key.

    Modèles vision recommandés : ``claude-3-5-sonnet-latest``,
    ``claude-3-opus-latest``.
    """

    @property
    def name(self) -> str:
        return "anthropic_vlm"


__all__ = ["AnthropicVLMAdapter"]
