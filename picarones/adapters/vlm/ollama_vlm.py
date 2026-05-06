"""``OllamaVLMAdapter`` — Modèles vision locaux via Ollama.

Sprint A14-S45.  Délègue à ``OllamaAdapter`` (local, sans clé API).
Modèles vision recommandés : ``llava``, ``llava:13b``, ``bakllava``,
``llama3.2-vision``.
"""

from __future__ import annotations

from picarones.adapters.llm.ollama_adapter import OllamaAdapter
from picarones.adapters.vlm.base import BaseVLMAdapter


class OllamaVLMAdapter(BaseVLMAdapter, OllamaAdapter):
    """VLM local via Ollama (llava, bakllava, llama3.2-vision)."""

    @property
    def name(self) -> str:
        return "ollama_vlm"

    @property
    def default_model(self) -> str:
        return "llava"


__all__ = ["OllamaVLMAdapter"]
