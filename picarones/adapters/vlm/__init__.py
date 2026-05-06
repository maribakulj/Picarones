"""Adapters VLM (Vision-Language Models) — Sprint A14-S45.

VLM = transcription directe par un modèle généraliste avec vision.
Distinct des OCR dédiés (Tesseract, Pero, Mistral OCR, Google Vision,
Azure DI) — un VLM consomme IMAGE et produit RAW_TEXT via prompt
multimodal, sans layout structuré natif.

Adapters livrés
---------------
- ``AnthropicVLMAdapter`` : Claude Sonnet/Opus avec vision.
- ``OpenAIVLMAdapter`` : GPT-4o, GPT-4-turbo, GPT-4-vision-preview.
- ``MistralVLMAdapter`` : Pixtral 12b/Large.
- ``OllamaVLMAdapter`` : LLaVA, BakLLaVA, llama3.2-vision (local).

Convention StepExecutor :

- ``input_types = {IMAGE}``
- ``output_types = {RAW_TEXT}``
- ``execute(inputs, params, context)`` encode l'image en base64,
  appelle le LLM avec un prompt de transcription, écrit le texte
  produit dans ``<stem>.<adapter_name>.txt`` à côté de l'image,
  retourne un Artifact RAW_TEXT.

Pas un shim sur les LLM adapters : c'est un mode d'usage
distinct (vision vs texte) avec un contrat StepExecutor différent.
"""

from __future__ import annotations

from picarones.adapters.vlm.anthropic_vlm import AnthropicVLMAdapter
from picarones.adapters.vlm.base import BaseVLMAdapter
from picarones.adapters.vlm.mistral_vlm import MistralVLMAdapter
from picarones.adapters.vlm.ollama_vlm import OllamaVLMAdapter
from picarones.adapters.vlm.openai_vlm import OpenAIVLMAdapter

__all__ = [
    "BaseVLMAdapter",
    "AnthropicVLMAdapter",
    "MistralVLMAdapter",
    "OllamaVLMAdapter",
    "OpenAIVLMAdapter",
]
