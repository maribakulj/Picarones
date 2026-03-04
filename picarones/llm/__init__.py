"""Adaptateurs LLM pour les pipelines OCR+LLM."""

from picarones.llm.base import BaseLLMAdapter, LLMResult
from picarones.llm.anthropic_adapter import AnthropicAdapter
from picarones.llm.mistral_adapter import MistralAdapter
from picarones.llm.ollama_adapter import OllamaAdapter
from picarones.llm.openai_adapter import OpenAIAdapter

__all__ = [
    "BaseLLMAdapter",
    "LLMResult",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "MistralAdapter",
    "OllamaAdapter",
]
