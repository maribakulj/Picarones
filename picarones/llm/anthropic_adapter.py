"""Adaptateur LLM — Anthropic (Claude Sonnet, Claude Haiku)."""

from __future__ import annotations

import os
from typing import Optional

from picarones.llm.base import BaseLLMAdapter


class AnthropicAdapter(BaseLLMAdapter):
    """Adaptateur pour les modèles Anthropic Claude.

    Clé API via la variable d'environnement ``ANTHROPIC_API_KEY``.

    Modes supportés : text_only, text_and_image, zero_shot.
    """

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def default_model(self) -> str:
        return "claude-sonnet-4-6"

    def __init__(
        self,
        model: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__(model, config)
        self._api_key = os.environ.get("ANTHROPIC_API_KEY")

    def _call(self, prompt: str, image_b64: Optional[str] = None) -> str:
        if not self._api_key:
            raise RuntimeError(
                "Clé API Anthropic manquante — définissez la variable d'environnement ANTHROPIC_API_KEY"
            )
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "Le package 'anthropic' n'est pas installé. Lancez : pip install anthropic"
            ) from exc

        client = anthropic.Anthropic(api_key=self._api_key)
        temperature = float(self.config.get("temperature", 0.0))
        max_tokens = int(self.config.get("max_tokens", 4096))

        if image_b64:
            content: list | str = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": prompt},
            ]
        else:
            content = prompt

        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": content}],
        )
        return response.content[0].text
