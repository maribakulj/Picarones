"""Adaptateur LLM — Mistral AI (Mistral Large, Pixtral)."""

from __future__ import annotations

import os
from typing import Optional

from picarones.llm.base import BaseLLMAdapter


class MistralAdapter(BaseLLMAdapter):
    """Adaptateur pour les modèles Mistral AI.

    Clé API via la variable d'environnement ``MISTRAL_API_KEY``.

    Modes supportés : text_only (tous modèles), text_and_image et zero_shot
    avec les modèles multimodaux (pixtral-12b, pixtral-large).
    """

    @property
    def name(self) -> str:
        return "mistral"

    @property
    def default_model(self) -> str:
        return "mistral-large-latest"

    def __init__(
        self,
        model: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__(model, config)
        self._api_key = os.environ.get("MISTRAL_API_KEY")

    def _call(self, prompt: str, image_b64: Optional[str] = None) -> str:
        if not self._api_key:
            raise RuntimeError(
                "Clé API Mistral manquante — définissez la variable d'environnement MISTRAL_API_KEY"
            )
        try:
            from mistralai import Mistral
        except ImportError as exc:
            raise RuntimeError(
                "Le package 'mistralai' n'est pas installé. Lancez : pip install mistralai"
            ) from exc

        client = Mistral(api_key=self._api_key)
        temperature = float(self.config.get("temperature", 0.0))
        max_tokens = int(self.config.get("max_tokens", 4096))

        if image_b64:
            content: list | str = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{image_b64}",
                },
            ]
        else:
            content = prompt

        response = client.chat.complete(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
