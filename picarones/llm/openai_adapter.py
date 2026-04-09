"""Adaptateur LLM — OpenAI (GPT-4o, GPT-4o-mini)."""

from __future__ import annotations

import logging
import os
from typing import Optional

from picarones.llm.base import BaseLLMAdapter

logger = logging.getLogger(__name__)


class OpenAIAdapter(BaseLLMAdapter):
    """Adaptateur pour les modèles OpenAI (GPT-4o, GPT-4o-mini).

    Clé API via la variable d'environnement ``OPENAI_API_KEY``.

    Modes supportés : text_only, text_and_image, zero_shot.
    """

    @property
    def name(self) -> str:
        return "openai"

    @property
    def default_model(self) -> str:
        return "gpt-4o"

    def __init__(
        self,
        model: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__(model, config)
        self._api_key = os.environ.get("OPENAI_API_KEY")

    def _call(self, prompt: str, image_b64: Optional[str] = None) -> str:
        if not self._api_key:
            raise RuntimeError(
                "Clé API OpenAI manquante — définissez la variable d'environnement OPENAI_API_KEY"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Le package 'openai' n'est pas installé. Lancez : pip install openai"
            ) from exc

        client = OpenAI(api_key=self._api_key)
        temperature = float(self.config.get("temperature", 0.0))
        max_tokens = int(self.config.get("max_tokens", 4096))

        if image_b64:
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
            ]
        else:
            content = prompt  # type: ignore[assignment]

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code == 401:
                logger.warning(
                    "[OpenAIAdapter] erreur HTTP 401 — clé API invalide (modèle=%s).",
                    self.model,
                )
            elif status_code == 429:
                logger.warning(
                    "[OpenAIAdapter] erreur HTTP 429 — rate limit (modèle=%s).",
                    self.model,
                )
            else:
                logger.warning(
                    "[OpenAIAdapter] erreur API (modèle=%s) : %s", self.model, exc,
                )
            raise

        if not response.choices:
            logger.warning(
                "[OpenAIAdapter] response.choices vide (modèle=%s).", self.model,
            )
            return ""
        return response.choices[0].message.content or ""
