"""Adaptateur LLM — Anthropic (Claude Sonnet, Claude Haiku)."""

from __future__ import annotations

import logging
import os
from typing import Optional

from picarones.llm.base import (
    BaseLLMAdapter,
    log_http_error,
    normalize_llm_content,
)

logger = logging.getLogger(__name__)


class AnthropicAdapter(BaseLLMAdapter):
    """Adaptateur pour les modèles Anthropic Claude.

    Clé API via la variable d'environnement ``ANTHROPIC_API_KEY``.

    Modes supportés : text_only, text_and_image, zero_shot.
    """

    api_key_env_var = "ANTHROPIC_API_KEY"

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

        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": content}],
            )
        except Exception as exc:
            # Chantier 4 — log discriminant (401/429/5xx) factorisé.
            # Auparavant Anthropic ne discriminait pas par code HTTP,
            # difficile à diagnostiquer (clé invalide vs rate limit).
            log_http_error(
                "AnthropicAdapter", self.model, exc,
                env_var=self.api_key_env_var,
            )
            raise

        if not response.content:
            logger.warning(
                "[AnthropicAdapter] réponse vide (modèle=%s, stop_reason=%s).",
                self.model, getattr(response, "stop_reason", None),
            )
            return ""

        # Chantier 4 — propagation du fix Sprint 15 : le SDK Anthropic
        # retourne ``response.content`` comme une liste de blocs
        # (``ContentBlock`` avec attribut ``text``). ``normalize_llm_content``
        # concatène le texte de tous les blocs au lieu de ne prendre que
        # le premier — utile quand le modèle émet plusieurs blocs.
        text = normalize_llm_content(response.content)
        if not text:
            block = response.content[0]
            logger.warning(
                "[AnthropicAdapter] bloc de type '%s' sans texte (modèle=%s).",
                getattr(block, "type", "unknown"), self.model,
            )
        return text
