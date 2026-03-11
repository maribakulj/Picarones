"""Adaptateur LLM — Mistral AI (Mistral Large, Pixtral)."""

from __future__ import annotations

import logging
import os
from typing import Optional

from picarones.llm.base import BaseLLMAdapter

logger = logging.getLogger(__name__)

# Modèles Mistral qui NE supportent PAS l'API chat/completions multimodale.
# Ces petits modèles sont text-only; le passer avec une image provoque une erreur.
_TEXT_ONLY_MODELS = frozenset({
    "ministral-3b-latest",
    "ministral-8b-latest",
    "mistral-tiny",
    "mistral-tiny-latest",
    "open-mistral-7b",
    "open-mixtral-8x7b",
})


class MistralAdapter(BaseLLMAdapter):
    """Adaptateur pour les modèles Mistral AI.

    Clé API via la variable d'environnement ``MISTRAL_API_KEY``.

    Modes supportés : text_only (tous modèles), text_and_image et zero_shot
    avec les modèles multimodaux (pixtral-12b, pixtral-large).

    Note
    ----
    Les modèles ``ministral-3b-latest`` et ``ministral-8b-latest`` ne supportent
    pas le mode multimodal — utiliser ``PipelineMode.TEXT_ONLY`` avec ces modèles.
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
        if self.model in _TEXT_ONLY_MODELS:
            logger.info(
                "[MistralAdapter] modèle '%s' : text-only (pas de support multimodal).",
                self.model,
            )

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

        # Les modèles text-only ne supportent pas les images
        if image_b64 and self.model in _TEXT_ONLY_MODELS:
            logger.warning(
                "[MistralAdapter] modèle '%s' ne supporte pas les images — "
                "image ignorée, appel en mode texte seul.",
                self.model,
            )
            image_b64 = None

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

        logger.info(
            "[MistralAdapter] appel %s — prompt=%d chars, image=%s",
            self.model, len(prompt), "oui" if image_b64 else "non",
        )

        try:
            response = client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            status_code = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
            if status_code == 401:
                logger.warning(
                    "[MistralAdapter] erreur HTTP 401 — clé API invalide ou expirée "
                    "(modèle=%s). Vérifier MISTRAL_API_KEY.",
                    self.model,
                )
            elif status_code == 429:
                logger.warning(
                    "[MistralAdapter] erreur HTTP 429 — quota dépassé ou rate-limit "
                    "(modèle=%s). Réessayer plus tard.",
                    self.model,
                )
            elif status_code is not None and status_code >= 500:
                logger.warning(
                    "[MistralAdapter] erreur HTTP %d — problème serveur Mistral "
                    "(modèle=%s) : %s",
                    status_code, self.model, exc,
                )
            else:
                logger.warning(
                    "[MistralAdapter] erreur lors de l'appel API (modèle=%s) : %s",
                    self.model, exc,
                )
            raise

        if not response.choices:
            logger.warning(
                "[MistralAdapter] response.choices vide (modèle=%s).",
                self.model,
            )
            return ""

        _choice = response.choices[0]
        raw = _choice.message.content
        _finish_reason = _choice.finish_reason

        # Le SDK mistralai peut retourner une liste de ContentChunk au lieu
        # d'une chaîne pour certains modèles/versions.  Normaliser en str.
        if isinstance(raw, list):
            raw = "".join(
                chunk.text if hasattr(chunk, "text") else str(chunk)
                for chunk in raw
            )

        text = raw or ""

        _completion_tokens = None
        if hasattr(response, "usage") and response.usage:
            _completion_tokens = getattr(response.usage, "completion_tokens", None)

        logger.info(
            "[MistralAdapter] réponse %s — finish_reason=%s, len=%d, tokens=%s",
            self.model, _finish_reason, len(text), _completion_tokens,
        )

        if not text.strip():
            logger.warning(
                "[MistralAdapter] réponse vide du modèle '%s' "
                "(finish_reason=%s, completion_tokens=%s). "
                "Vérifier le prompt et la compatibilité du modèle.",
                self.model, _finish_reason, _completion_tokens,
            )

        return text
