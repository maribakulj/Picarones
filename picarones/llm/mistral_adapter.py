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

        # DEBUG — prompt envoyé (tronqué à 200 chars)
        logger.debug(
            "[MistralAdapter] appel %s — longueur prompt : %d caractères, image : %s\n"
            "[MistralAdapter] DEBUG prompt (200 premiers chars) : %r",
            self.model, len(prompt), "oui" if image_b64 else "non",
            prompt[:200],
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
            # DEBUG — statut HTTP en cas d'erreur
            logger.debug(
                "[MistralAdapter] DEBUG exception type=%s status_code=%s message=%s",
                type(exc).__name__, status_code, exc,
            )
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

        # DEBUG — réponse brute (choices complètes)
        try:
            choices_debug = [
                {
                    "index": c.index,
                    "finish_reason": c.finish_reason,
                    "message_role": c.message.role if c.message else None,
                    "content_type": type(c.message.content).__name__ if c.message else None,
                    "content_len": len(c.message.content) if c.message and c.message.content else 0,
                    "content_preview": (c.message.content[:80] if c.message and isinstance(c.message.content, str) else repr(c.message.content)[:80]) if c.message else None,
                }
                for c in (response.choices or [])
            ]
        except Exception as _exc:  # noqa: BLE001
            choices_debug = f"<erreur sérialisation choices : {_exc}>"
        logger.debug(
            "[MistralAdapter] DEBUG response.choices : %s",
            choices_debug,
        )
        # DEBUG — usage tokens si disponible
        if hasattr(response, "usage") and response.usage:
            logger.debug(
                "[MistralAdapter] DEBUG usage : prompt_tokens=%s completion_tokens=%s total_tokens=%s",
                getattr(response.usage, "prompt_tokens", "?"),
                getattr(response.usage, "completion_tokens", "?"),
                getattr(response.usage, "total_tokens", "?"),
            )

        if not response.choices:
            logger.warning(
                "[MistralAdapter] DEBUG response.choices est vide — modèle=%s. "
                "Réponse complète : %r",
                self.model, response,
            )
            return ""

        raw = response.choices[0].message.content
        # DEBUG — valeur brute extraite avant retour
        logger.debug(
            "[MistralAdapter] DEBUG choices[0].message.content type=%s valeur=%r",
            type(raw).__name__,
            raw[:200] if isinstance(raw, str) else raw,
        )
        text = raw or ""

        if not text or not text.strip():
            logger.warning(
                "[MistralAdapter] réponse vide reçue du modèle '%s' "
                "(longueur brute : %s). "
                "Vérifier que le modèle supporte l'API chat/completions et "
                "que le prompt contient bien {ocr_output}.",
                self.model, len(raw) if raw is not None else "None",
            )
        else:
            logger.debug(
                "[MistralAdapter] réponse reçue — %d caractères, extrait : %r",
                len(text), text[:120],
            )

        return text
