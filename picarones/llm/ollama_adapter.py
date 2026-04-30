"""Adaptateur LLM — Ollama (modèles locaux : Llama 3, Gemma, Phi, Mistral local…)."""

from __future__ import annotations

import logging
from typing import Optional
from urllib.parse import urlparse

from picarones.llm.base import BaseLLMAdapter, normalize_llm_content

logger = logging.getLogger(__name__)


class OllamaAdapter(BaseLLMAdapter):
    """Adaptateur pour les modèles locaux via Ollama.

    Aucune clé API requise. Nécessite un serveur Ollama actif (par défaut
    sur http://localhost:11434).

    Modes supportés :
    - text_only      : tous modèles Ollama
    - text_and_image : modèles multimodaux (llava, bakllava, moondream…)
    - zero_shot      : modèles multimodaux uniquement

    Configuration (via ``config``) :
    - ``base_url`` : URL du serveur Ollama (défaut : http://localhost:11434)
    """

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def default_model(self) -> str:
        return "llama3"

    def __init__(
        self,
        model: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__(model, config)
        base_url = self.config.get("base_url", "http://localhost:11434").rstrip("/")
        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(
                f"URL Ollama invalide (schéma '{parsed.scheme}' non autorisé, "
                f"seuls http/https sont acceptés) : {base_url}"
            )
        self._base_url = base_url

    def _call(self, prompt: str, image_b64: Optional[str] = None) -> str:
        import json
        import urllib.error
        import urllib.request

        temperature = float(self.config.get("temperature", 0.0))
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if image_b64:
            payload["images"] = [image_b64]

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            logger.warning(
                "[OllamaAdapter] erreur HTTP %d (modèle=%s) : %s",
                exc.code, self.model, exc,
            )
            raise RuntimeError(
                f"Erreur HTTP {exc.code} du serveur Ollama ({self._base_url}) : {exc}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Impossible de joindre le serveur Ollama sur {self._base_url}. "
                f"Vérifiez qu'Ollama est démarré (ollama serve). Erreur : {exc}"
            ) from exc

        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(
                "[OllamaAdapter] réponse JSON invalide (modèle=%s) : %s",
                self.model, raw[:200],
            )
            raise RuntimeError(
                f"Réponse JSON invalide du serveur Ollama : {exc}"
            ) from exc

        # Chantier 4 — propagation du fix Sprint 15 : Ollama retourne
        # ``response`` en string mais on normalise par défense (cas où
        # un futur build retournerait un format structuré).
        text = normalize_llm_content(result.get("response", ""))
        if not text:
            logger.warning(
                "[OllamaAdapter] réponse vide (modèle=%s).", self.model,
            )
        return text
