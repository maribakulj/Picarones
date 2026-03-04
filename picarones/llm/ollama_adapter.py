"""Adaptateur LLM — Ollama (modèles locaux : Llama 3, Gemma, Phi, Mistral local…)."""

from __future__ import annotations

from typing import Optional

from picarones.llm.base import BaseLLMAdapter


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
        self._base_url = self.config.get("base_url", "http://localhost:11434").rstrip("/")

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
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Impossible de joindre le serveur Ollama sur {self._base_url}. "
                f"Vérifiez qu'Ollama est démarré (ollama serve). Erreur : {exc}"
            ) from exc
        return result.get("response", "")
