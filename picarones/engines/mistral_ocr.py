"""Adaptateur OCR — Mistral OCR (API vision Mistral AI).

Utilise l'API Mistral pour la reconnaissance de texte sur documents
patrimoniaux via le modèle multimodal Mistral.

Clé API : variable d'environnement ``MISTRAL_API_KEY``.

Documentation API : https://docs.mistral.ai/
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional

from picarones.engines.base import BaseOCREngine


class MistralOCREngine(BaseOCREngine):
    """Moteur OCR via l'API Mistral AI (modèle vision).

    Configuration
    -------------
    model : str
        Modèle Mistral à utiliser (défaut : ``"pixtral-12b-2409"``).
        Les modèles multimodaux supportant la vision sont :
        ``pixtral-12b-2409``, ``pixtral-large-latest``.
    prompt : str
        Prompt envoyé avec l'image. Défaut : instruction générique de transcription.
    max_tokens : int
        Limite de tokens en sortie (défaut : 4096).
    """

    @property
    def name(self) -> str:
        return "mistral_ocr"

    def version(self) -> str:
        return self.config.get("model", "mistral-ocr-latest")

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._api_key = os.environ.get("MISTRAL_API_KEY")
        self._model = self.config.get("model", "mistral-ocr-latest")
        self._prompt = self.config.get(
            "prompt",
            "Transcris fidèlement le texte visible sur cette image de document "
            "historique. Retourne uniquement le texte, sans commentaire.",
        )
        self._max_tokens = int(self.config.get("max_tokens", 4096))

    def _run_ocr(self, image_path: Path) -> str:
        if not self._api_key:
            raise RuntimeError(
                "Clé API Mistral manquante — définissez la variable d'environnement MISTRAL_API_KEY"
            )

        suffix = image_path.suffix.lower()
        media_type = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".tif": "image/tiff",
            ".tiff": "image/tiff", ".webp": "image/webp",
        }.get(suffix, "image/jpeg")

        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        image_url = f"data:{media_type};base64,{image_b64}"

        if "mistral-ocr" in self._model.lower():
            return self._run_ocr_native_api(image_url)
        return self._run_ocr_vision_api(image_url)

    def _run_ocr_native_api(self, image_url: str) -> str:
        """Endpoint dédié /v1/ocr (pour mistral-ocr-latest et variantes)."""
        import json
        import urllib.request

        payload = json.dumps({
            "model": self._model,
            "document": {"type": "image_url", "image_url": image_url},
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.mistral.ai/v1/ocr",
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        pages = data.get("pages", [])
        return "\n\n".join(p.get("markdown", "") for p in pages).strip()

    def _run_ocr_vision_api(self, image_url: str) -> str:
        """API vision/chat Mistral (pour pixtral-12b, pixtral-large, etc.)."""
        try:
            try:
                from mistralai.client import Mistral
            except ImportError:
                from mistralai import Mistral  # type: ignore[no-redef]
        except ImportError as exc:
            raise RuntimeError(
                "Le package 'mistralai' n'est pas installé. Lancez : pip install mistralai"
            ) from exc

        client = Mistral(api_key=self._api_key)
        response = client.chat.complete(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._prompt},
                        {"type": "image_url", "image_url": image_url},
                    ],
                }
            ],
            max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content or ""
