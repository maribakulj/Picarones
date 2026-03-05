"""Adaptateur OCR — Google Cloud Vision API.

Utilise l'API Google Cloud Vision pour la détection de texte dans des
documents (méthode ``DOCUMENT_TEXT_DETECTION``, optimisée pour les textes
denses et multilinguistiques).

Authentification :
  - Via service account JSON : variable d'environnement
    ``GOOGLE_APPLICATION_CREDENTIALS`` → chemin vers le fichier JSON
  - Via clé API simple : variable d'environnement ``GOOGLE_API_KEY``

Le mode service account est recommandé pour la production.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from picarones.engines.base import BaseOCREngine


class GoogleVisionEngine(BaseOCREngine):
    """Moteur OCR via l'API Google Cloud Vision.

    Configuration
    -------------
    language_hints : list[str]
        Suggestions de langue (ex : ``["fr"]``). Améliore la précision.
    feature_type : str
        Type de détection : ``"DOCUMENT_TEXT_DETECTION"`` (défaut, pour textes
        denses) ou ``"TEXT_DETECTION"`` (pour textes courts).
    """

    @property
    def name(self) -> str:
        return "google_vision"

    def version(self) -> str:
        return "v1"

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._api_key = os.environ.get("GOOGLE_API_KEY")
        self._credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        self._language_hints: list[str] = self.config.get("language_hints", ["fr"])
        self._feature_type: str = self.config.get("feature_type", "DOCUMENT_TEXT_DETECTION")

    def _run_ocr(self, image_path: Path) -> str:
        # Priorité : SDK google-cloud-vision si disponible, sinon REST direct
        if self._credentials_path:
            return self._run_via_sdk(image_path)
        elif self._api_key:
            return self._run_via_rest(image_path)
        else:
            raise RuntimeError(
                "Authentification Google Vision manquante. Définissez "
                "GOOGLE_APPLICATION_CREDENTIALS (service account JSON) "
                "ou GOOGLE_API_KEY."
            )

    def _run_via_sdk(self, image_path: Path) -> str:
        try:
            from google.cloud import vision
        except ImportError as exc:
            raise RuntimeError(
                "Le package 'google-cloud-vision' n'est pas installé. "
                "Lancez : pip install google-cloud-vision"
            ) from exc

        client = vision.ImageAnnotatorClient()
        image_bytes = image_path.read_bytes()
        image = vision.Image(content=image_bytes)

        if self._feature_type == "DOCUMENT_TEXT_DETECTION":
            response = client.document_text_detection(
                image=image,
                image_context=vision.ImageContext(
                    language_hints=self._language_hints
                ),
            )
            return response.full_text_annotation.text
        else:
            response = client.text_detection(
                image=image,
                image_context=vision.ImageContext(
                    language_hints=self._language_hints
                ),
            )
            texts = response.text_annotations
            return texts[0].description if texts else ""

    def _run_via_rest(self, image_path: Path) -> str:
        """Appel REST direct (sans SDK), avec clé API simple."""
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload = {
            "requests": [
                {
                    "image": {"content": image_b64},
                    "features": [{"type": self._feature_type, "maxResults": 1}],
                    "imageContext": {"languageHints": self._language_hints},
                }
            ]
        }
        url = f"https://vision.googleapis.com/v1/images:annotate?key={self._api_key}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Google Vision API erreur {exc.code}: {exc.read().decode()}") from exc

        responses = result.get("responses", [{}])
        if not responses:
            return ""
        r = responses[0]
        if "error" in r:
            raise RuntimeError(f"Google Vision API erreur : {r['error']}")

        if self._feature_type == "DOCUMENT_TEXT_DETECTION":
            return r.get("fullTextAnnotation", {}).get("text", "")
        else:
            texts = r.get("textAnnotations", [])
            return texts[0]["description"] if texts else ""
