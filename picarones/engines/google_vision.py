"""Adaptateur OCR — Google Cloud Vision API.

Utilise l'API Google Cloud Vision pour la détection de texte dans des
documents (méthode ``DOCUMENT_TEXT_DETECTION``, optimisée pour les textes
denses et multilinguistiques).

Authentification :
  - Via service account JSON : variable d'environnement
    ``GOOGLE_APPLICATION_CREDENTIALS`` → chemin vers le fichier JSON
  - Via clé API simple : variable d'environnement ``GOOGLE_API_KEY``

Le mode service account est recommandé pour la production.

Sprint 50 — exposition des token_confidences
---------------------------------------------
``DOCUMENT_TEXT_DETECTION`` expose ``Word.confidence`` au niveau mot
sur chaque ``page > block > paragraph > word``.  L'adapter parcourt
cette hiérarchie et émet une entrée par mot au format Sprint 42.
Les deux chemins (SDK ``google-cloud-vision`` et REST direct via
``urllib``) sont normalisés vers une représentation unifiée.

Pour ``TEXT_DETECTION`` (mode "court"), aucune confidence par mot
n'est exposée : ``token_confidences = None``.

Refactor du chantier 1 (post-Sprint 97)
---------------------------------------
L'adapter ne surcharge plus ``run()`` — il implémente ``_run_with_native``
et ``_extract_raw_confidences`` (les hooks factorisés dans ``BaseOCREngine``).
Comportement externe et octets de sortie strictement identiques.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

from picarones.evaluation.engines.base import BaseOCREngine


logger = logging.getLogger(__name__)


class GoogleVisionEngine(BaseOCREngine):
    """Moteur OCR via l'API Google Cloud Vision.

    Configuration
    -------------
    language_hints : list[str]
        Suggestions de langue (ex : ``["fr"]``). Améliore la précision.
    feature_type : str
        Type de détection : ``"DOCUMENT_TEXT_DETECTION"`` (défaut, pour textes
        denses) ou ``"TEXT_DETECTION"`` (pour textes courts).
    expose_confidences : bool
        ``True`` (défaut) : extrait ``Word.confidence`` quand
        ``feature_type=DOCUMENT_TEXT_DETECTION`` (Sprint 50).
        ``False`` : désactive l'extraction (économise quelques ms par
        image).
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
        """Retourne uniquement le texte (interface ``BaseOCREngine``)."""
        text, _full = self._run_with_native(image_path)
        return text

    def _run_with_native(
        self, image_path: Path,
    ) -> tuple[str, Optional[dict]]:
        """Exécute l'OCR et retourne ``(text, full_text_annotation_dict)``.

        ``full_text_annotation_dict`` est :
        - le JSON brut ``fullTextAnnotation`` du REST quand on passe
          par REST,
        - une représentation dict normalisée quand on passe par SDK,
        - ``None`` pour ``TEXT_DETECTION`` (mode court sans
          confidence par mot).
        """
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

    def _run_via_sdk(self, image_path: Path) -> tuple[str, Optional[dict]]:
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
            text = response.full_text_annotation.text
            full = self._sdk_full_text_to_dict(response.full_text_annotation)
            return text, full
        else:
            response = client.text_detection(
                image=image,
                image_context=vision.ImageContext(
                    language_hints=self._language_hints
                ),
            )
            texts = response.text_annotations
            text = texts[0].description if texts else ""
            return text, None

    def _run_via_rest(self, image_path: Path) -> tuple[str, Optional[dict]]:
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
        url = "https://vision.googleapis.com/v1/images:annotate"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": self._api_key,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Google Vision API erreur {exc.code}: {exc.read().decode()}") from exc

        responses = result.get("responses", [{}])
        if not responses:
            return "", None
        r = responses[0]
        if "error" in r:
            raise RuntimeError(f"Google Vision API erreur : {r['error']}")

        if self._feature_type == "DOCUMENT_TEXT_DETECTION":
            full = r.get("fullTextAnnotation") or None
            text = (full or {}).get("text", "") if isinstance(full, dict) else ""
            return text, full
        else:
            texts = r.get("textAnnotations", [])
            text = texts[0]["description"] if texts else ""
            return text, None

    # ──────────────────────────────────────────────────────────────────
    # Conversion SDK → dict normalisé (pour traitement uniforme)
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _sdk_full_text_to_dict(full_text_annotation: Any) -> dict:
        """Convertit une réponse proto SDK en dict avec la même
        structure que le REST : ``{pages: [{blocks: [{paragraphs:
        [{words: [{confidence, symbols: [{text}]}]}]}]}]}``."""
        pages = []
        for page in getattr(full_text_annotation, "pages", []) or []:
            blocks = []
            for block in getattr(page, "blocks", []) or []:
                paragraphs = []
                for para in getattr(block, "paragraphs", []) or []:
                    words = []
                    for word in getattr(para, "words", []) or []:
                        symbols = [
                            {"text": getattr(s, "text", "")}
                            for s in getattr(word, "symbols", []) or []
                        ]
                        words.append({
                            "confidence": float(getattr(word, "confidence", 0.0)),
                            "symbols": symbols,
                        })
                    paragraphs.append({"words": words})
                blocks.append({"paragraphs": paragraphs})
            pages.append({"blocks": blocks})
        return {"pages": pages}

    # ──────────────────────────────────────────────────────────────────
    # Extraction des token_confidences au format Sprint 42
    # ──────────────────────────────────────────────────────────────────

    def _extract_raw_confidences(
        self, native: Any,
    ) -> Optional[list[dict[str, Any]]]:
        """Parcourt ``pages → blocks → paragraphs → words`` et émet
        ``{"token": mot, "confidence": float}`` par mot.

        Le mot est reconstitué par concaténation des
        ``word.symbols[i].text``.  ``word.confidence`` ∈ [0, 1] (la
        normalisation par la base accepte directement ce format).
        """
        if not self.config.get("expose_confidences", True):
            return None
        if not native or not isinstance(native, dict):
            return None
        out: list[dict[str, Any]] = []
        for page in native.get("pages") or []:
            if not isinstance(page, dict):
                continue
            for block in page.get("blocks") or []:
                if not isinstance(block, dict):
                    continue
                for para in block.get("paragraphs") or []:
                    if not isinstance(para, dict):
                        continue
                    for word in para.get("words") or []:
                        if not isinstance(word, dict):
                            continue
                        text = "".join(
                            (s or {}).get("text", "")
                            for s in (word.get("symbols") or [])
                        ).strip()
                        if not text:
                            continue
                        conf = word.get("confidence")
                        if conf is None:
                            continue
                        out.append({"token": text, "confidence": conf})
        return out or None
