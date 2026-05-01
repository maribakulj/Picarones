"""Adaptateur OCR — Mistral OCR (API vision Mistral AI).

Utilise l'API Mistral pour la reconnaissance de texte sur documents
patrimoniaux via le modèle multimodal Mistral.

Clé API : variable d'environnement ``MISTRAL_API_KEY``.

Documentation API : https://docs.mistral.ai/

Sprint 49 — exposition des token_confidences
---------------------------------------------
L'API ``/v1/ocr`` peut renvoyer des champs ``confidence`` au niveau
page, block, line ou word selon le modèle.  L'adapter parse la réponse
brute (``raw_response``) en plus du markdown : il cherche
récursivement les paires ``(text, confidence)`` exploitables et les
retourne au format Sprint 42.  Si la réponse ne contient aucun champ
de confidence (cas de l'API chat/vision pour ``pixtral-*``),
``token_confidences = None``.

Refactor du chantier 1 (post-Sprint 97)
---------------------------------------
L'adapter ne surcharge plus ``run()`` — il implémente ``_run_with_native``
et ``_extract_raw_confidences`` (les hooks factorisés dans ``BaseOCREngine``).
Comportement externe et octets de sortie strictement identiques.
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any, Optional

from picarones.engines.base import BaseOCREngine


logger = logging.getLogger(__name__)


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
    expose_confidences : bool
        ``True`` (défaut) : extrait les ``confidence`` de la réponse
        ``/v1/ocr`` quand elles sont présentes (Sprint 49). ``False`` :
        désactive complètement l'extraction.
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
        """API rétrocompat : retourne uniquement le texte."""
        text, _raw = self._run_with_native(image_path)
        return text

    def _run_with_native(
        self, image_path: Path,
    ) -> tuple[str, Optional[dict]]:
        """Hook framework (chantier 1) — délègue à ``_run_ocr_with_response``
        pour permettre aux tests Sprint 49 de monkeypatcher l'appel réseau
        sous son nom historique.
        """
        return self._run_ocr_with_response(image_path)

    def _run_ocr_with_response(
        self, image_path: Path,
    ) -> tuple[str, Optional[dict]]:
        """Exécute l'OCR et retourne ``(text, raw_response)``.

        ``raw_response`` est le JSON brut de l'API ``/v1/ocr`` (chemin
        natif) ou ``None`` (chemin chat/vision pour ``pixtral-*``).
        Centralisé pour que ``run()`` puisse extraire les
        ``token_confidences`` sans dupliquer la requête API.
        """
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
        return self._run_ocr_vision_api(image_url), None

    def _run_ocr_native_api(self, image_url: str) -> tuple[str, dict]:
        """Endpoint dédié /v1/ocr (pour mistral-ocr-latest et variantes).

        Retourne ``(text, raw_response_dict)`` pour permettre
        l'extraction des confidences en post-traitement.
        """
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
        text = "\n\n".join(p.get("markdown", "") for p in pages).strip()
        return text, data

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

    def _extract_raw_confidences(
        self, native: Any,
    ) -> Optional[list[dict[str, Any]]]:
        """Extrait les paires ``(token, confidence)`` de la réponse
        ``/v1/ocr`` quand elles existent.

        Mistral OCR peut exposer ``confidence`` à différents niveaux
        (page, block, line, word) selon le modèle.  L'extracteur
        cherche dans les structures suivantes en cascade :

        1. ``pages[i].words[j]`` avec ``{"text", "confidence"}``
        2. ``pages[i].lines[j]`` avec ``{"text", "confidence"}`` →
           propage la confidence aux mots de la ligne (comme Pero OCR
           Sprint 48)
        3. ``pages[i].blocks[j]`` avec ``{"text", "confidence"}`` →
           idem, propage à chaque mot

        Retourne ``None`` si aucun champ ``confidence`` exploitable
        n'est trouvé (cas le plus courant si l'API renvoie uniquement
        du markdown sans annotation, ou si on est sur le chemin
        chat/vision ``pixtral-*``).
        """
        if not self.config.get("expose_confidences", True):
            return None
        if not native or not isinstance(native, dict):
            return None
        out: list[dict[str, Any]] = []
        pages = native.get("pages") or []
        for page in pages:
            if not isinstance(page, dict):
                continue
            # Niveau 1 : words explicites
            for w in page.get("words") or []:
                self._maybe_emit_word(w, out)
            # Niveau 2 : lines avec confidence propagée
            for line in page.get("lines") or []:
                self._emit_lines_or_blocks(line, out)
            # Niveau 3 : blocks avec confidence propagée
            for block in page.get("blocks") or []:
                self._emit_lines_or_blocks(block, out)
        return out or None

    @staticmethod
    def _maybe_emit_word(word: Any, out: list) -> None:
        if not isinstance(word, dict):
            return
        text = (word.get("text") or "").strip()
        conf = word.get("confidence")
        if not text or conf is None:
            return
        out.append({"token": text, "confidence": conf})

    @staticmethod
    def _emit_lines_or_blocks(item: Any, out: list) -> None:
        """Pour une line/block, propage sa confidence à chaque mot."""
        if not isinstance(item, dict):
            return
        text = (item.get("text") or "").strip()
        conf = item.get("confidence")
        if not text or conf is None:
            return
        for word in text.split():
            if word:
                out.append({"token": word, "confidence": conf})

    def _extract_token_confidences_from_response(
        self, response: Any,
    ) -> Optional[list[dict[str, Any]]]:
        """Alias rétrocompat (Sprint 49) — extrait les confidences d'une réponse JSON.

        Wrapper qui chaîne ``_extract_raw_confidences`` puis
        ``_normalize_token_confidences`` (filtrage tokens vides / négatifs).
        """
        raw = self._extract_raw_confidences(response)
        return self._normalize_token_confidences(raw)
