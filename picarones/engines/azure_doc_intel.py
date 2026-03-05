"""Adaptateur OCR — Azure Document Intelligence (anciennement Form Recognizer).

Utilise l'API Azure Document Intelligence pour la reconnaissance de texte
dans des documents historiques.

Variables d'environnement requises :
  - ``AZURE_DOC_INTEL_KEY``      : clé API Azure
  - ``AZURE_DOC_INTEL_ENDPOINT`` : URL de l'endpoint (ex : https://moninstance.cognitiveservices.azure.com/)

Documentation : https://learn.microsoft.com/azure/ai-services/document-intelligence/
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from picarones.engines.base import BaseOCREngine


class AzureDocIntelEngine(BaseOCREngine):
    """Moteur OCR via Azure Document Intelligence.

    Configuration
    -------------
    model_id : str
        Modèle Azure à utiliser. Défaut : ``"prebuilt-read"`` (lecture générique).
        Alternatives : ``"prebuilt-document"``, ``"prebuilt-layout"``
        ou un modèle entraîné personnalisé.
    locale : str
        Paramètre de locale pour améliorer la précision (ex : ``"fr-FR"``).
    api_version : str
        Version de l'API Azure (défaut : ``"2024-02-29-preview"``).
    """

    @property
    def name(self) -> str:
        return "azure_doc_intel"

    def version(self) -> str:
        return self.config.get("api_version", "2024-02-29-preview")

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._api_key = os.environ.get("AZURE_DOC_INTEL_KEY")
        self._endpoint = (
            os.environ.get("AZURE_DOC_INTEL_ENDPOINT", "").rstrip("/")
            or self.config.get("endpoint", "").rstrip("/")
        )
        self._model_id: str = self.config.get("model_id", "prebuilt-read")
        self._locale: str = self.config.get("locale", "fr-FR")
        self._api_version: str = self.config.get("api_version", "2024-02-29-preview")

    def _run_ocr(self, image_path: Path) -> str:
        if not self._api_key:
            raise RuntimeError(
                "Clé API Azure manquante — définissez la variable d'environnement AZURE_DOC_INTEL_KEY"
            )
        if not self._endpoint:
            raise RuntimeError(
                "Endpoint Azure manquant — définissez la variable d'environnement AZURE_DOC_INTEL_ENDPOINT"
            )

        # Essai via SDK Azure si disponible, sinon REST direct
        try:
            return self._run_via_sdk(image_path)
        except ImportError:
            return self._run_via_rest(image_path)

    def _run_via_sdk(self, image_path: Path) -> str:
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential

        client = DocumentIntelligenceClient(
            endpoint=self._endpoint,
            credential=AzureKeyCredential(self._api_key),
        )
        with open(image_path, "rb") as f:
            poller = client.begin_analyze_document(
                model_id=self._model_id,
                body=f,
                locale=self._locale,
                content_type="application/octet-stream",
            )
        result = poller.result()
        return "\n".join(
            line.content
            for page in result.pages
            for line in (page.lines or [])
        )

    def _run_via_rest(self, image_path: Path) -> str:
        """Appel REST direct (sans SDK Azure)."""
        image_bytes = image_path.read_bytes()
        analyze_url = (
            f"{self._endpoint}/documentintelligence/documentModels/"
            f"{self._model_id}:analyze"
            f"?api-version={self._api_version}&locale={self._locale}"
        )

        # Soumettre l'image
        req = urllib.request.Request(
            analyze_url,
            data=image_bytes,
            headers={
                "Ocp-Apim-Subscription-Key": self._api_key,
                "Content-Type": "application/octet-stream",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                operation_url = resp.headers.get("Operation-Location", "")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"Azure Document Intelligence erreur {exc.code}: {exc.read().decode()}"
            ) from exc

        if not operation_url:
            raise RuntimeError("Azure : pas d'Operation-Location dans la réponse")

        # Polling du résultat (Azure est asynchrone)
        headers = {"Ocp-Apim-Subscription-Key": self._api_key}
        for attempt in range(30):
            time.sleep(1 + attempt * 0.5)
            poll_req = urllib.request.Request(operation_url, headers=headers)
            with urllib.request.urlopen(poll_req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            status = result.get("status", "")
            if status == "succeeded":
                return self._extract_text_from_result(result)
            if status in {"failed", "canceled"}:
                raise RuntimeError(f"Azure Document Intelligence : analyse {status}")
            # status == "running" → continuer à attendre

        raise RuntimeError("Azure Document Intelligence : timeout — analyse trop longue")

    @staticmethod
    def _extract_text_from_result(result: dict) -> str:
        """Extrait le texte brut depuis la réponse JSON Azure."""
        pages = result.get("analyzeResult", {}).get("pages", [])
        lines: list[str] = []
        for page in pages:
            for line in page.get("lines", []):
                content = line.get("content", "")
                if content:
                    lines.append(content)
        return "\n".join(lines)
