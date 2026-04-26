"""Adaptateur OCR — Azure Document Intelligence (anciennement Form Recognizer).

Utilise l'API Azure Document Intelligence pour la reconnaissance de texte
dans des documents historiques.

Variables d'environnement requises :
  - ``AZURE_DOC_INTEL_KEY``      : clé API Azure
  - ``AZURE_DOC_INTEL_ENDPOINT`` : URL de l'endpoint (ex : https://moninstance.cognitiveservices.azure.com/)

Documentation : https://learn.microsoft.com/azure/ai-services/document-intelligence/

Sprint 51 — exposition des token_confidences
---------------------------------------------
La réponse Azure expose ``analyzeResult.pages[].words[]`` avec
``content`` et ``confidence`` (∈ [0, 1]).  L'adapter parcourt cette
hiérarchie et émet une entrée par mot au format Sprint 42.

Le texte ``EngineResult.text`` est extrait depuis ``pages[].lines[]``
(préservation rétrocompat octet par octet).  Les deux chemins (SDK et
REST) sont normalisés vers une représentation dict unifiée.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

from picarones.engines.base import BaseOCREngine, EngineResult


logger = logging.getLogger(__name__)


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
    expose_confidences : bool
        ``True`` (défaut) : extrait ``Word.confidence`` de la réponse
        Azure (Sprint 51).
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
        """API rétrocompat : retourne uniquement le texte."""
        text, _result = self._run_ocr_with_result(image_path)
        return text

    def _run_ocr_with_result(
        self, image_path: Path,
    ) -> tuple[str, Optional[dict]]:
        """Exécute l'OCR et retourne ``(text, analyze_result_dict)``.

        ``analyze_result_dict`` est la sous-structure
        ``analyzeResult`` (avec ``pages[].words[]`` portant les
        confidences) — normalisée entre les chemins SDK et REST.
        """
        if not self._api_key:
            raise RuntimeError(
                "Clé API Azure manquante — définissez la variable d'environnement AZURE_DOC_INTEL_KEY"
            )
        if not self._endpoint:
            raise RuntimeError(
                "Endpoint Azure manquant — définissez la variable d'environnement AZURE_DOC_INTEL_ENDPOINT"
            )

        try:
            return self._run_via_sdk(image_path)
        except ImportError:
            return self._run_via_rest(image_path)

    def _run_via_sdk(self, image_path: Path) -> tuple[str, dict]:
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
        text = "\n".join(
            line.content
            for page in result.pages
            for line in (page.lines or [])
        )
        analyze_result = self._sdk_result_to_dict(result)
        return text, analyze_result

    def _run_via_rest(self, image_path: Path) -> tuple[str, Optional[dict]]:
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
                text = self._extract_text_from_result(result)
                analyze_result = result.get("analyzeResult") or None
                return text, analyze_result
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

    # ──────────────────────────────────────────────────────────────────
    # Conversion SDK → dict normalisé
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _sdk_result_to_dict(result: Any) -> dict:
        """Convertit l'objet SDK en dict ``{"pages": [{"words":
        [{"content", "confidence"}]}]}`` pour traitement uniforme avec
        le chemin REST."""
        pages = []
        for page in getattr(result, "pages", []) or []:
            words = []
            for word in getattr(page, "words", []) or []:
                content = getattr(word, "content", "") or ""
                conf = getattr(word, "confidence", None)
                words.append({
                    "content": content,
                    "confidence": float(conf) if conf is not None else None,
                })
            pages.append({"words": words})
        return {"pages": pages}

    # ──────────────────────────────────────────────────────────────────
    # Extraction des token_confidences au format Sprint 42
    # ──────────────────────────────────────────────────────────────────

    def _extract_token_confidences_from_result(
        self, analyze_result: Optional[dict],
    ) -> Optional[list[dict[str, Any]]]:
        """Parcourt ``pages[].words[]`` et émet
        ``{"token": str, "confidence": float}`` par mot.

        Filtrage cohérent avec les autres adapters : confidence None /
        négative ignorée, contenu vide ignoré.

        Retourne ``None`` si :
        - ``analyze_result`` est ``None``,
        - ``expose_confidences=False``,
        - aucun mot exploitable n'a été trouvé (cas peu probable —
          Azure expose presque toujours les words).
        """
        if not self.config.get("expose_confidences", True):
            return None
        if not analyze_result or not isinstance(analyze_result, dict):
            return None
        try:
            out: list[dict[str, Any]] = []
            for page in analyze_result.get("pages") or []:
                if not isinstance(page, dict):
                    continue
                for word in page.get("words") or []:
                    if not isinstance(word, dict):
                        continue
                    content = (word.get("content") or "").strip()
                    if not content:
                        continue
                    conf = word.get("confidence")
                    if conf is None:
                        continue
                    try:
                        conf_val = float(conf)
                    except (TypeError, ValueError):
                        continue
                    if conf_val < 0:
                        continue
                    out.append({"token": content, "confidence": conf_val})
            return out or None
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[azure_doc_intel] extraction des token_confidences dégradée : %s",
                exc,
            )
            return None

    def run(self, image_path: str | Path) -> EngineResult:
        """Exécute Azure Document Intelligence et expose les
        ``Word.confidence`` natifs (Sprint 51).

        L'API est appelée une seule fois (avec polling pour Azure
        asynchrone) ; texte et structure ``analyzeResult`` sont
        récupérés ensemble.  Aucun overhead vs l'implémentation
        historique.
        """
        image_path = Path(image_path)
        start = time.perf_counter()
        text = ""
        error: Optional[str] = None
        token_confidences: Optional[list[dict[str, Any]]] = None
        try:
            text, analyze_result = self._run_ocr_with_result(image_path)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        else:
            token_confidences = self._extract_token_confidences_from_result(
                analyze_result,
            )
        duration = time.perf_counter() - start
        return EngineResult(
            engine_name=self.name,
            image_path=str(image_path),
            text=text,
            duration_seconds=round(duration, 4),
            error=error,
            metadata={"engine_version": self._safe_version()},
            token_confidences=token_confidences,
        )
