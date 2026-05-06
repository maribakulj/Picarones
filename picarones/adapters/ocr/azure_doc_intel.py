"""``AzureDocIntelAdapter`` natif — Sprint A14-S34.

Migration native du legacy ``picarones.engines.azure_doc_intel`` vers
``BaseOCRAdapter`` (S26).  **Pas un shim**.

Le legacy reste en place jusqu'au S46.

Cas d'usage BnF
---------------
Azure Document Intelligence (anciennement Form Recognizer) propose
plusieurs modèles préentraînés :

- ``prebuilt-read`` (défaut) : lecture générique optimisée pour les
  documents textuels denses.
- ``prebuilt-document`` : extraction layout + champs.
- ``prebuilt-layout`` : analyse de mise en page.
- modèles personnalisés entraînés.

L'API est asynchrone : on poste l'image et on poll un endpoint
status jusqu'à obtenir le résultat.

L'adapter route automatiquement vers SDK
(``azure-ai-documentintelligence``) si disponible, sinon REST
direct via ``urllib`` (avec polling).

Configuration
-------------
Constructeur :

- ``name`` (défaut ``"azure_doc_intel"``).
- ``endpoint`` : URL de l'endpoint (overrides
  ``AZURE_DOC_INTEL_ENDPOINT``).
- ``api_key`` : clé API (overrides ``AZURE_DOC_INTEL_KEY``).
- ``model_id`` (défaut ``"prebuilt-read"``).
- ``locale`` (défaut ``"fr-FR"``).
- ``api_version`` (défaut ``"2024-02-29-preview"``).
- ``timeout_seconds`` (défaut 60) : timeout par requête HTTP.
- ``max_polling_attempts`` (défaut 30) : nombre max de polls REST.
- ``polling_interval_base`` (défaut 1.0) : intervalle de base entre
  polls (incrémenté de 0.5s par tentative — backoff linéaire
  identique au legacy).

Comportement
------------
1. Valide IMAGE input.
2. Résout endpoint + api_key (explicite > env).
3. Tente le SDK ; sur ImportError, fallback REST.
4. Pour le REST : POST → Operation-Location → poll jusqu'à
   ``succeeded`` / ``failed`` / ``canceled``.
5. Extrait le texte ligne par ligne dans l'ordre pages × lines.
6. Écrit dans ``<stem>.<name>.txt`` à côté de l'image.

Anti-sur-ingénierie
-------------------
- Pas d'extraction de confidences (legacy S51 — reportée).
- Pas de support multi-langue dans une même requête.
- Pas de retry au-delà du polling (qui est un retry implicite).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from picarones.adapters.ocr.base import BaseOCRAdapter, OCRAdapterError
from picarones.domain.artifacts import Artifact, ArtifactType


class AzureDocIntelAdapter(BaseOCRAdapter):
    """Adapter Azure Document Intelligence natif au contrat S26.

    Parameters
    ----------
    name:
        Identifiant lisible.  Défaut ``"azure_doc_intel"``.
    endpoint:
        URL Azure (override ``AZURE_DOC_INTEL_ENDPOINT``).
    api_key:
        Clé API Azure (override ``AZURE_DOC_INTEL_KEY``).
    model_id:
        ``"prebuilt-read"`` (défaut), ``"prebuilt-document"``,
        ``"prebuilt-layout"``, ou un modèle entraîné personnalisé.
    locale:
        Locale Azure (défaut ``"fr-FR"``).
    api_version:
        Version d'API Azure (défaut ``"2024-02-29-preview"``).
    timeout_seconds:
        Timeout HTTP (défaut 60).
    max_polling_attempts:
        Nombre max de polls REST (défaut 30).
    polling_interval_base:
        Intervalle de base entre polls (défaut 1.0s, +0.5s/attempt).

    Raises
    ------
    OCRAdapterError
        Au constructeur si name invalide ou paramètres hors plage.
    """

    input_types = frozenset({ArtifactType.IMAGE})
    output_types = frozenset({ArtifactType.RAW_TEXT})
    execution_mode = "io"

    def __init__(
        self,
        *,
        name: str = "azure_doc_intel",
        endpoint: str | None = None,
        api_key: str | None = None,
        model_id: str = "prebuilt-read",
        locale: str = "fr-FR",
        api_version: str = "2024-02-29-preview",
        timeout_seconds: float = 60.0,
        max_polling_attempts: int = 30,
        polling_interval_base: float = 1.0,
    ) -> None:
        if not name or not name.strip():
            raise OCRAdapterError(
                "AzureDocIntelAdapter : name vide non autorisé.",
            )
        if not all(c.isalnum() or c in "_-" for c in name):
            raise OCRAdapterError(
                f"AzureDocIntelAdapter : name invalide {name!r} — "
                "alphanumérique + _ - uniquement.",
            )
        if timeout_seconds <= 0:
            raise OCRAdapterError(
                f"AzureDocIntelAdapter : timeout_seconds doit être > 0, "
                f"reçu {timeout_seconds}.",
            )
        if max_polling_attempts <= 0:
            raise OCRAdapterError(
                f"AzureDocIntelAdapter : max_polling_attempts doit être "
                f"> 0, reçu {max_polling_attempts}.",
            )
        if polling_interval_base < 0:
            raise OCRAdapterError(
                f"AzureDocIntelAdapter : polling_interval_base doit être "
                f">= 0, reçu {polling_interval_base}.",
            )
        self._name = name
        self._explicit_endpoint = endpoint
        self._explicit_api_key = api_key
        self._model_id = model_id
        self._locale = locale
        self._api_version = api_version
        self._timeout = timeout_seconds
        self._max_polling_attempts = max_polling_attempts
        self._polling_base = polling_interval_base

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_id(self) -> str:
        return self._model_id

    def _resolve_api_key(self) -> str:
        key = self._explicit_api_key or os.environ.get("AZURE_DOC_INTEL_KEY")
        if not key:
            raise OCRAdapterError(
                f"{self.name} : clé API Azure manquante. Définir "
                "AZURE_DOC_INTEL_KEY ou passer api_key= au constructeur.",
            )
        return key

    def _resolve_endpoint(self) -> str:
        endpoint = (
            self._explicit_endpoint
            or os.environ.get("AZURE_DOC_INTEL_ENDPOINT", "")
        ).rstrip("/")
        if not endpoint:
            raise OCRAdapterError(
                f"{self.name} : endpoint Azure manquant. Définir "
                "AZURE_DOC_INTEL_ENDPOINT ou passer endpoint= au "
                "constructeur.",
            )
        return endpoint

    def execute(
        self,
        inputs: dict[ArtifactType, Artifact],
        params: dict[str, Any],
        context: Any,
    ) -> dict[ArtifactType, Artifact]:
        if ArtifactType.IMAGE not in inputs:
            raise OCRAdapterError(
                f"{self.name} : input IMAGE manquant.",
            )
        image_artifact = inputs[ArtifactType.IMAGE]
        if image_artifact.uri is None:
            raise OCRAdapterError(
                f"{self.name} : artefact image "
                f"{image_artifact.id!r} sans URI.",
            )
        image_path = Path(image_artifact.uri)
        if not image_path.exists():
            raise OCRAdapterError(
                f"{self.name} : image introuvable {image_path!r}.",
            )

        api_key = self._resolve_api_key()
        endpoint = self._resolve_endpoint()

        # On tente le SDK d'abord ; sur ImportError, fallback REST.
        try:
            text = self._call_via_sdk(image_path, endpoint, api_key)
        except _SDKMissing:
            text = self._call_via_rest(image_path, endpoint, api_key)

        text_path = (
            image_path.parent / f"{image_path.stem}.{self.name}.txt"
        )
        text_path.write_text(text, encoding="utf-8")

        return {
            ArtifactType.RAW_TEXT: Artifact(
                id=f"{context.document_id}:{self.name}:raw_text",
                document_id=context.document_id,
                type=ArtifactType.RAW_TEXT,
                produced_by_step="ocr",
                uri=str(text_path),
            ),
        }

    # ──────────────────────────────────────────────────────────────
    # SDK
    # ──────────────────────────────────────────────────────────────

    def _call_via_sdk(
        self, image_path: Path, endpoint: str, api_key: str,
    ) -> str:
        try:
            from azure.ai.documentintelligence import (
                DocumentIntelligenceClient,
            )
            from azure.core.credentials import AzureKeyCredential
        except ImportError as exc:
            raise _SDKMissing() from exc

        try:
            client = DocumentIntelligenceClient(
                endpoint=endpoint,
                credential=AzureKeyCredential(api_key),
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
        except _SDKMissing:
            raise
        except Exception as exc:
            raise OCRAdapterError(
                f"{self.name} : SDK Azure a levé : "
                f"{type(exc).__name__}: {exc}",
            ) from exc
        return text

    # ──────────────────────────────────────────────────────────────
    # REST avec polling
    # ──────────────────────────────────────────────────────────────

    def _call_via_rest(
        self, image_path: Path, endpoint: str, api_key: str,
    ) -> str:
        image_bytes = image_path.read_bytes()
        analyze_url = (
            f"{endpoint}/documentintelligence/documentModels/"
            f"{self._model_id}:analyze"
            f"?api-version={self._api_version}&locale={self._locale}"
        )
        req = urllib.request.Request(
            analyze_url,
            data=image_bytes,
            headers={
                "Ocp-Apim-Subscription-Key": api_key,
                "Content-Type": "application/octet-stream",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                operation_url = resp.headers.get("Operation-Location", "")
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8")
            except Exception:  # noqa: BLE001
                pass
            raise OCRAdapterError(
                f"{self.name} : Azure Document Intelligence erreur "
                f"{exc.code} : {body}",
            ) from exc
        except Exception as exc:
            raise OCRAdapterError(
                f"{self.name} : erreur API Azure : "
                f"{type(exc).__name__}: {exc}",
            ) from exc

        if not operation_url:
            raise OCRAdapterError(
                f"{self.name} : Azure n'a pas retourné Operation-Location.",
            )

        # Polling du résultat (Azure asynchrone).
        headers = {"Ocp-Apim-Subscription-Key": api_key}
        for attempt in range(self._max_polling_attempts):
            time.sleep(self._polling_base + attempt * 0.5)
            poll_req = urllib.request.Request(operation_url, headers=headers)
            try:
                with urllib.request.urlopen(
                    poll_req, timeout=self._timeout,
                ) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
            except Exception as exc:
                raise OCRAdapterError(
                    f"{self.name} : erreur de polling Azure : "
                    f"{type(exc).__name__}: {exc}",
                ) from exc
            status = result.get("status", "")
            if status == "succeeded":
                return self._extract_text_from_rest_result(result)
            if status in {"failed", "canceled"}:
                raise OCRAdapterError(
                    f"{self.name} : analyse Azure {status} : "
                    f"{result.get('error', {})}",
                )
            # running → continue
        raise OCRAdapterError(
            f"{self.name} : timeout polling Azure après "
            f"{self._max_polling_attempts} tentatives.",
        )

    @staticmethod
    def _extract_text_from_rest_result(result: dict) -> str:
        pages = result.get("analyzeResult", {}).get("pages", [])
        lines: list[str] = []
        for page in pages:
            for line in page.get("lines", []):
                content = line.get("content", "")
                if content:
                    lines.append(content)
        return "\n".join(lines)


class _SDKMissing(Exception):
    """Sentinel interne pour signaler que le SDK Azure n'est pas
    installé.  Capturé par ``execute`` pour fallback REST.

    Ne fuit jamais au caller — c'est un détail d'implémentation.
    """


__all__ = ["AzureDocIntelAdapter"]
