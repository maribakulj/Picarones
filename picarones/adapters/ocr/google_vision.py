"""``GoogleVisionAdapter`` natif — Sprint A14-S33.

Cas d'usage BnF
---------------
Google Cloud Vision propose deux modes d'OCR :

- ``DOCUMENT_TEXT_DETECTION`` (défaut) : optimisé pour les textes
  denses et multilinguistiques — retourne une ``fullTextAnnotation``
  hiérarchique (pages → blocks → paragraphs → words → symbols) avec
  un texte plat ``text``.
- ``TEXT_DETECTION`` : mode court, retourne uniquement les
  ``textAnnotations[0].description``.

L'adapter route automatiquement vers SDK (auth service account) ou
REST direct (auth clé API) selon la configuration disponible.

Configuration
-------------
Constructeur :

- ``name`` (défaut ``"google_vision"``).
- ``language_hints`` (défaut ``["fr"]``) : suggestions Vision API.
- ``feature_type`` (défaut ``"DOCUMENT_TEXT_DETECTION"``).
- ``api_key`` : clé API Google.  Si ``None``, lit ``GOOGLE_API_KEY``.
- ``credentials_path`` : chemin vers un service account JSON.  Si
  ``None``, lit ``GOOGLE_APPLICATION_CREDENTIALS``.
- ``timeout_seconds`` (défaut 60).

Au moins une des deux authentifications (SDK ou REST) doit être
disponible.

Anti-sur-ingénierie
-------------------
- Pas d'extraction de confidences (à ajouter quand un caller en aura besoin).
- Pas de pré-validation du JSON service account — le SDK le fait.
- Pas de support batch — un appel par image.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from picarones.adapters._retry import call_with_retry
from picarones.adapters.ocr.base import BaseOCRAdapter, OCRAdapterError
from picarones.adapters.output_paths import resolve_output_path
from picarones.domain.artifacts import Artifact, ArtifactType


_VALID_FEATURE_TYPES = frozenset({"DOCUMENT_TEXT_DETECTION", "TEXT_DETECTION"})


class GoogleVisionAdapter(BaseOCRAdapter):
    """Adapter Google Cloud Vision natif au contrat S26.

    Parameters
    ----------
    name:
        Identifiant lisible.  Défaut ``"google_vision"``.
    language_hints:
        Suggestions Vision API.  Défaut ``["fr"]``.
    feature_type:
        ``"DOCUMENT_TEXT_DETECTION"`` (défaut) ou ``"TEXT_DETECTION"``.
    api_key:
        Clé API explicite.  Si ``None``, lit ``GOOGLE_API_KEY``.
    credentials_path:
        Chemin service account JSON explicite.  Si ``None``, lit
        ``GOOGLE_APPLICATION_CREDENTIALS``.
    timeout_seconds:
        Timeout HTTP (REST).  Défaut 60.

    Raises
    ------
    OCRAdapterError
        Au constructeur si name ou feature_type invalides.
    """

    input_types = frozenset({ArtifactType.IMAGE})
    output_types = frozenset({ArtifactType.RAW_TEXT})
    execution_mode = "io"

    def __init__(
        self,
        *,
        name: str = "google_vision",
        language_hints: list[str] | None = None,
        feature_type: str = "DOCUMENT_TEXT_DETECTION",
        api_key: str | None = None,
        credentials_path: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not name or not name.strip():
            raise OCRAdapterError(
                "GoogleVisionAdapter : name vide non autorisé.",
            )
        if not all(c.isalnum() or c in "_-" for c in name):
            raise OCRAdapterError(
                f"GoogleVisionAdapter : name invalide {name!r} — "
                "alphanumérique + _ - uniquement.",
            )
        if feature_type not in _VALID_FEATURE_TYPES:
            raise OCRAdapterError(
                f"GoogleVisionAdapter : feature_type invalide "
                f"{feature_type!r}.  Valeurs valides : "
                f"{sorted(_VALID_FEATURE_TYPES)}.",
            )
        if timeout_seconds <= 0:
            raise OCRAdapterError(
                f"GoogleVisionAdapter : timeout_seconds doit être > 0, "
                f"reçu {timeout_seconds}.",
            )
        self._name = name
        self._language_hints = list(language_hints or ["fr"])
        self._feature_type = feature_type
        self._explicit_api_key = api_key
        self._explicit_credentials = credentials_path
        self._timeout = timeout_seconds

    @property
    def name(self) -> str:
        return self._name

    @property
    def feature_type(self) -> str:
        return self._feature_type

    def _resolve_credentials_path(self) -> str | None:
        return self._explicit_credentials or os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS",
        )

    def _resolve_api_key(self) -> str | None:
        return self._explicit_api_key or os.environ.get("GOOGLE_API_KEY")

    def execute(
        self,
        inputs: dict[ArtifactType, Artifact],
        params: dict[str, Any],
        context: Any,
    ) -> dict[ArtifactType, Artifact]:
        """Exécute Google Vision OCR sur l'image fournie.

        Routing :

        - Si un service account JSON est disponible
          (``credentials_path`` ou ``GOOGLE_APPLICATION_CREDENTIALS``)
          → passe par le SDK ``google-cloud-vision``.
        - Sinon, si une clé API simple est disponible
          (``api_key`` ou ``GOOGLE_API_KEY``) → passe par REST direct
          via ``urllib``.
        - Sinon → ``OCRAdapterError``.
        """
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

        creds = self._resolve_credentials_path()
        api_key = self._resolve_api_key()

        if creds:
            text = self._call_via_sdk(image_path)
        elif api_key:
            text = self._call_via_rest(image_path, api_key)
        else:
            raise OCRAdapterError(
                f"{self.name} : authentification manquante. Définir "
                "GOOGLE_APPLICATION_CREDENTIALS (service account JSON) "
                "ou GOOGLE_API_KEY.",
            )

        text_path = resolve_output_path(
            input_path=image_path,
            adapter_name=self.name,
            suffix="txt",
            context=context,
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
    # SDK / REST
    # ──────────────────────────────────────────────────────────────

    def _call_via_sdk(self, image_path: Path) -> str:
        try:
            from google.cloud import vision
        except ImportError as exc:
            raise OCRAdapterError(
                f"{self.name} : SDK google-cloud-vision non installé. "
                "Installer avec : pip install google-cloud-vision",
            ) from exc

        try:
            client = vision.ImageAnnotatorClient()
            image = vision.Image(content=image_path.read_bytes())
            ctx = vision.ImageContext(language_hints=self._language_hints)

            if self._feature_type == "DOCUMENT_TEXT_DETECTION":
                response = client.document_text_detection(
                    image=image, image_context=ctx,
                )
                text = response.full_text_annotation.text
            else:
                response = client.text_detection(
                    image=image, image_context=ctx,
                )
                texts = response.text_annotations
                text = texts[0].description if texts else ""
        except Exception as exc:
            raise OCRAdapterError(
                f"{self.name} : SDK Google Vision a levé : "
                f"{type(exc).__name__}: {exc}",
            ) from exc
        return text

    def _call_via_rest(self, image_path: Path, api_key: str) -> str:
        image_b64 = base64.b64encode(
            image_path.read_bytes(),
        ).decode("ascii")
        payload = json.dumps({
            "requests": [{
                "image": {"content": image_b64},
                "features": [
                    {"type": self._feature_type, "maxResults": 1},
                ],
                "imageContext": {"languageHints": self._language_hints},
            }],
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://vision.googleapis.com/v1/images:annotate",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": api_key,
            },
        )
        def _do_call() -> dict:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))

        try:
            result = call_with_retry(_do_call, label=self.name)
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8")
            except Exception:  # noqa: BLE001
                pass
            raise OCRAdapterError(
                f"{self.name} : Google Vision API erreur {exc.code} : {body}",
            ) from exc
        except Exception as exc:
            raise OCRAdapterError(
                f"{self.name} : erreur API Google Vision : "
                f"{type(exc).__name__}: {exc}",
            ) from exc

        responses = result.get("responses", [{}])
        if not responses:
            return ""
        r = responses[0]
        if "error" in r:
            raise OCRAdapterError(
                f"{self.name} : Google Vision API erreur : {r['error']}",
            )

        if self._feature_type == "DOCUMENT_TEXT_DETECTION":
            full = r.get("fullTextAnnotation") or {}
            return full.get("text", "") if isinstance(full, dict) else ""
        # TEXT_DETECTION
        texts = r.get("textAnnotations", [])
        return texts[0]["description"] if texts else ""


__all__ = ["GoogleVisionAdapter"]
