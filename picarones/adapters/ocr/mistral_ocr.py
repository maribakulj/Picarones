"""``MistralOCRAdapter`` natif — Sprint A14-S32.

Migration native du legacy ``picarones.engines.mistral_ocr.MistralOCREngine``
vers le contrat ``BaseOCRAdapter`` (S26).  **Pas un shim** : la classe
implémente directement le contrat du nouveau monde.

Le legacy ``MistralOCREngine`` reste en place jusqu'au S46.

Cas d'usage BnF
---------------
Mistral AI fournit deux familles d'OCR :

- **API dédiée ``/v1/ocr``** pour les modèles ``mistral-ocr-*`` —
  endpoint optimisé qui renvoie des pages structurées en markdown
  (et parfois des confidences mot par mot).
- **API vision/chat** pour les modèles ``pixtral-*`` —
  reconnaissance via prompt textuel + image base64.

L'adapter route automatiquement selon le nom du modèle.

Configuration
-------------
Constructeur :

- ``name`` (défaut ``"mistral_ocr"``) : identifiant de l'instance.
- ``model`` (défaut ``"mistral-ocr-latest"``) : modèle Mistral.
  - ``mistral-ocr-*`` → endpoint dédié ;
  - ``pixtral-*`` → API vision/chat.
- ``prompt`` : texte du prompt pour les modèles vision.  Défaut :
  instruction générique de transcription.
- ``max_tokens`` (défaut 4096) : limite tokens en sortie pour les
  modèles vision.
- ``api_key`` : clé API Mistral.  Si ``None`` (défaut), lit la
  variable d'environnement ``MISTRAL_API_KEY``.
- ``timeout_seconds`` (défaut 60) : timeout HTTP pour ``urllib``.

Comportement
------------
1. Vérifie présence d'un ``Artifact`` ``IMAGE`` avec URI valide.
2. Encode l'image en base64 + détecte ``image/...`` MIME selon
   l'extension.
3. Route vers ``/v1/ocr`` ou chat/vision selon ``model``.
4. Concatène le markdown / texte de toutes les pages.
5. Écrit dans ``<stem>.<name>.txt`` à côté de l'image.
6. Retourne un ``Artifact`` ``RAW_TEXT``.

Anti-sur-ingénierie
-------------------
- Pas de retry / backoff (le caller wrappe si besoin).
- Pas d'extraction de confidences (legacy S49 — reportées au
  sprint ``ConfidenceArtifact``).
- Pas de support multi-page (l'image est traitée comme une seule
  page d'entrée — Mistral OCR retourne une liste de pages dont on
  concatène les markdowns).
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

from picarones.adapters.ocr.base import BaseOCRAdapter, OCRAdapterError
from picarones.domain.artifacts import Artifact, ArtifactType


_DEFAULT_PROMPT = (
    "Transcris fidèlement le texte visible sur cette image de document "
    "historique. Retourne uniquement le texte, sans commentaire."
)


_MEDIA_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
}


class MistralOCRAdapter(BaseOCRAdapter):
    """Adapter Mistral OCR natif au contrat S26.

    Parameters
    ----------
    name:
        Identifiant lisible.  Défaut ``"mistral_ocr"``.
    model:
        Modèle Mistral.  ``mistral-ocr-*`` → API dédiée ``/v1/ocr``,
        ``pixtral-*`` → API vision/chat.  Défaut ``"mistral-ocr-latest"``.
    prompt:
        Prompt pour les modèles vision.
    max_tokens:
        Limite tokens en sortie pour les modèles vision.  Défaut 4096.
    api_key:
        Clé API Mistral.  Si ``None`` (défaut), lit
        ``MISTRAL_API_KEY``.
    timeout_seconds:
        Timeout HTTP pour les appels ``urllib``.  Défaut 60.

    Raises
    ------
    OCRAdapterError
        Si ``name`` est invalide au constructeur.
    """

    input_types = frozenset({ArtifactType.IMAGE})
    output_types = frozenset({ArtifactType.RAW_TEXT})
    execution_mode = "io"

    def __init__(
        self,
        *,
        name: str = "mistral_ocr",
        model: str = "mistral-ocr-latest",
        prompt: str = _DEFAULT_PROMPT,
        max_tokens: int = 4096,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not name or not name.strip():
            raise OCRAdapterError(
                "MistralOCRAdapter : name vide non autorisé.",
            )
        if not all(c.isalnum() or c in "_-" for c in name):
            raise OCRAdapterError(
                f"MistralOCRAdapter : name invalide {name!r} — "
                "alphanumérique + _ - uniquement.",
            )
        if max_tokens <= 0:
            raise OCRAdapterError(
                f"MistralOCRAdapter : max_tokens doit être > 0, "
                f"reçu {max_tokens}.",
            )
        if timeout_seconds <= 0:
            raise OCRAdapterError(
                f"MistralOCRAdapter : timeout_seconds doit être > 0, "
                f"reçu {timeout_seconds}.",
            )
        self._name = name
        self._model = model
        self._prompt = prompt
        self._max_tokens = max_tokens
        self._explicit_api_key = api_key
        self._timeout = timeout_seconds

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    def _resolve_api_key(self) -> str:
        """Résout la clé API : explicite > env var.

        Lève ``OCRAdapterError`` si aucune clé n'est disponible.
        """
        key = self._explicit_api_key or os.environ.get("MISTRAL_API_KEY")
        if not key:
            raise OCRAdapterError(
                f"{self.name} : clé API Mistral manquante. "
                "Définir MISTRAL_API_KEY ou passer api_key= au "
                "constructeur.",
            )
        return key

    def _encode_image(self, image_path: Path) -> str:
        """Retourne ``data:<mime>;base64,<...>`` pour l'image."""
        suffix = image_path.suffix.lower()
        media_type = _MEDIA_TYPES.get(suffix, "image/jpeg")
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:{media_type};base64,{image_b64}"

    def execute(
        self,
        inputs: dict[ArtifactType, Artifact],
        params: dict[str, Any],
        context: Any,
    ) -> dict[ArtifactType, Artifact]:
        """Exécute Mistral OCR sur l'image fournie.

        Route vers l'API appropriée selon ``self.model`` :
        - ``mistral-ocr-*`` → ``/v1/ocr`` via ``urllib`` ;
        - ``pixtral-*`` → API chat/vision via SDK ``mistralai``.

        Raises
        ------
        OCRAdapterError
            Erreur d'input, clé manquante, SDK absent (pour pixtral),
            ou API Mistral en erreur.
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

        api_key = self._resolve_api_key()
        image_url = self._encode_image(image_path)

        # Le préfixe ``mistral-ocr-*`` est documenté par Mistral pour
        # l'API dédiée ``/v1/ocr``.  Tout autre nom (``pixtral-*``,
        # etc.) bascule sur l'API chat/vision.  Match strict par
        # préfixe pour éviter qu'un modèle exotique nommé
        # ``pixtral-MISTRAL-OCR-fancy`` ne soit confondu.
        if self._model.lower().startswith("mistral-ocr"):
            text = self._call_native_ocr_api(image_url, api_key)
        else:
            text = self._call_chat_vision_api(image_url, api_key)

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
    # API natives
    # ──────────────────────────────────────────────────────────────

    def _call_native_ocr_api(self, image_url: str, api_key: str) -> str:
        """Appelle ``POST /v1/ocr`` via urllib et retourne le markdown
        concaténé."""
        payload = json.dumps({
            "model": self._model,
            "document": {"type": "image_url", "image_url": image_url},
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.mistral.ai/v1/ocr",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:
            raise OCRAdapterError(
                f"{self.name} : erreur API Mistral /v1/ocr : "
                f"{type(exc).__name__}: {exc}",
            ) from exc
        pages = data.get("pages", [])
        text = "\n\n".join(p.get("markdown", "") for p in pages).strip()
        return text

    def _call_chat_vision_api(self, image_url: str, api_key: str) -> str:
        """Appelle l'API chat/vision Mistral via le SDK ``mistralai``."""
        try:
            try:
                from mistralai.client import Mistral
            except ImportError:
                from mistralai import Mistral  # type: ignore[no-redef]
        except ImportError as exc:
            raise OCRAdapterError(
                f"{self.name} : SDK 'mistralai' non installé. "
                "Installer avec : pip install mistralai",
            ) from exc

        client = Mistral(api_key=api_key)
        try:
            response = client.chat.complete(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self._prompt},
                            {"type": "image_url", "image_url": image_url},
                        ],
                    },
                ],
                max_tokens=self._max_tokens,
            )
        except Exception as exc:
            raise OCRAdapterError(
                f"{self.name} : erreur API Mistral chat : "
                f"{type(exc).__name__}: {exc}",
            ) from exc

        # Mistral peut retourner ``content`` sous forme de
        # ``list[ContentChunk]`` au lieu de ``str``.  Le helper
        # ``normalize_llm_content`` gère les deux formats.
        from picarones.adapters.llm.base import normalize_llm_content

        try:
            raw_content = response.choices[0].message.content
        except (AttributeError, IndexError) as exc:
            raise OCRAdapterError(
                f"{self.name} : réponse Mistral chat malformée : {exc}",
            ) from exc

        return normalize_llm_content(raw_content) or ""


__all__ = ["MistralOCRAdapter"]
