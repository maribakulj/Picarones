"""``CalamariAdapter`` — adapter pour Calamari OCR.

Implémente le contrat ``BaseOCRAdapter`` (couche 5) :
``execute(inputs, params, context) → dict[ArtifactType, Artifact]``.

Cas d'usage BnF
---------------
Calamari est un OCR open-source basé TensorFlow / Keras, conçu pour
les imprimés historiques et la transcription ligne par ligne.
Modèles disponibles via OCR-D, Wikisource, et le hub Calamari.
Particulièrement performant en ensemble (vote multi-modèles).

Configuration
-------------
Constructeur :

- ``name`` (défaut ``"calamari"``) : identifiant de l'instance.
- ``checkpoint`` (obligatoire) : chemin vers le modèle Calamari
  (fichier ``.ckpt`` ou répertoire de modèles pour le voting).
- ``voter`` (défaut ``"confidence_voter_default_ctc"``) : stratégie
  de vote quand plusieurs modèles sont passés en ensemble.
- ``batch_size`` (défaut ``1``) : taille de batch pour l'inférence
  ligne par ligne.  ``1`` privilégie la simplicité ; augmenter pour
  un gain de débit GPU.

Comportement
------------
1. Vérifie qu'un ``Artifact`` ``IMAGE`` est présent.
2. Lazy-import de ``calamari_ocr`` et ``PIL``.
3. Charge le ``Predictor`` (cache par instance).
4. Calamari attend des **lignes** d'image, pas des pages.  L'adapter
   ne fait pas de segmentation : il OCRise l'image entière comme
   une ligne unique.  Pour un workflow page → lignes, l'utilisateur
   doit pré-segmenter (Kraken pageseg ou OCR-D segmenter) et appeler
   Calamari sur chaque ligne séparément — futur enrichissement à
   prévoir quand un consommateur en aura besoin.
5. Écrit la prédiction dans ``<stem>.<name>.txt``.

Anti-sur-ingénierie
-------------------
- Pas de segmentation embarquée — Calamari est un *line recognizer*,
  pas un page OCR.  L'utilisateur compose avec un segmenter externe
  s'il a besoin du flux page → lignes.
- Pas de confidences pour l'instant — Calamari expose
  ``Prediction.avg_char_probability`` qui pourra alimenter un
  ``CONFIDENCES`` artifact dans une itération future.
- Modèle chargé une fois par instance, partagé entre appels successifs
  (Predictor TensorFlow non recréé à chaque image).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from picarones.adapters.ocr.base import BaseOCRAdapter, OCRAdapterError
from picarones.adapters.output_paths import resolve_output_path
from picarones.domain.artifacts import Artifact, ArtifactType


class CalamariAdapter(BaseOCRAdapter):
    """Adapter Calamari OCR (Sprint Phase 3 — chantier post-rewrite).

    Parameters
    ----------
    name:
        Identifiant lisible de l'instance.  Défaut ``"calamari"``.
        Doit être alphanumérique + ``_-``.
    checkpoint:
        Chemin vers le checkpoint Calamari (``.ckpt`` ou dossier de
        modèles pour ensemble voting).  **Obligatoire** : Calamari
        n'embarque pas de modèle par défaut.
    voter:
        Nom de la stratégie de vote pour ensembles multi-modèles.
        Défaut ``"confidence_voter_default_ctc"``.
    batch_size:
        Taille de batch pour l'inférence.  Défaut 1.

    Raises
    ------
    OCRAdapterError
        Si ``name`` invalide, ``checkpoint`` vide ou inexistant.
    """

    input_types = frozenset({ArtifactType.IMAGE})
    output_types = frozenset({ArtifactType.RAW_TEXT})
    execution_mode = "cpu"

    def __init__(
        self,
        *,
        name: str = "calamari",
        checkpoint: str | Path | None = None,
        voter: str = "confidence_voter_default_ctc",
        batch_size: int = 1,
    ) -> None:
        if not name or not name.strip():
            raise OCRAdapterError(
                "CalamariAdapter : name vide non autorisé.",
            )
        if not all(c.isalnum() or c in "_-" for c in name):
            raise OCRAdapterError(
                f"CalamariAdapter : name invalide {name!r} — "
                "alphanumérique + _ - uniquement.",
            )
        if not checkpoint:
            raise OCRAdapterError(
                "CalamariAdapter : checkpoint est obligatoire — "
                "Calamari n'embarque pas de modèle par défaut.  "
                "Télécharger un modèle depuis le hub Calamari et "
                "pointer son chemin (``.ckpt`` ou dossier).",
            )
        if batch_size < 1:
            raise OCRAdapterError(
                f"CalamariAdapter : batch_size doit être ≥ 1, reçu "
                f"{batch_size}.",
            )
        self._name = name
        self._checkpoint = Path(checkpoint)
        self._voter = voter
        self._batch_size = batch_size
        # Predictor chargé paresseusement.
        self._predictor: Any | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def checkpoint(self) -> Path:
        return self._checkpoint

    @property
    def voter(self) -> str:
        return self._voter

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def execute(
        self,
        inputs: dict[ArtifactType, Artifact],
        params: dict[str, Any],
        context: Any,
    ) -> dict[ArtifactType, Artifact]:
        """Exécute Calamari sur l'image fournie.

        Raises
        ------
        OCRAdapterError
            - input ``IMAGE`` absent ou sans URI ;
            - fichier image / checkpoint introuvable ;
            - ``calamari_ocr`` non installé ;
            - erreur Calamari (modèle invalide, inférence).
        """
        if ArtifactType.IMAGE not in inputs:
            raise OCRAdapterError(f"{self.name} : input IMAGE manquant.")
        image_artifact = inputs[ArtifactType.IMAGE]
        if image_artifact.uri is None:
            raise OCRAdapterError(
                f"{self.name} : artefact image {image_artifact.id!r} "
                "sans URI.",
            )
        image_path = Path(image_artifact.uri)
        if not image_path.exists():
            raise OCRAdapterError(
                f"{self.name} : image introuvable {image_path!r}.",
            )
        if not self._checkpoint.exists():
            raise OCRAdapterError(
                f"{self.name} : checkpoint introuvable "
                f"{self._checkpoint!r}.",
            )

        # Lazy-import — message explicite si dépendance absente.
        try:
            import numpy as np
            from calamari_ocr.ocr.predict.predictor import (  # type: ignore[import-not-found]
                Predictor,
                PredictorParams,
            )
            from PIL import Image
        except ImportError as exc:
            raise OCRAdapterError(
                f"{self.name} : calamari-ocr non installé.  "
                "Installer avec : pip install 'picarones[calamari]' "
                "(ou 'pip install calamari-ocr>=2.0').",
            ) from exc

        # Charger le Predictor une seule fois.
        if self._predictor is None:
            try:
                params = PredictorParams()
                params.silent = True
                self._predictor = Predictor.from_checkpoint(
                    params=params,
                    checkpoint=str(self._checkpoint),
                )
            except Exception as exc:
                raise OCRAdapterError(
                    f"{self.name} : chargement checkpoint "
                    f"{self._checkpoint!r} échoué : "
                    f"{type(exc).__name__}: {exc}",
                ) from exc

        # OCR ligne : Calamari attend des numpy arrays grayscale.
        try:
            with Image.open(image_path) as image:
                img_array = np.array(image.convert("L"))
            results = list(self._predictor.predict_raw([img_array]))
            if not results:
                text = ""
            else:
                # Calamari ≥ 2.0 retourne des PredictionResult avec
                # ``.outputs.sentence`` (post-voting) ou ``.sentence``.
                result = results[0]
                if hasattr(result, "outputs"):
                    text = getattr(result.outputs, "sentence", "")
                else:
                    text = getattr(result, "sentence", "")
        except Exception as exc:
            raise OCRAdapterError(
                f"{self.name} : Calamari a levé sur "
                f"{image_path!r} : {type(exc).__name__}: {exc}",
            ) from exc

        text = (text or "").strip()

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


__all__ = ["CalamariAdapter"]
