"""``KrakenAdapter`` — adapter pour Kraken HTR (manuscrits + imprimés).

Implémente le contrat ``BaseOCRAdapter`` (couche 5) :
``execute(inputs, params, context) → dict[ArtifactType, Artifact]``.

Cas d'usage BnF
---------------
Kraken est l'engine open-source de référence pour les manuscrits et
imprimés anciens où Tesseract ne fonctionne pas — segmentation par
ligne de base + reconnaissance LSTM.  C'est l'OCR ciblé par
HTR-United, l'écosystème de partage de modèles HTR pour le
patrimoine écrit.

Configuration
-------------
Constructeur :

- ``name`` (défaut ``"kraken"``) : identifiant de l'instance.
- ``model_path`` (obligatoire) : chemin vers le modèle ``.mlmodel``.
  Kraken ne fournit pas de modèle par défaut — l'utilisateur doit en
  pointer un (téléchargeable depuis https://htr-united.github.io/ ou
  https://zenodo.org).
- ``binarize`` (défaut ``True``) : applique la binarisation
  ``nlbin`` avant segmentation.
- ``text_direction`` (défaut ``"horizontal-lr"``) : direction du
  texte (passée à ``pageseg.segment``).

Comportement
------------
1. Vérifie qu'un ``Artifact`` ``IMAGE`` est présent.
2. Lazy-import de ``kraken`` et ``PIL``.
3. Charge le modèle (cache par instance).
4. Binarise + segmente l'image.
5. Reconnaît chaque ligne, concatène avec un saut de ligne.
6. Écrit le résultat dans ``<stem>.<name>.txt`` à côté de l'image.

Anti-sur-ingénierie
-------------------
- Pas d'extraction de confidences pour l'instant — Kraken expose des
  scores par caractère via ``rpred``, à brancher quand un caller en
  aura besoin (les VOC types de confidences sont par-token, ici on a
  par-char).
- Pas de support batch — un appel par image.
- Pas de retry — si Kraken plante, on remonte ``OCRAdapterError``.
- ``execution_mode="cpu"`` même si Kraken peut tourner sur GPU :
  la décision pool est laissée au runner (un opérateur GPU peut
  exporter ``CUDA_VISIBLE_DEVICES`` et tourner en ThreadPool sans
  conflit).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from picarones.adapters.ocr.base import BaseOCRAdapter, OCRAdapterError
from picarones.adapters.output_paths import resolve_output_path
from picarones.domain.artifacts import Artifact, ArtifactType


class KrakenAdapter(BaseOCRAdapter):
    """Adapter Kraken HTR (Sprint Phase 3 — chantier post-rewrite).

    Parameters
    ----------
    name:
        Identifiant lisible de l'instance.  Défaut ``"kraken"``.
        Doit être alphanumérique + ``_-``.
    model_path:
        Chemin vers le modèle ``.mlmodel`` Kraken.  **Obligatoire** :
        Kraken n'embarque pas de modèle par défaut.
    binarize:
        Si ``True`` (défaut), applique ``binarization.nlbin`` avant
        segmentation.  À désactiver pour des images déjà binarisées.
    text_direction:
        Direction de lecture passée à ``pageseg.segment``.  Défaut
        ``"horizontal-lr"`` (gauche-droite horizontal).

    Raises
    ------
    OCRAdapterError
        Si ``name`` invalide, ``model_path`` vide ou inexistant.
    """

    input_types = frozenset({ArtifactType.IMAGE})
    output_types = frozenset({ArtifactType.RAW_TEXT})
    execution_mode = "cpu"

    def __init__(
        self,
        *,
        name: str = "kraken",
        model_path: str | Path | None = None,
        binarize: bool = True,
        text_direction: str = "horizontal-lr",
    ) -> None:
        if not name or not name.strip():
            raise OCRAdapterError(
                "KrakenAdapter : name vide non autorisé.",
            )
        if not all(c.isalnum() or c in "_-" for c in name):
            raise OCRAdapterError(
                f"KrakenAdapter : name invalide {name!r} — "
                "alphanumérique + _ - uniquement.",
            )
        if not model_path:
            raise OCRAdapterError(
                "KrakenAdapter : model_path est obligatoire — Kraken "
                "n'embarque pas de modèle par défaut.  Télécharger un "
                "modèle ``.mlmodel`` depuis HTR-United "
                "(https://htr-united.github.io/) et pointer son chemin.",
            )
        self._name = name
        self._model_path = Path(model_path)
        self._binarize = binarize
        self._text_direction = text_direction
        # Modèle chargé paresseusement à la première utilisation
        # — partagé entre les appels successifs de la même instance.
        self._model: Any | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def binarize(self) -> bool:
        return self._binarize

    @property
    def text_direction(self) -> str:
        return self._text_direction

    def execute(
        self,
        inputs: dict[ArtifactType, Artifact],
        params: dict[str, Any],
        context: Any,
    ) -> dict[ArtifactType, Artifact]:
        """Exécute Kraken sur l'image fournie.

        Raises
        ------
        OCRAdapterError
            - input ``IMAGE`` absent ou sans URI ;
            - fichier image / modèle introuvable ;
            - ``kraken`` ou ``PIL`` non installé ;
            - erreur Kraken (segmentation, reconnaissance).
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
        if not self._model_path.exists():
            raise OCRAdapterError(
                f"{self.name} : modèle introuvable "
                f"{self._model_path!r}.",
            )

        # Lazy-import de kraken + PIL — si absents, message explicite.
        try:
            from kraken import binarization, pageseg, rpred  # type: ignore[import-not-found]
            from kraken.lib import models  # type: ignore[import-not-found]
            from PIL import Image
        except ImportError as exc:
            raise OCRAdapterError(
                f"{self.name} : kraken/Pillow non installés.  "
                "Installer avec : pip install 'picarones[kraken]' "
                "(ou 'pip install kraken>=4.0').",
            ) from exc

        # Charger le modèle (une seule fois par instance).
        if self._model is None:
            try:
                self._model = models.load_any(str(self._model_path))
            except Exception as exc:
                raise OCRAdapterError(
                    f"{self.name} : chargement modèle "
                    f"{self._model_path!r} échoué : "
                    f"{type(exc).__name__}: {exc}",
                ) from exc

        # Pipeline Kraken : binarisation → segmentation → reco.
        try:
            with Image.open(image_path) as image:
                proc_image = (
                    binarization.nlbin(image) if self._binarize else image
                )
                segmentation = pageseg.segment(
                    proc_image, text_direction=self._text_direction,
                )
                predictions = rpred.rpred(
                    self._model, image, segmentation,
                )
                lines = [p.prediction for p in predictions if p.prediction]
                text = "\n".join(lines)
        except Exception as exc:
            raise OCRAdapterError(
                f"{self.name} : Kraken a levé sur "
                f"{image_path!r} : {type(exc).__name__}: {exc}",
            ) from exc

        text = text.strip()

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


__all__ = ["KrakenAdapter"]
