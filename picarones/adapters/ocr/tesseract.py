"""``TesseractAdapter`` natif — Sprint A14-S30.

Migration native du legacy ``picarones.engines.tesseract.TesseractEngine``
vers le contrat ``BaseOCRAdapter`` (S26).  **Pas un shim** : la classe
implémente directement le contrat du nouveau monde, sans héritage du
legacy.

Le legacy ``TesseractEngine`` reste en place pour les callers qui
n'ont pas encore migré ; sa suppression viendra au S46 quand la
parité sera atteinte sur tous les adapters.

Cas d'usage BnF
---------------
Tesseract 5 reste l'OCR open-source de référence pour les corpus
imprimés et certains manuscrits réguliers.  L'adapter est CPU-bound
(Tesseract appelle une lib C en sous-process) — déclaré
``execution_mode="cpu"`` pour que le runner utilise un
``ProcessPoolExecutor``.

Configuration
-------------
Constructeur :

- ``name`` (défaut ``"tesseract"``) : identifiant de l'instance.
  Sert de suffixe au fichier de sortie ``<stem>.<name>.txt`` —
  permet de coexister avec plusieurs configurations Tesseract dans
  un même benchmark.
- ``lang`` (défaut ``"fra"``) : code langue Tesseract (``"fra"``,
  ``"lat"``, ``"eng"``, ``"fra+lat"``).
- ``psm`` (défaut ``6``) : Page Segmentation Mode (0-13).
- ``oem`` (défaut ``3``) : OCR Engine Mode.
- ``tesseract_cmd`` (défaut ``None``) : chemin vers l'exécutable
  ``tesseract`` si non standard.

Comportement
------------
1. Vérifie qu'un ``Artifact`` ``IMAGE`` est présent dans ``inputs``
   et qu'il porte une ``uri`` filesystem.
2. Lazy-import de ``pytesseract`` et ``PIL`` — si absent, lève
   ``OCRAdapterError`` avec message explicite.
3. Applique ``tesseract_cmd`` s'il est fourni.
4. Appelle ``pytesseract.image_to_string`` avec ``lang`` et
   ``--oem N --psm M``.
5. Écrit le texte dans ``<stem>.<name>.txt`` à côté de l'image
   (cohérent avec le pattern ``PrecomputedTextAdapter`` — un caller
   peut relire la sortie via cet adapter pour la comparer dans un
   second run).
6. Retourne un ``Artifact`` ``RAW_TEXT`` pointant vers le fichier
   produit.

Anti-sur-ingénierie
-------------------
- Pas de retry — Tesseract échoue rarement sur une image valide,
  et un appelant peut wrapper si besoin.
- Pas d'extraction de confidences (legacy S47) — reporté à un
  sprint dédié qui définira ``ConfidenceArtifact`` typé.  La
  fonctionnalité reste disponible via le legacy
  ``picarones.engines.tesseract.TesseractEngine`` jusqu'au S46.
- Pas de validation de l'encodage de l'image — Tesseract gère.
- Pas de support batch — un appel par image (le runner gère le
  parallélisme inter-documents).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from picarones.adapters.ocr.base import BaseOCRAdapter, OCRAdapterError
from picarones.adapters.output_paths import resolve_output_path
from picarones.domain.artifacts import Artifact, ArtifactType


class TesseractAdapter(BaseOCRAdapter):
    """Adapter Tesseract 5 natif au nouveau contrat (S26).

    Parameters
    ----------
    name:
        Identifiant lisible de l'instance.  Défaut ``"tesseract"``.
        Doit être alphanumérique + ``_-`` (composant de nom de fichier).
    lang:
        Code langue Tesseract (``"fra"``, ``"lat"``, ``"eng"``, ...).
        Défaut ``"fra"``.
    psm:
        Page Segmentation Mode entre 0 et 13.  Défaut 6
        (single uniform block of text).
    oem:
        OCR Engine Mode (0-3).  Défaut 3 (LSTM, le plus précis).
    tesseract_cmd:
        Chemin custom vers l'exécutable ``tesseract``.  Défaut
        ``None`` (laisse pytesseract trouver l'installation système).

    Raises
    ------
    OCRAdapterError
        Si le ``name`` ou les valeurs de ``psm`` / ``oem`` sont
        invalides.
    """

    input_types = frozenset({ArtifactType.IMAGE})
    # ``output_types`` est une property d'instance plutôt qu'une
    # constante de classe : son contenu dépend de
    # ``expose_confidences``.  Sans ce conditionnement, l'executor
    # validerait un output ``CONFIDENCES`` que l'adapter ne produit
    # pas en mode opt-out.
    execution_mode = "cpu"

    def __init__(
        self,
        *,
        name: str = "tesseract",
        lang: str = "fra",
        psm: int = 6,
        oem: int = 3,
        tesseract_cmd: str | None = None,
        expose_confidences: bool = True,
    ) -> None:
        if not name or not name.strip():
            raise OCRAdapterError(
                "TesseractAdapter : name vide non autorisé.",
            )
        if not all(c.isalnum() or c in "_-" for c in name):
            raise OCRAdapterError(
                f"TesseractAdapter : name invalide {name!r} — "
                "alphanumérique + _ - uniquement.",
            )
        if not 0 <= psm <= 13:
            raise OCRAdapterError(
                f"TesseractAdapter : psm doit être ∈ [0, 13], reçu {psm}.",
            )
        if not 0 <= oem <= 3:
            raise OCRAdapterError(
                f"TesseractAdapter : oem doit être ∈ [0, 3], reçu {oem}.",
            )
        self._name = name
        self._lang = lang
        self._psm = psm
        self._oem = oem
        self._tesseract_cmd = tesseract_cmd
        self._expose_confidences = expose_confidences

    @property
    def name(self) -> str:
        return self._name

    @property
    def output_types(self) -> frozenset:  # type: ignore[override]
        """Output_types dynamique selon ``expose_confidences``.

        Si l'instance expose les confidences, déclare
        ``{RAW_TEXT, CONFIDENCES}`` ; sinon ``{RAW_TEXT}`` seul.
        Le ``PipelinePlanner`` lit cette propriété pour valider
        que les types s'enchaînent.
        """
        if self._expose_confidences:
            return frozenset(
                {ArtifactType.RAW_TEXT, ArtifactType.CONFIDENCES},
            )
        return frozenset({ArtifactType.RAW_TEXT})

    @property
    def expose_confidences(self) -> bool:
        return self._expose_confidences

    @property
    def lang(self) -> str:
        return self._lang

    @property
    def psm(self) -> int:
        return self._psm

    @property
    def oem(self) -> int:
        return self._oem

    def execute(
        self,
        inputs: dict[ArtifactType, Artifact],
        params: dict[str, Any],
        context: Any,
    ) -> dict[ArtifactType, Artifact]:
        """Exécute Tesseract sur l'image fournie.

        Raises
        ------
        OCRAdapterError
            - input ``IMAGE`` absent ;
            - artefact image sans URI ;
            - fichier image introuvable ;
            - ``pytesseract`` ou ``PIL`` non installé ;
            - erreur Tesseract (lib system manquante, etc.).
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

        # Lazy-import de pytesseract + PIL — si absents, message
        # explicite plutôt qu'``ImportError`` au top-level.
        try:
            import pytesseract  # type: ignore[import-untyped]
            from PIL import Image
        except ImportError as exc:
            raise OCRAdapterError(
                f"{self.name} : pytesseract/Pillow non installés. "
                "Installer avec : pip install pytesseract pillow",
            ) from exc

        # Application du tesseract_cmd custom si fourni.
        if self._tesseract_cmd is not None:
            pytesseract.pytesseract.tesseract_cmd = self._tesseract_cmd

        # OCR.
        custom_config = f"--oem {self._oem} --psm {self._psm}"
        try:
            with Image.open(image_path) as image:
                text = pytesseract.image_to_string(
                    image,
                    lang=self._lang,
                    config=custom_config,
                )
        except Exception as exc:
            raise OCRAdapterError(
                f"{self.name} : Tesseract a levé sur "
                f"{image_path!r} : {type(exc).__name__}: {exc}",
            ) from exc

        text = text.strip()

        # Le helper résout vers le workspace si fourni (sandbox par
        # doc), sinon écrit à côté de l'image — cohérent avec le
        # pattern ``PrecomputedTextAdapter`` qui peut relire la sortie.
        text_path = resolve_output_path(
            input_path=image_path,
            adapter_name=self.name,
            suffix="txt",
            context=context,
        )
        text_path.write_text(text, encoding="utf-8")

        outputs: dict = {
            ArtifactType.RAW_TEXT: Artifact(
                id=f"{context.document_id}:{self.name}:raw_text",
                document_id=context.document_id,
                type=ArtifactType.RAW_TEXT,
                produced_by_step="ocr",
                uri=str(text_path),
            ),
        }

        # Extraction des confidences via image_to_data (best-effort).
        # Si l'extraction échoue, on log et on saute — l'OCR reste
        # valide, seule la calibration est indisponible pour ce doc.
        if self._expose_confidences:
            confidences_artifact = self._extract_and_persist_confidences(
                image_path=image_path,
                text_path=text_path,
                pytesseract_module=pytesseract,
                pil_image_class=Image,
                custom_config=custom_config,
                document_id=context.document_id,
            )
            if confidences_artifact is not None:
                outputs[ArtifactType.CONFIDENCES] = confidences_artifact

        return outputs

    def _extract_and_persist_confidences(
        self,
        *,
        image_path: Path,
        text_path: Path,
        pytesseract_module,
        pil_image_class,
        custom_config: str,
        document_id: str,
    ) -> Artifact | None:
        """Appelle ``image_to_data`` puis écrit le sidecar JSON.

        Retourne l'``Artifact CONFIDENCES`` ou ``None`` si l'extraction
        a échoué (warning loggé, OCR reste valide).
        """
        import logging
        logger = logging.getLogger(__name__)

        from picarones.adapters.ocr.confidences import (
            filter_valid_tokens,
            write_confidences_sidecar,
        )

        try:
            with pil_image_class.open(image_path) as image:
                data = pytesseract_module.image_to_data(
                    image,
                    lang=self._lang,
                    config=custom_config,
                    output_type=pytesseract_module.Output.DICT,
                )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning(
                "[%s] image_to_data indisponible (%s) — calibration "
                "sautée pour ce document.", self._name, exc,
            )
            return None

        # Format Tesseract : dict {"text": [...], "conf": [...]}.
        texts = data.get("text") or []
        confs = data.get("conf") or []
        raw = [
            {"text": t, "confidence": c}
            for t, c in zip(texts, confs)
        ]
        tokens = filter_valid_tokens(raw)
        return write_confidences_sidecar(
            text_path=text_path,
            adapter_name=self._name,
            tokens=tokens,
            document_id=document_id,
            extractor="tesseract",
        )


__all__ = ["TesseractAdapter"]


__all__ = ["TesseractAdapter"]
