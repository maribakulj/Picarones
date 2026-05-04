"""``PrecomputedTextAdapter`` — premier adapter natif du nouveau monde.

Sprint A14-S26 du rewrite ciblé.

Cas d'usage BnF
---------------
*« J'ai déjà fait tourner Tesseract, GPT-4-vision, Pero OCR et un
service cloud sur mon corpus.  J'ai 4 répertoires de fichiers
``.txt`` à côté de mes images.  Je veux comparer ces 4 sorties dans
Picarones — je n'ai pas besoin de re-lancer un OCR, j'ai juste besoin
de la machinerie d'évaluation. »*

Ce besoin est légitime et fréquent à la BnF : une part importante
du travail de comparaison se fait sur des transcriptions déjà
produites par d'autres outils.  Ré-exécuter un OCR à chaque
benchmark est gaspillage.

Convention de nommage
---------------------
Pour une image ``<stem>.png`` (ou ``.jpg``, ``.tif``, etc.), le
texte pré-calculé est lu depuis :

::

    <stem>.<source_label>.txt

dans le **même répertoire** que l'image.  Exemple avec deux
sources concurrentes :

::

    folio_001.png
    folio_001.tesseract.txt    # produit par Tesseract
    folio_001.pero.txt         # produit par Pero OCR
    folio_001.gpt4v.txt        # produit par GPT-4 Vision
    folio_001.gt.txt           # vérité terrain

Plusieurs ``PrecomputedTextAdapter`` peuvent coexister dans une
même YAML avec des ``source_label`` distincts — chacun lit son
propre fichier, le ``BenchmarkService`` les traite en parallèle.

Configuration YAML
------------------

::

    pipelines:
      - name: tesseract_baseline
        initial_inputs: [image]
        steps:
          - id: ocr
            adapter_class: picarones.adapters.ocr.precomputed.PrecomputedTextAdapter
            adapter_kwargs:
              source_label: tesseract
            input_types: [image]
            output_types: [raw_text]

      - name: gpt4v_alternative
        initial_inputs: [image]
        steps:
          - id: ocr
            adapter_class: picarones.adapters.ocr.precomputed.PrecomputedTextAdapter
            adapter_kwargs:
              source_label: gpt4v
            input_types: [image]
            output_types: [raw_text]

Comportement « fichier manquant »
---------------------------------
Par défaut, si le fichier ``<stem>.<source_label>.txt`` est absent,
l'adapter lève ``OCRAdapterError`` — le pipeline executor marque le
step comme failed pour ce document, et le ``BenchmarkService`` le
voit en ``failed_metrics``.  Pas de fallback silencieux qui
mentirait sur la couverture du benchmark.

L'option ``missing_text_policy="empty"`` permet, à la demande
explicite du caller, de remplacer un fichier absent par une chaîne
vide — utile pour mesurer ce qui se passerait si une source était
indisponible sur certains documents.  Par défaut : ``"raise"``.

Anti-sur-ingénierie
-------------------
- Pas de découverte automatique de tous les ``source_label``
  présents dans un répertoire.  Le caller déclare explicitement
  les sources qu'il veut comparer.
- Pas de cache.  Le filesystem fait son boulot.
- Pas de validation d'encodage exotique.  ``utf-8`` strict ; un
  fichier mal encodé lève une erreur lisible.
- Pas d'extraction structurelle.  Cet adapter sort du ``RAW_TEXT``,
  point.  Pour comparer des ALTO_XML pré-calculés, c'est un
  ``PrecomputedAltoAdapter`` futur (pattern identique).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from picarones.adapters.ocr.base import BaseOCRAdapter, OCRAdapterError
from picarones.domain.artifacts import Artifact, ArtifactType


class PrecomputedTextAdapter(BaseOCRAdapter):
    """Adapter qui lit du texte OCR pré-calculé depuis le filesystem.

    Parameters
    ----------
    source_label:
        Étiquette identifiant la source du texte pré-calculé
        (ex : ``"tesseract"``, ``"gpt4v"``, ``"pero"``).  Doit être
        composée uniquement de caractères alphanumériques, ``_`` et
        ``-`` — c'est un composant de nom de fichier.
    missing_text_policy:
        ``"raise"`` (défaut) → fichier absent lève ``OCRAdapterError``.
        ``"empty"`` → fichier absent remplacé par chaîne vide
        (l'adapter produit alors un ``Artifact`` pointant sur un
        fichier vide).

    Raises
    ------
    OCRAdapterError
        Si ``source_label`` est invalide.
    """

    input_types = frozenset({ArtifactType.IMAGE})
    output_types = frozenset({ArtifactType.RAW_TEXT})
    execution_mode = "io"

    def __init__(
        self,
        *,
        source_label: str,
        missing_text_policy: Literal["raise", "empty"] = "raise",
    ) -> None:
        if not source_label or not source_label.strip():
            raise OCRAdapterError(
                "PrecomputedTextAdapter : source_label vide.",
            )
        if not all(
            c.isalnum() or c in "_-" for c in source_label
        ):
            raise OCRAdapterError(
                f"PrecomputedTextAdapter : source_label invalide "
                f"{source_label!r} — alphanumérique + _ - uniquement.",
            )
        if missing_text_policy not in ("raise", "empty"):
            raise OCRAdapterError(
                f"missing_text_policy doit être 'raise' ou 'empty', "
                f"reçu {missing_text_policy!r}.",
            )
        self._source_label = source_label
        self._missing_policy = missing_text_policy

    @property
    def name(self) -> str:
        return f"precomputed_{self._source_label}"

    @property
    def source_label(self) -> str:
        return self._source_label

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
        text_path = (
            image_path.parent / f"{image_path.stem}.{self._source_label}.txt"
        )

        if not text_path.exists():
            if self._missing_policy == "empty":
                # On crée le fichier vide pour rester cohérent : tout
                # ``Artifact`` produit a une URI vers un fichier
                # lisible.
                text_path.write_text("", encoding="utf-8")
            else:
                raise OCRAdapterError(
                    f"{self.name} : fichier pré-calculé introuvable "
                    f"pour {image_path.name!r} : "
                    f"{text_path.name!r} attendu dans "
                    f"{image_path.parent!r}.",
                )

        # Validation rapide de l'encodage UTF-8 (lecture qui leverait
        # si encodage exotique).
        try:
            text_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise OCRAdapterError(
                f"{self.name} : {text_path!r} n'est pas en UTF-8 : "
                f"{exc}",
            ) from exc

        return {
            ArtifactType.RAW_TEXT: Artifact(
                id=f"{context.document_id}:{self.name}:raw_text",
                document_id=context.document_id,
                type=ArtifactType.RAW_TEXT,
                produced_by_step="ocr",
                uri=str(text_path),
            ),
        }


__all__ = ["PrecomputedTextAdapter"]
