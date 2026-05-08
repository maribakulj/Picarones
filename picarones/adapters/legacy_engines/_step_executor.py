"""``LegacyOCREngineExecutor`` — wrapper ``BaseOCREngine`` → ``StepExecutor``.

Sprint A.1 du plan v2.0 (préparation à la suppression de
``OCRLLMPipeline``).  Le wrapper présente les 5 OCR engines legacy
(``TesseractEngine``, ``PeroOCREngine``, ``MistralOCREngine``,
``AzureDocIntelEngine``, ``GoogleVisionEngine``) comme des
``StepExecutor`` consommables par ``PipelineExecutor``.

Pourquoi
--------
``OCRLLMPipeline`` historique compose un ``BaseOCREngine`` + un
``BaseLLMAdapter`` en mémoire.  Le rewrite consomme un ``PipelineSpec``
exécuté par ``PipelineExecutor`` qui résout chaque step en
``StepExecutor``.  Pour migrer progressivement (Sprint B), il faut
pouvoir injecter un OCR engine legacy dans le ``PipelineExecutor`` sans
réimplémenter chacun des 5 adapters au contrat ``BaseOCRAdapter``.

Le wrapper résout cette tension : il accepte une instance
``BaseOCREngine`` au constructeur, expose les attributs
``StepExecutor`` (``input_types``, ``output_types``, ``execution_mode``,
``execute``), et délègue à ``engine.run(image_path)`` en interne.

Trace de retrait
----------------
Ce wrapper est lui-même legacy au sens du Sprint H : il sera supprimé
en même temps que ``BaseOCREngine`` quand les 5 moteurs concrets
auront migré vers ``BaseOCRAdapter`` (qui existe déjà côté rewrite —
cf. ``picarones.adapters.ocr.tesseract.TesseractAdapter`` et al.).

Anti-sur-ingénierie
-------------------
- Pas de retry au niveau du wrapper (l'engine legacy gère ses propres
  retries dans ``run()`` si configuré).
- Pas de capture custom des confidences (le rewrite a son propre
  artifact ``CONFIDENCES`` dédié, pas mappé ici).
- ``run().error`` non vide → on lève ``OCRAdapterError`` ; le
  ``PipelineExecutor`` capturera et marquera le step en échec.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from picarones.adapters.legacy_engines.base import BaseOCREngine
from picarones.adapters.ocr.base import OCRAdapterError
from picarones.adapters.output_paths import resolve_output_path
from picarones.domain.artifacts import Artifact, ArtifactType


class LegacyOCREngineExecutor:
    """Présente un ``BaseOCREngine`` legacy comme ``StepExecutor``.

    Parameters
    ----------
    engine:
        Instance d'un sous-classe de ``BaseOCREngine`` (Tesseract,
        Pero, Mistral OCR, Google Vision, Azure DI).

    Attributes
    ----------
    name:
        Délégué à ``engine.name``.
    input_types:
        ``frozenset({ArtifactType.IMAGE})`` — un OCR consomme une image.
    output_types:
        ``frozenset({ArtifactType.RAW_TEXT})`` — produit du texte plat.
    execution_mode:
        Hérité de ``engine.execution_mode`` (``"io"`` pour les engines
        cloud, ``"cpu"`` pour Tesseract/Pero qui sont CPU-bound).

    Examples
    --------
    >>> from picarones.adapters.legacy_engines.tesseract import TesseractEngine
    >>> from picarones.adapters.legacy_engines._step_executor import (
    ...     LegacyOCREngineExecutor,
    ... )
    >>> step = LegacyOCREngineExecutor(TesseractEngine({"lang": "fra"}))
    >>> step.input_types
    frozenset({<ArtifactType.IMAGE: 'image'>})
    >>> step.output_types
    frozenset({<ArtifactType.RAW_TEXT: 'raw_text'>})
    """

    input_types: frozenset = frozenset({ArtifactType.IMAGE})
    output_types: frozenset = frozenset({ArtifactType.RAW_TEXT})

    def __init__(self, engine: BaseOCREngine) -> None:
        # Duck-typing tolérant : on accepte un ``BaseOCREngine`` réel
        # ou un mock qui expose ``run()`` et ``name``.  Cela permet
        # aux tests existants (Sprint 15) qui injectent des
        # ``MagicMock`` de continuer à fonctionner.
        if not (
            hasattr(engine, "run") and callable(engine.run)
            and hasattr(engine, "name")
        ):
            raise OCRAdapterError(
                "LegacyOCREngineExecutor requires an object with ``run()`` "
                f"and ``name`` ; got {type(engine).__name__}."
            )
        self._engine = engine
        # Le runner choisit ``ProcessPoolExecutor`` pour ``"cpu"``
        # (Tesseract/Pero) et ``ThreadPoolExecutor`` pour ``"io"``
        # (Mistral/Google/Azure).  On respecte le mode déclaré par
        # l'engine — ``"io"`` par défaut si l'engine ne le déclare pas
        # (cas du mock).
        self.execution_mode: str = getattr(engine, "execution_mode", "io")
        if not isinstance(self.execution_mode, str):
            self.execution_mode = "io"

    @property
    def name(self) -> str:
        return self._engine.name

    def execute(
        self,
        inputs: dict[ArtifactType, Artifact],
        params: dict[str, Any],
        context: Any,
    ) -> dict[ArtifactType, Artifact]:
        """Exécute l'OCR engine legacy et retourne un ``Artifact RAW_TEXT``.

        Parameters
        ----------
        inputs:
            Doit contenir ``ArtifactType.IMAGE``.  L'URI de l'artefact
            image est passée à ``engine.run()``.
        params:
            Ignorés.  La configuration de l'engine passe par son
            constructeur, pas par les ``params`` du step.
        context:
            ``RunContext``.  Sert à composer les ``Artifact.id`` et à
            résoudre le chemin d'écriture du texte produit
            (``context.workspace_uri``).

        Returns
        -------
        dict[ArtifactType, Artifact]
            ``{ArtifactType.RAW_TEXT: Artifact(uri=<text_file>)}``.

        Raises
        ------
        OCRAdapterError
            Si ``inputs[IMAGE]`` est absent, sans URI, ou si
            ``engine.run()`` retourne un ``EngineResult`` en erreur.
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
                f"{self.name} : fichier image introuvable {image_path!r}.",
            )

        result = self._engine.run(image_path)
        if not result.success:
            raise OCRAdapterError(
                f"{self.name} : OCR engine a échoué ({result.error}).",
            )

        # Le contrat StepExecutor exige des artifacts avec URI filesystem
        # — on écrit le texte produit dans le workspace du run.
        out_path = resolve_output_path(
            input_path=image_path,
            adapter_name=self.name,
            suffix="raw_text.txt",
            context=context,
        )
        out_path.write_text(result.text, encoding="utf-8")

        return {
            ArtifactType.RAW_TEXT: Artifact(
                id=f"{context.document_id}:{self.name}:raw_text",
                document_id=context.document_id,
                type=ArtifactType.RAW_TEXT,
                produced_by_step="ocr",
                uri=str(out_path),
            ),
        }


__all__ = ["LegacyOCREngineExecutor"]
