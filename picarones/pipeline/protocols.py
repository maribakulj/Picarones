"""``StepExecutor`` (Protocol)

Contrat que doit satisfaire tout adapter exécutable par le pipeline
runner.  Une fonction ou une classe peut satisfaire le protocole —
le runner ne se soucie que de l'interface.

Implémentations concrètes au Sprint S11 dans ``picarones/adapters/``
(Tesseract, Pero OCR, Mistral OCR, Google Vision, Azure DI, OpenAI,
Anthropic, Mistral, Ollama, ...).

Pattern d'utilisation cible :

.. code-block:: python

    class TesseractExecutor:
        name = "tesseract"
        input_types = frozenset({ArtifactType.IMAGE})
        output_types = frozenset({ArtifactType.RAW_TEXT})
        execution_mode = "cpu"

        def execute(
            self,
            inputs: dict[ArtifactType, Artifact],
            params: dict,
            context: RunContext,
        ) -> dict[ArtifactType, Artifact]:
            image_artifact = inputs[ArtifactType.IMAGE]
            text = pytesseract.image_to_string(image_artifact.uri, **params)
            return {ArtifactType.RAW_TEXT: build_text_artifact(text, context)}
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.pipeline.types import RunContext


#: Mode d'exécution déclaré par l'adapter.  Le runner choisit
#: ``ProcessPoolExecutor`` pour ``"cpu"``, ``ThreadPoolExecutor`` pour
#: ``"io"``.
ExecutionMode = Literal["io", "cpu"]


@runtime_checkable
class StepExecutor(Protocol):
    """Contrat d'un adapter exécutable.

    Trois propriétés statiques (le runner les inspecte sans appeler
    ``execute()``) :

    - ``name`` : identifiant stable (cf. ``PipelineStep.adapter_name``).
    - ``input_types`` : types consommés.
    - ``output_types`` : types produits.
    - ``execution_mode`` : ``"io"`` ou ``"cpu"``.

    Une méthode ``execute(inputs, params, context) -> dict[ArtifactType, Artifact]``.

    Le runner garantit que :

    - ``inputs`` contient au moins tous les types listés dans
      ``input_types``.
    - ``params`` est le dict ``PipelineStep.params`` (copie).
    - ``context`` est le ``RunContext`` du document courant.

    L'adapter garantit que :

    - Le dict retourné contient au moins tous les types listés dans
      ``output_types``.  Le runner valide cette propriété et marque
      le step en échec si un type promis manque.
    - Toute exception levée est propagée au runner ; ne rien capturer
      silencieusement.

    Le ``execute`` reste **pur du point de vue du runner** : il
    peut faire des side effects (écrire un fichier, appeler une API),
    mais le runner garantit qu'il ne sera pas appelé deux fois pour
    le même couple ``(document_id, step_id)`` dans le même run
    (cache du Sprint S7).
    """

    @property
    def name(self) -> str: ...

    @property
    def input_types(self) -> frozenset[ArtifactType]: ...

    @property
    def output_types(self) -> frozenset[ArtifactType]: ...

    @property
    def execution_mode(self) -> ExecutionMode: ...

    def execute(
        self,
        inputs: dict[ArtifactType, Artifact],
        params: dict[str, str | int | float | bool],
        context: RunContext,
    ) -> dict[ArtifactType, Artifact]: ...


__all__ = ["StepExecutor", "ExecutionMode"]
