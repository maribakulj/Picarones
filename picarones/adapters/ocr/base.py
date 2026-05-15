"""``BaseOCRAdapter`` — contrat pour un adapter OCR (couche 5).

Contrat
-------
Un adapter OCR :

- Déclare ses ``input_types`` (typiquement
  ``frozenset({ArtifactType.IMAGE})``).
- Déclare ses ``output_types`` (ensemble maximal *possible* —
  typiquement ``frozenset({ArtifactType.RAW_TEXT})``, ou plus pour
  les moteurs structurés) et, si certains de ces types sont opt-in
  ou best-effort, restreint ``effective_output_types`` au strict
  garanti (cf. ``TesseractAdapter``).
- Déclare son ``execution_mode`` : ``"io"`` (défaut, ThreadPool) ou
  ``"cpu"`` (ProcessPool).
- Implémente
  ``execute(inputs, params, context) -> dict[ArtifactType, Artifact]``.

Le ``Artifact`` retourné porte une ``uri`` filesystem — convention
qui permet au ``payload_loader`` de le lire ultérieurement et
garantit la traçabilité et le streaming.

Anti-sur-ingénierie
-------------------
- Pas de hiérarchie d'erreurs.  Un adapter qui échoue lève
  ``OCRAdapterError`` (ou laisse passer une exception).  Le
  ``PipelineExecutor`` (S7) catch et marque le step en échec.
- Pas de cache au niveau de l'ABC.  Si un adapter veut cacher ses
  résultats, c'est dans son implémentation (compose ``ArtifactStore``
  S7 si besoin).
- Pas de retry.  Idem.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.domain.errors import AdapterStepError


class OCRAdapterError(AdapterStepError):
    """Erreur typée pour un échec d'adapter OCR du nouveau monde.

    Hérite de ``AdapterStepError`` (racine commune avec LLM et VLM)
    qui hérite de ``PicaronesError``.  Un caller peut catcher
    ``AdapterStepError`` pour toute erreur d'adapter sans connaître
    la sous-classe.

    Le ``PipelineExecutor`` capture cette exception (et toute autre)
    et marque le step correspondant comme failed avec
    ``StepResult.error`` renseigné.  Les callers downstream
    (``BenchmarkService``, vues) verront le pipeline en échec sans
    crash global.
    """


class BaseOCRAdapter(ABC):
    """Classe de base pour un adapter OCR du nouveau monde.

    Toute sous-classe doit :

    1. Surcharger la propriété ``name`` (identifiant lisible, utilisé
       dans les ``Artifact.id`` et le run_manifest).
    2. Implémenter ``execute(inputs, params, context)``.

    Les attributs de classe ``input_types`` / ``output_types`` /
    ``execution_mode`` sont fournis par défaut pour le cas le plus
    courant (image → texte, IO-bound).  Une sous-classe qui produit
    de l'ALTO surcharge ``output_types``, etc.

    Exemple
    -------

    ::

        class MyOCRAdapter(BaseOCRAdapter):
            @property
            def name(self) -> str:
                return "my_ocr"

            def execute(self, inputs, params, context):
                image_artifact = inputs[ArtifactType.IMAGE]
                # ... appel OCR sur image_artifact.uri ...
                # ... écriture du résultat sur disque ...
                return {
                    ArtifactType.RAW_TEXT: Artifact(
                        id=f"{context.document_id}:{self.name}:raw_text",
                        document_id=context.document_id,
                        type=ArtifactType.RAW_TEXT,
                        produced_by_step="ocr",
                        uri=str(out_path),
                    ),
                }
    """

    #: Types d'artefacts attendus en entrée.  Le ``PipelineExecutor``
    #: utilise cette info pour valider la compatibilité des steps
    #: enchaînés.
    input_types: frozenset[ArtifactType] = frozenset({ArtifactType.IMAGE})

    #: Ensemble **maximal** de types qu'un adapter de cette classe
    #: *peut* produire.  ``execute`` doit produire au moins les types
    #: déclarés dans :pyattr:`effective_output_types` (cf. ce contrat
    #: ci-dessous) — pas forcément tout ``output_types`` si certains
    #: artefacts sont opt-in ou best-effort.
    output_types: frozenset[ArtifactType] = frozenset({ArtifactType.RAW_TEXT})

    #: ``"io"`` (ThreadPool) ou ``"cpu"`` (ProcessPool).  Indique au
    #: runner quel type de pool utiliser pour la concurrence.
    execution_mode: str = "io"

    @property
    def effective_output_types(self) -> frozenset[ArtifactType]:
        """Sous-ensemble de ``output_types`` que **cette instance**
        produit de façon *garantie*, compte tenu de sa configuration.

        Défaut : ``output_types`` — le cas courant où l'adapter
        déclare exactement ce qu'il produit (tous les adapters OCR
        sauf Tesseract).

        Pourquoi cette distinction : ``_canonical_adapter_to_spec``
        génère la ``PipelineStep.output_types`` du benchmark mono-step
        à partir de cette propriété, et le ``PipelineExecutor`` marque
        le step en échec (``missing_output: [...]``) si un type déclaré
        n'a pas été produit.  Un adapter dont ``output_types`` annonce
        des artefacts *opt-in* ou *best-effort* (ex. ``TesseractAdapter``
        et ses sidecars ``CONFIDENCES`` / ``ALTO_XML``) DOIT restreindre
        cette propriété au strict garanti — sinon un extra non produit
        ferait échouer tout l'OCR alors que ``RAW_TEXT`` est valide
        (régression « analyse caractères vide » : ``engine_error``
        positionné → hooks ``requires_success`` sautés).
        """
        return self.output_types

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifiant lisible de l'adapter (ex : ``"tesseract"``,
        ``"precomputed_text"``).  Utilisé dans les ``Artifact.id`` du
        nouveau monde et dans le ``run_manifest``."""

    @abstractmethod
    def execute(
        self,
        inputs: dict[ArtifactType, Artifact],
        params: dict[str, Any],
        context: Any,
    ) -> dict[ArtifactType, Artifact]:
        """Exécute l'OCR sur les entrées et retourne les artefacts produits.

        Parameters
        ----------
        inputs:
            Map ``ArtifactType → Artifact`` avec au minimum les types
            déclarés dans ``self.input_types``.  L'adapter peut
            ignorer les entrées surnuméraires.
        params:
            Paramètres dynamiques du step (typiquement vides — la
            configuration de l'adapter passe par son constructeur).
        context:
            ``RunContext`` du run en cours (porte ``document_id``,
            ``code_version``, ``pipeline_name``).

        Returns
        -------
        dict[ArtifactType, Artifact]
            Map des artefacts produits.  Doit contenir au moins les
            types déclarés dans ``self.effective_output_types`` (les
            artefacts opt-in/best-effort de ``self.output_types`` sont
            facultatifs).

        Raises
        ------
        OCRAdapterError
            Erreur typée pour signaler un échec côté adapter (input
            invalide, fichier introuvable, etc.).
        """


__all__ = ["BaseOCRAdapter", "OCRAdapterError"]
