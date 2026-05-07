"""Adaptateur ``BaseModule`` → ``StepExecutor`` (Phase 7.B).

Pont entre le contrat module legacy
(:class:`picarones.domain.module_protocol.BaseModule`,
``process(dict[ArtifactType, payload]) → dict[ArtifactType, payload]``)
et le contrat canonique
(:class:`picarones.pipeline.protocols.StepExecutor`,
``execute(dict[ArtifactType, Artifact], params, context)
 → dict[ArtifactType, Artifact]``).

Pourquoi ce module
------------------
Sub-phase 7.B du plan de convergence
(``docs/migration/pipeline-convergence-plan.md``) : on fait
consommer en interne le ``PipelineExecutor`` canonique par le
``PipelineRunner`` legacy.  Cela élimine la duplication de
moteur d'exécution (1 seul code path à maintenir) tout en
préservant l'API legacy ``BaseModule`` pour les modules qui en
hériteraient encore.

Le wrapper est **interne au module** : aucun caller production
ne devrait importer ``_BaseModuleAdapter``.  Les modules tiers
qui contribuent à un benchmark composé continuent d'écrire des
sous-classes de ``BaseModule`` ; le wrapper fait l'adaptation
au moment de l'exécution.

Sémantique des payloads
-----------------------
Les modules ``BaseModule`` historiques travaillent avec des
**payloads bruts** :

- ``ArtifactType.IMAGE`` → ``str`` (chemin filesystem)
- ``ArtifactType.RAW_TEXT`` / ``ArtifactType.CORRECTED_TEXT`` → ``str`` (texte inline)
- ``ArtifactType.ALTO_XML`` / ``ArtifactType.PAGE_XML`` → ``str`` (XML inline)
- ``ArtifactType.ENTITIES`` → ``list[dict]``
- ``ArtifactType.READING_ORDER`` → ``list[str]``

Le canonique ``Artifact`` Pydantic immutable n'a pas de champ
``content`` direct — le contenu se lit via ``uri``.  Le wrapper
résout cette incompatibilité via un **registre d'inline
payloads** in-process : chaque ``Artifact`` produit a un ``id``
unique, et le registre map ``id → payload`` pour la durée d'un
run.

Cela évite l'I/O disque pour chaque step (qui pollue le wall-
clock du chronométrage et pose des problèmes de cleanup en
test).  Trade-off : le wrapper ne fonctionne qu'**en
mono-process**.  La parallélisation inter-document via
``ProcessPoolExecutor`` (encore inutilisée par
``PipelineRunner``) requerrait une autre stratégie (URI
``data:``, sérialisation Pickle des payloads, etc.).

Anti-sur-ingénierie
-------------------
- Pas de cache d'artefacts (le registre est purement transient).
- Pas de provenance détaillée (les ``Artifact`` produits ont
  ``provenance=None`` ; le legacy ``PipelineRunner`` ne portait
  pas cette info).
- Pas de garantie inter-process (cf. trade-off ci-dessus).
"""

from __future__ import annotations

import logging
from typing import Any

from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.domain.module_protocol import BaseModule, ExecutionMode
from picarones.pipeline.types import RunContext

logger = logging.getLogger(__name__)


class _PayloadRegistry:
    """Registre in-process ``Artifact.id → payload``.

    Utilisé par :class:`_BaseModuleAdapter` pour matérialiser
    inline-payload ↔ ``Artifact`` sans I/O disque.

    Une instance par run de pipeline mono-document.  Le
    ``PipelineRunner`` qui consomme cet adapter est responsable
    d'instancier un registre par appel ``run()``.
    """

    def __init__(self) -> None:
        self._payloads: dict[str, Any] = {}

    def store(self, artifact_id: str, payload: Any) -> None:
        """Enregistre un payload inline sous ``artifact_id``."""
        self._payloads[artifact_id] = payload

    def get(self, artifact_id: str) -> Any:
        """Retourne le payload enregistré ou lève ``KeyError``."""
        if artifact_id not in self._payloads:
            raise KeyError(
                f"Payload introuvable pour artifact_id={artifact_id!r}.  "
                "Le registre attend que tous les Artifacts produits par "
                "une étape soient enregistrés en parallèle.",
            )
        return self._payloads[artifact_id]

    def __contains__(self, artifact_id: str) -> bool:
        return artifact_id in self._payloads

    def clear(self) -> None:
        """Vide le registre.  À appeler entre deux runs."""
        self._payloads.clear()


class _BaseModuleAdapter:
    """Wrappe un :class:`BaseModule` pour satisfaire le Protocol
    :class:`StepExecutor`.

    Le wrapper expose les attributs du module legacy
    (``name``, ``input_types``, ``output_types``,
    ``execution_mode``) et implémente ``execute()`` qui :

    1. Extrait les payloads des ``Artifact`` d'entrée via le
       registre (ou via ``artifact.uri`` pour les types
       file-based).
    2. Invoque ``module.process(payloads)``.
    3. Wrappe chaque payload de sortie dans un ``Artifact``
       (avec ``id`` dérivé de ``context.document_id`` + nom
       du module + type).
    4. Enregistre le payload de sortie dans le registre pour
       qu'une étape downstream puisse le consommer.
    """

    #: Types pour lesquels ``Artifact.uri`` porte directement la
    #: valeur attendue par le ``BaseModule`` historique (chemin
    #: filesystem).  Pour les autres types, on passe par le
    #: registre.
    _URI_BACKED_TYPES: frozenset[ArtifactType] = frozenset({
        ArtifactType.IMAGE,
    })

    def __init__(
        self,
        module: BaseModule,
        registry: _PayloadRegistry,
    ) -> None:
        self._module = module
        self._registry = registry

    @property
    def name(self) -> str:
        return self._module.name

    @property
    def input_types(self) -> frozenset[ArtifactType]:
        return frozenset(self._module.input_types)

    @property
    def output_types(self) -> frozenset[ArtifactType]:
        return frozenset(self._module.output_types)

    @property
    def execution_mode(self) -> ExecutionMode:
        # Mypy ne sait pas que le legacy ``BaseModule.execution_mode``
        # est typé ``Literal["io", "cpu"]`` — on coerce.
        return self._module.execution_mode  # type: ignore[return-value]

    def execute(
        self,
        inputs: dict[ArtifactType, Artifact],
        params: dict[str, Any],
        context: RunContext,
    ) -> dict[ArtifactType, Artifact]:
        """Convertit ``inputs``/``outputs`` entre les deux contrats.

        Parameters
        ----------
        inputs:
            Map ``ArtifactType → Artifact`` fournie par le
            ``PipelineExecutor`` canonique.
        params:
            Paramètres du step.  Le wrapper les ignore (le legacy
            ``BaseModule.process`` ne prend pas de params — ils
            sont configurés via le constructeur du module).
        context:
            ``RunContext`` du run en cours.

        Returns
        -------
        dict[ArtifactType, Artifact]
            Outputs sous forme ``Artifact`` typés.  Les payloads
            inline sont enregistrés dans ``self._registry`` pour
            consommation par les étapes downstream.
        """
        # 1. Extraire les payloads des Artifacts d'entrée
        payloads: dict[ArtifactType, Any] = {}
        for at, artifact in inputs.items():
            if at in self._URI_BACKED_TYPES:
                # IMAGE : le module attend un chemin string
                payloads[at] = artifact.uri or ""
            else:
                # Autres types : payload inline via registre
                if artifact.id in self._registry:
                    payloads[at] = self._registry.get(artifact.id)
                elif artifact.uri:
                    # Fallback : artefact registré ailleurs avec uri
                    # filesystem — on lit le contenu textuel.
                    from pathlib import Path
                    payloads[at] = Path(artifact.uri).read_text(
                        encoding="utf-8",
                    )
                else:
                    raise KeyError(
                        f"Artifact {artifact.id!r} (type={at.value}) sans "
                        f"payload disponible : ni dans le registre, ni via uri."
                    )

        # 2. Invoquer le module legacy
        outputs = self._module.process(payloads)

        # 3. Wrappe chaque output dans un Artifact + registre
        out_artifacts: dict[ArtifactType, Artifact] = {}
        for at, payload in outputs.items():
            artifact_id = self._build_artifact_id(context, at)
            self._registry.store(artifact_id, payload)
            artifact = Artifact(
                id=artifact_id,
                document_id=context.document_id,
                type=at,
                produced_by_step=self._module.name,
                # uri / content_hash / provenance sont None — le
                # legacy n'avait pas ces concepts.
            )
            out_artifacts[at] = artifact
        return out_artifacts

    def _build_artifact_id(
        self,
        context: RunContext,
        artifact_type: ArtifactType,
    ) -> str:
        """Construit un ``Artifact.id`` unique pour cette
        production.

        Format : ``<document_id>:<step_name>:<artifact_type>``.
        Cohérent avec la convention du wiring rewrite (cf.
        ``adapters/ocr/tesseract.py``).
        """
        return f"{context.document_id}:{self._module.name}:{artifact_type.value}"


def wrap_initial_inputs(
    inputs: dict[ArtifactType, Any],
    registry: _PayloadRegistry,
    document_id: str,
) -> dict[ArtifactType, Artifact]:
    """Convertit les ``initial_inputs`` legacy en ``dict[ArtifactType, Artifact]``.

    Le ``PipelineRunner`` legacy accepte ``initial_inputs:
    dict[ArtifactType, Any]`` où chaque valeur est un payload
    brut (chemin pour IMAGE, texte inline pour TEXT, ...).  Cette
    fonction les wrappe en ``Artifact`` typés et enregistre les
    payloads inline dans le registre.

    Parameters
    ----------
    inputs:
        Map legacy.
    registry:
        Registre de payloads (à utiliser dans le même run).
    document_id:
        ``DocumentRef.id`` du document.  Sert à construire
        les ``Artifact.id`` initiaux.

    Returns
    -------
    dict[ArtifactType, Artifact]
        Inputs canoniques.
    """
    out: dict[ArtifactType, Artifact] = {}
    for at, payload in inputs.items():
        artifact_id = f"{document_id}:__initial__:{at.value}"
        if at == ArtifactType.IMAGE:
            # Chemin filesystem : ``uri`` direct
            artifact = Artifact(
                id=artifact_id,
                document_id=document_id,
                type=at,
                uri=str(payload) if payload else None,
            )
        else:
            # Payload inline : on enregistre + Artifact sans uri
            registry.store(artifact_id, payload)
            artifact = Artifact(
                id=artifact_id,
                document_id=document_id,
                type=at,
            )
        out[at] = artifact
    return out


__all__ = [
    "_BaseModuleAdapter",
    "_PayloadRegistry",
    "wrap_initial_inputs",
]
