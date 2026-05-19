"""``ArtifactCachePort`` — port (Protocol) consommé par ``PipelineExecutor``.

inversion de dépendance pour le branchement
``ArtifactStore`` dans le pipeline.

Pourquoi ce Protocol
--------------------
La couche ``pipeline/`` est plus interne que ``adapters/`` dans la
hiérarchie documentée du rewrite (``domain → formats → evaluation
→ pipeline → adapters → app → reports → interfaces``).  Importer
depuis ``adapters/`` dans ``pipeline/`` violerait la règle de
dépendance.

On applique l'inversion de dépendance (pattern hexagonal /
ports-and-adapters) :

- ``pipeline/`` définit le **port** ``ArtifactCachePort`` (ce
  module) — ce que le pipeline a besoin de consommer.
- ``adapters/storage/artifact_store.ArtifactStore`` (S29) est
  l'**adapter** qui satisfait ce port par duck typing.
- Toute autre implémentation tierce (Redis, S3, GCS, ...) qui
  implémente ces 5 méthodes est compatible.

Convention duck typing
----------------------
``StoredArtifact`` est aussi exposé comme Protocol minimal pour
éviter d'importer la dataclass concrète depuis ``adapters/``.
Les implémentations réelles fournissent une dataclass plus riche ;
``pipeline/`` ne consomme que ``stored.artifact`` et
``stored.artifact.uri``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from picarones.domain.artifacts import Artifact


@runtime_checkable
class CachedArtifactRef(Protocol):
    """Port minimal consommé par ``read_cached_outputs``.

    Les implémentations concrètes peuvent porter des champs
    supplémentaires (``payload``, ``key``, …) ; ``pipeline/``
    n'utilise que l'``Artifact`` reconstitué.
    """

    @property
    def artifact(self) -> Artifact:  # pragma: no cover — Protocol
        ...


@runtime_checkable
class ArtifactCachePort(Protocol):
    """Contrat minimal d'un cache d'artefacts consommable par
    ``PipelineExecutor`` pour la reprise par hash.

    Les méthodes correspondent **exactement** à l'API publique de
    ``ArtifactStore`` (S29) — ``ArtifactStore`` est donc compatible
    par duck typing sans rien changer.

    Pas d'``isinstance(store, ArtifactCachePort)`` requis : Python
    type-checke à l'usage (les méthodes manquantes lèvent
    ``AttributeError`` au runtime).  Le ``@runtime_checkable``
    autorise un test ``isinstance`` côté caller s'il veut une
    validation explicite.
    """

    def get(self, key: str) -> CachedArtifactRef | None:  # pragma: no cover
        ...

    def put(
        self,
        key: str,
        artifact: Artifact,
        payload: bytes | None = None,
    ) -> None:  # pragma: no cover
        ...

    def __contains__(self, key: str) -> bool:  # pragma: no cover
        ...


__all__ = ["ArtifactCachePort", "CachedArtifactRef"]
