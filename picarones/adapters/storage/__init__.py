"""Adaptateurs de stockage — Sprint S29.

Stocks d'artefacts indexés par hash multi-paramètres pour la
reprise des runs longs.

Modules livrés
--------------
- ``artifact_store.py`` (S29) — ``ArtifactKey``, ``StoredArtifact``,
  ``ArtifactStore`` (ABC), ``InMemoryArtifactStore``,
  ``FilesystemArtifactStore``.

Pattern : un ``Storage`` est instancié par un ``app/services/``,
pas créé ad-hoc dans un router FastAPI ou un module métier.  Ça
permet d'injecter un mock en test, de basculer SQLite → Postgres
si besoin, et de centraliser les permissions/quotas.

Distinct du ``picarones/pipeline/cache.py`` (S7)
------------------------------------------------
``ArtifactCache`` (S7) reste exposé pour les callers qui en
dépendent en interne.  ``ArtifactStore`` (S29) est la nouvelle
API canonique : hash multi-paramètres (model_version, normalization
profile, projection spec), persistance optionnelle sur filesystem,
abstraction ABC.

Cibles à venir
--------------
- S37 : déplacement de ``picarones.web.jobs`` (SQLite job store).
- Post-livraison : ``picarones.measurements.history`` (SQLite
  history) et stores distribués (S3, GCS, …).
"""

from __future__ import annotations

from picarones.adapters.storage.artifact_store import (
    ArtifactKey,
    ArtifactStore,
    ArtifactStoreError,
    FilesystemArtifactStore,
    InMemoryArtifactStore,
    StoredArtifact,
)
from picarones.adapters.storage.job_store import (
    JobRecord,
    JobStore,
    JobStoreError,
)

__all__ = [
    "ArtifactKey",
    "ArtifactStore",
    "ArtifactStoreError",
    "FilesystemArtifactStore",
    "InMemoryArtifactStore",
    "StoredArtifact",
    "JobStore",
    "JobRecord",
    "JobStoreError",
]
