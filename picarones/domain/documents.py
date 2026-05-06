"""``DocumentRef`` — Sprint A14-S4.

Référence à un document du corpus, avec ses vérités terrain
multi-niveaux.  Ne porte **pas** le contenu : juste les chemins/URIs
et les types.  Le contenu est chargé à la demande par les adapters
de format (``picarones.formats.*``).

Pourquoi pas une dataclass simple ?
-----------------------------------
On utilise pydantic pour la validation systématique : un caller qui
construit un ``DocumentRef`` avec une GT typée ``ALTO_XML`` mais
pointant vers un ``foo.txt`` doit échouer immédiatement, pas plus
tard dans le pipeline.

Anti-sur-ingénierie
-------------------
On ne porte pas ici (à ajouter au cas par cas) :

- ``language`` (vit dans ``CorpusSpec.metadata``).
- ``script_type`` (vit dans la stratification du runner, S15).
- ``image_quality`` (calculé par un adapter d'analyse, pas une
  propriété du document de référence).
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

from picarones.domain.artifacts import ArtifactType

#: Identifiant de document : alphanum + ``_.-/`` (les ``/`` permettent
#: les hiérarchies type ``volA/folio_001``).  Pas d'espaces, pas de
#: caractères de contrôle, pas d'octets nuls.
_DOC_ID_RE = re.compile(r"^[A-Za-z0-9_.\-/]+$")


class GroundTruthRef(BaseModel):
    """Pointeur vers une vérité terrain pour un niveau donné.

    Distinct du contenu : on charge le fichier à la demande via
    l'adapter de format approprié.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: ArtifactType
    """Type de la GT (TEXT, ALTO_XML, PAGE_XML, ENTITIES, READING_ORDER)."""

    uri: str = Field(min_length=1, max_length=2048)
    """Chemin filesystem (relatif ou absolu) ou URI distant."""


class DocumentRef(BaseModel):
    """Référence à un document du corpus.

    Immuable.  Construit par un adapter de corpus
    (``picarones.adapters.corpus.*``) lors du chargement, consommé
    par le pipeline executor (``picarones.pipeline``).

    Attributs
    ---------
    id:
        Identifiant unique du document dans le corpus.  Convention
        usuelle : nom de fichier sans extension (``"folio_001"``)
        ou chemin relatif (``"volA/folio_001"``).
    image_uri:
        Chemin vers l'image source.  ``None`` autorisé pour les
        documents purement textuels (corpus déjà transcrit où
        l'image n'est pas disponible).
    ground_truths:
        Liste des vérités terrain disponibles pour ce document, une
        par niveau.  La même clé ``type`` ne doit pas apparaître
        deux fois (validé).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=256)
    image_uri: str | None = Field(default=None, max_length=2048)
    ground_truths: tuple[GroundTruthRef, ...] = Field(default_factory=tuple)

    @field_validator("id")
    @classmethod
    def _validate_doc_id(cls, v: str) -> str:
        if not _DOC_ID_RE.match(v):
            from picarones.domain.errors import CorpusSpecError
            raise CorpusSpecError(
                f"document id invalide : {v!r}.  "
                f"Doit matcher {_DOC_ID_RE.pattern!r}."
            )
        # Défense en profondeur path-traversal : ``..`` comme segment
        # de chemin permet d'écrire hors workspace via
        # ``resolve_output_path``.  Le seul rempart au niveau supérieur
        # est l'extraction ZIP (zip-slip protection) — un caller qui
        # construit ``DocumentRef(id="../../etc/passwd")``
        # programmatiquement contournait tout.
        if ".." in v.split("/"):
            from picarones.domain.errors import CorpusSpecError
            raise CorpusSpecError(
                f"document id contient un segment '..' : {v!r}. "
                "Path traversal rejeté."
            )
        return v

    @field_validator("ground_truths")
    @classmethod
    def _validate_unique_gt_types(
        cls, v: tuple[GroundTruthRef, ...],
    ) -> tuple[GroundTruthRef, ...]:
        seen: set[ArtifactType] = set()
        for gt in v:
            if gt.type in seen:
                from picarones.domain.errors import CorpusSpecError
                raise CorpusSpecError(
                    f"GT dupliquée pour le type {gt.type.value!r}.  "
                    "Un document ne peut avoir qu'une seule GT par niveau."
                )
            seen.add(gt.type)
        return v

    def gt_for(self, artifact_type: ArtifactType) -> GroundTruthRef | None:
        """Retourne la GT du niveau demandé, ou ``None`` si absente."""
        for gt in self.ground_truths:
            if gt.type == artifact_type:
                return gt
        return None

    @property
    def available_gt_types(self) -> tuple[ArtifactType, ...]:
        """Niveaux de GT disponibles pour ce document, dans l'ordre
        d'insertion."""
        return tuple(gt.type for gt in self.ground_truths)


__all__ = ["DocumentRef", "GroundTruthRef"]
