"""``Projector`` (Protocol) + ``ProjectionReport`` — Sprint A14-S5 / S25.

Un projecteur convertit un ``Artifact`` d'un type vers un autre,
en documentant explicitement ce qu'il **perd** au passage.

Chaque appel produit un ``ProjectionReport`` qui sera affiché par
le rapport pour expliciter à l'utilisateur ce que la comparaison
ignore.  Sans ce report, comparer "Tesseract texte brut" et
"VLM + reconstruction ALTO" sur la sortie texte serait
trompeur — l'utilisateur penserait juger les pipelines en bloc
alors qu'il ne juge qu'une projection.

payload retourné directement
-----------------------------------------
``project()`` retourne désormais ``(Artifact, payload, report)``
au lieu de ``(Artifact, report)``.  Le projecteur a déjà calculé
le contenu projeté (texte pour ALTO→texte, etc.) — le retourner
directement évite à l'executor de devoir le re-charger via un
``payload_loader`` qui ne saurait pas le récupérer (l'artefact
projeté n'a typiquement pas d'URI puisqu'il est intermédiaire).

Avant S25, l'executor appelait ``loader(projected_artifact)`` —
ce qui obligeait les tests à pré-stocker manuellement le payload
projeté dans une map (cf. le hack ``payloads[":projected_text"]``
des tests S17/S18).  Après S25, l'executor utilise directement le
payload retourné — la projection fonctionne bout-en-bout sans
collaboration explicite du loader.

Implémentations concrètes au Sprint S14 dans
``picarones/evaluation/projectors/`` :

- ``AltoToText``, ``PageToText``, ``CanonicalToText``
- ``MarkdownToText``
- ``IdentityProjector`` (pour les vues qui n'ont pas besoin de
  projection mais veulent une API uniforme).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from picarones.domain.artifacts import Artifact, ArtifactType


class ProjectionReport(BaseModel):
    """Rapport produit par un projecteur sur un artefact source.

    Immuable.  Sérialisable JSON pour persistance dans le run
    manifest.

    Attributs
    ---------
    source_artifact_id:
        Id de l'artefact source.
    source_type:
        Type de l'artefact source.
    target_type:
        Type de l'artefact projeté.
    projector_name:
        Identifiant du projecteur utilisé.
    lossy:
        ``True`` si la projection perd de l'information (cas usuel :
        ALTO → texte perd les coordonnées).  ``False`` pour une
        projection identité.
    ignored_dimensions:
        Liste des dimensions explicitement ignorées (``"geometry"``,
        ``"block_structure"``, ``"reading_order"``, ``"confidence"``,
        ...).  Affiché dans le rapport.
    warnings:
        Avertissements méthodologiques à propager dans le rapport
        (ex : "ordre de lecture deviné par défaut, peut diverger
        de l'intention éditoriale").
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_artifact_id: str
    source_type: ArtifactType
    target_type: ArtifactType
    projector_name: str
    lossy: bool = True
    ignored_dimensions: tuple[str, ...] = Field(default_factory=tuple)
    warnings: tuple[str, ...] = Field(default_factory=tuple)


@runtime_checkable
class Projector(Protocol):
    """Contrat d'un projecteur.

    Une implémentation expose deux choses : sa **signature de types**
    statique (pour que le registre puisse l'indexer) et un appel
    ``project(artifact, params) -> (Artifact, payload, ProjectionReport)``.

    Note d'implémentation : on ne contraint pas que le projecteur
    soit une classe — une simple fonction qui satisfait le protocole
    convient.  Les projecteurs canoniques du S14 seront probablement
    des classes pour porter leur configuration via constructeur,
    mais ce n'est pas une exigence du contrat.

    Le ``payload`` retourné est le contenu du nouvel artefact projeté
    (str pour RAW_TEXT, dict pour ENTITIES, etc.) — l'executor S25
    l'utilise directement sans re-passer par un ``payload_loader``.
    """

    @property
    def name(self) -> str: ...

    @property
    def source_type(self) -> ArtifactType: ...

    @property
    def target_type(self) -> ArtifactType: ...

    def project(
        self,
        artifact: Artifact,
        params: dict[str, str | int | float | bool],
    ) -> tuple[Artifact, Any, ProjectionReport]: ...


__all__ = ["Projector", "ProjectionReport"]
