"""``ProjectionSpec``

Une projection convertit un artefact d'un type vers un autre, en
documentant explicitement la perte d'information (cf.
``ProjectionReport`` dans ``picarones.evaluation.projectors.base``).

``ProjectionSpec`` est la **déclaration** d'une projection ; elle
ne contient pas la logique du projecteur (qui vit dans
``picarones.evaluation.projectors.*``).  Cette séparation permet
à un ``EvaluationView`` de référencer une projection par nom dans
un YAML, sans imposer un couplage à une implémentation concrète.

Anti-sur-ingénierie
-------------------
Pas de validation forte du nom du projecteur ici (le registre
``ProjectorRegistry`` validera à la résolution, S14).  Pas de typage
strict sur ``params`` (différent par projecteur — un projecteur
ALTO→texte voudra ``{"reading_order": "natural"}``, un projecteur
CANONICAL→texte voudra autre chose).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from picarones.domain.artifacts import ArtifactType


class ProjectionSpec(BaseModel):
    """Spec déclarative d'une projection entre deux types d'artefacts.

    Attributs
    ---------
    source_type:
        Type de l'artefact en entrée du projecteur.
    target_type:
        Type de l'artefact en sortie.  Peut être identique à
        ``source_type`` (projection identité — utile pour signaler
        explicitement "pas de projection" tout en gardant l'API
        uniforme).
    projector_name:
        Identifiant du projecteur dans ``ProjectorRegistry``.
        Convention : ``"<source>_to_<target>"`` (ex : ``"alto_to_text"``,
        ``"page_to_text"``, ``"canonical_to_text"``).
    params:
        Dictionnaire de paramètres passé au projecteur.  Différent
        par projecteur ; pas de validation cross-projecteur ici.
        Le projecteur lui-même validera ce qu'il attend.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_type: ArtifactType
    target_type: ArtifactType
    projector_name: str = Field(min_length=1, max_length=128)
    params: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @property
    def is_identity(self) -> bool:
        """Vrai si la spec décrit une projection identité
        (source_type == target_type).  Utile à un caller qui veut
        court-circuiter l'appel au projecteur."""
        return self.source_type == self.target_type


__all__ = ["ProjectionSpec"]
