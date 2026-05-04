"""``ProjectorRegistry`` — Sprint A14-S13.

Container instancié explicitement qui mappe ``projector_name``
vers une instance ``Projector``.  Symétrique du ``MetricRegistry``
(S5) : pas de singleton global, pas de side-effect d'import.

Pattern d'utilisation
---------------------

.. code-block:: python

    from picarones.evaluation.projectors import (
        ProjectorRegistry, AltoToText,
    )
    from picarones.formats.alto import AltoToText as _AltoToText

    registry = ProjectorRegistry()
    registry.register(_AltoToText())
    registry.register(PageToText())

    projector = registry.get("alto_to_text")
    target_artifact, payload, report = projector.project(source_artifact, {})

Au S20, ce registre sera construit par
``app/services/registry_service.py`` au démarrage de l'application.
Pour S13-S18, chaque test ou consommateur l'instancie explicitement.

Anti-sur-ingénierie
-------------------
Pas de versioning de projecteur, pas de namespace, pas de recherche
par tag.  Ces extras viendront quand un caller en aura concrètement
besoin (probablement avec les projecteurs contribués par des modules
tiers, post-livraison).
"""

from __future__ import annotations

from picarones.domain.errors import PicaronesError
from picarones.evaluation.projectors.base import Projector


class ProjectorRegistrationError(PicaronesError):
    """Tentative d'enregistrement invalide d'un projecteur."""


class ProjectorNotFoundError(PicaronesError):
    """Le projecteur demandé n'est pas enregistré."""


class ProjectorRegistry:
    """Container mutable de projecteurs indexés par ``name``.

    Thread-safe en lecture après initialisation ; la séquence
    d'enregistrement attendue est : un seul service, au démarrage,
    enregistre tous les projecteurs en une fois, puis l'instance
    est figée par convention.
    """

    def __init__(self) -> None:
        self._projectors: dict[str, Projector] = {}

    # ──────────────────────────────────────────────────────────────────
    # Enregistrement
    # ──────────────────────────────────────────────────────────────────

    def register(self, projector: Projector) -> None:
        """Enregistre un projecteur.

        Raises
        ------
        ProjectorRegistrationError
            Si un projecteur du même nom est déjà enregistré (sauf
            re-enregistrement strict du même objet, toléré pour les
            tests qui re-instancient).
        """
        if not hasattr(projector, "name"):
            raise ProjectorRegistrationError(
                "register : l'objet n'expose pas d'attribut ``name``."
            )
        if not isinstance(projector, Projector):
            raise ProjectorRegistrationError(
                f"register : {projector!r} ne satisfait pas le protocole "
                "Projector (attributs ``name``, ``source_type``, "
                "``target_type``, méthode ``project``)."
            )
        existing = self._projectors.get(projector.name)
        if existing is not None:
            if existing is projector:
                return  # idempotent
            raise ProjectorRegistrationError(
                f"Projecteur {projector.name!r} déjà enregistré avec "
                "une autre instance."
            )
        self._projectors[projector.name] = projector

    # ──────────────────────────────────────────────────────────────────
    # Lecture
    # ──────────────────────────────────────────────────────────────────

    def __contains__(self, name: str) -> bool:
        return name in self._projectors

    def __len__(self) -> int:
        return len(self._projectors)

    def names(self) -> list[str]:
        """Liste des noms enregistrés (ordre d'enregistrement)."""
        return list(self._projectors.keys())

    def get(self, name: str) -> Projector:
        """Récupère le projecteur par son ``name``.

        Raises
        ------
        ProjectorNotFoundError
            Si le nom n'est pas enregistré.
        """
        if name not in self._projectors:
            raise ProjectorNotFoundError(
                f"Projecteur {name!r} non enregistré.  "
                f"Disponibles : {sorted(self._projectors)}."
            )
        return self._projectors[name]


__all__ = [
    "ProjectorRegistry",
    "ProjectorRegistrationError",
    "ProjectorNotFoundError",
]
