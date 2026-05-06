"""``ProjectionEngine`` — Sprint A14-S27.

Le S13 fusionnait dans ``DefaultEvaluationViewExecutor`` deux
responsabilités distinctes : transformer un artefact d'un type vers
un autre (« projeter ») **et** calculer les métriques sur les
payloads (« évaluer »).  La cible architecturale les sépare en
deux moteurs spécialisés à responsabilité unique :

- ``ProjectionEngine`` (ce module) : transforme un ``Artifact``
  candidat selon une ``ProjectionSpec`` et retourne le nouvel
  artefact, son ``payload`` calculé, et un ``ProjectionReport``
  documentant les pertes.
- ``EvaluationEngine`` (cf. ``evaluation_engine.py``) : calcule les
  métriques sur des payloads.

L'executor de vue (``DefaultEvaluationViewExecutor``) orchestre les
deux : projection d'abord, puis chargement, normalisation, et
évaluation.  Il ne contient plus de logique de projection ni de
calcul de métrique — uniquement la séquence et la collecte d'erreurs.

Pourquoi cette séparation
-------------------------
- **Réutilisation** : le ``PipelineExecutor`` (S28+) appelle
  ``ProjectionEngine.project`` directement quand il transforme un
  artefact entre deux étapes du DAG, sans dépendre de l'executor de
  vue.
- **Testabilité** : on peut tester la projection sur des artefacts
  arbitraires sans construire un ``EvaluationView`` ni un
  ``MetricRegistry``.
- **Lisibilité** : chaque moteur expose une API minimale et
  vérifiable au type.

Anti-sur-ingénierie
-------------------
Pas de cache de payload entre projections, pas de batch, pas de
pré-validation des params (le projecteur lui-même validera ce qu'il
attend).  Le moteur est volontairement minimal — la complexité vit
dans les projecteurs (cf. ``picarones/evaluation/projectors/``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from picarones.domain.artifacts import Artifact
from picarones.domain.errors import ProjectionError
from picarones.domain.projection_spec import ProjectionSpec
from picarones.evaluation.projectors.base import ProjectionReport
from picarones.evaluation.projectors.registry import (
    ProjectorNotFoundError,
    ProjectorRegistry,
)


@dataclass(frozen=True)
class ProjectionResult:
    """Résultat d'un appel à ``ProjectionEngine.project``.

    Attributes
    ----------
    artifact:
        Artefact effectif après projection.  Si la spec était
        ``None`` ou identité, c'est l'artefact d'entrée tel quel.
    payload:
        Payload calculé par le projecteur, ou ``None`` si aucune
        projection n'a été effectuée (le caller chargera depuis
        son ``payload_loader``).
    report:
        Rapport de projection si une projection a eu lieu, ou
        ``None`` pour une vue sans projection (identité).

    Notes
    -----
    Frozen dataclass : aucune mutation post-construction.  La
    sérialisation passe par ``ProjectionReport`` (pydantic) qui sait
    déjà se sérialiser ; ``ProjectionResult`` reste un container
    interne entre engine et executor.
    """

    artifact: Artifact
    payload: Any | None
    report: ProjectionReport | None

    @property
    def has_projection(self) -> bool:
        """Vrai si une projection effective a eu lieu (report présent)."""
        return self.report is not None


class ProjectionEngine:
    """Moteur de projection d'artefacts selon une ``ProjectionSpec``.

    Responsabilité unique : prendre un ``Artifact`` et une éventuelle
    ``ProjectionSpec``, retourner un ``ProjectionResult``.  Pas de
    chargement de payload depuis un loader externe (le projecteur
    fournit le payload calculé directement, depuis Sprint S25).  Pas
    de connaissance des métriques ni des vues.

    Parameters
    ----------
    projector_registry:
        Registre des projecteurs disponibles, instancié explicitement
        au démarrage de l'application.  Pas de singleton global, pas
        de side-effect d'import.
    """

    def __init__(self, projector_registry: ProjectorRegistry) -> None:
        if not isinstance(projector_registry, ProjectorRegistry):
            raise TypeError(
                "projector_registry doit être un ProjectorRegistry."
            )
        self._projectors = projector_registry

    @property
    def projectors(self) -> ProjectorRegistry:
        """Accès en lecture au registre sous-jacent (utile aux tests)."""
        return self._projectors

    def project(
        self,
        artifact: Artifact,
        spec: ProjectionSpec | None,
    ) -> ProjectionResult:
        """Applique la projection si pertinente.

        Comportement :

        - ``spec is None`` ou ``spec.is_identity`` →
          ``ProjectionResult`` avec l'artefact d'entrée tel quel,
          ``payload=None``, ``report=None``.  Le caller utilisera
          son payload_loader pour charger l'artefact original.
        - Sinon : résout le projecteur dans le registre, exécute
          ``project()``, et retourne le ``ProjectionResult`` complet
          avec payload calculé.

        Raises
        ------
        ProjectionError
            Si le projecteur référencé n'est pas enregistré, ou si
            le projecteur lève une exception interne (wrappée dans
            une ``ProjectionError`` qui préserve la chaîne ``__cause__``).
        """
        if spec is None or spec.is_identity:
            return ProjectionResult(
                artifact=artifact, payload=None, report=None,
            )

        try:
            projector = self._projectors.get(spec.projector_name)
        except ProjectorNotFoundError as exc:
            raise ProjectionError(
                f"Projecteur {spec.projector_name!r} introuvable "
                "dans le ProjectorRegistry."
            ) from exc

        try:
            target, payload, report = projector.project(
                artifact, dict(spec.params),
            )
        except ProjectionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ProjectionError(
                f"Projecteur {spec.projector_name!r} a levé sur "
                f"l'artefact {artifact.id!r} : {exc}"
            ) from exc

        return ProjectionResult(
            artifact=target, payload=payload, report=report,
        )


__all__ = ["ProjectionEngine", "ProjectionResult"]
