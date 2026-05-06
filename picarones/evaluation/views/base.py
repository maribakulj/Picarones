"""``EvaluationViewExecutor`` (Protocol) + ``ViewResult`` — Sprint A14-S5.

Le contrat d'exécution d'une vue d'évaluation.  Implémentation
concrète au Sprint S13 dans
``picarones.evaluation.views.executor``.

Pattern d'utilisation cible :

.. code-block:: python

    from picarones.evaluation.registry import MetricRegistry
    from picarones.evaluation.views.executor import DefaultViewExecutor

    registry = build_default_registry()  # S20
    executor = DefaultViewExecutor(registry, projector_registry)

    for view in eval_spec.views:
        result = executor.evaluate(
            view=view,
            candidate=pipeline_artifact,
            ground_truth=gt_artifact,
        )
        # result.metric_values : dict[str, Any]
        # result.projection_report : ProjectionReport | None
        # result.warnings : tuple[str, ...]
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from picarones.domain.artifacts import Artifact
from picarones.domain.evaluation_spec import EvaluationView
from picarones.evaluation.projectors.base import ProjectionReport


class ViewResult(BaseModel):
    """Résultat de l'évaluation d'une vue sur une paire (candidat, GT).

    Immuable.  Sérialisable JSON pour persistance dans le
    ``view_results.jsonl`` du run.

    Attributs
    ---------
    view_name:
        Nom de la vue qui a produit ce résultat.
    pipeline_name:
        Nom du pipeline qui a produit l'artefact candidat.  Champ
        structurel — les renderers (CSV/JSON/HTML) ne doivent pas
        deviner cette information par parsing de
        ``candidate_artifact_id``.
    candidate_artifact_id:
        Id de l'artefact évalué (avant projection éventuelle).
    ground_truth_artifact_id:
        Id de l'artefact GT utilisé pour la comparaison.
    metric_values:
        Dict ``{metric_name: value}`` pour chaque métrique calculée
        avec succès.  Une métrique qui a échoué est absente du dict
        et apparaît dans ``failed_metrics`` avec le message d'erreur.
    failed_metrics:
        Dict ``{metric_name: error_message}`` pour les métriques qui
        ont levé une exception.  Permet au rapport d'afficher
        "métrique X non calculée : raison" plutôt que de la cacher.
    projection_report:
        ``ProjectionReport`` produit si la vue a appliqué une
        projection.  ``None`` si la vue compare l'artefact tel quel.
    warnings:
        Avertissements à propager dans le rapport (typiquement les
        ``warnings`` de ``EvaluationView`` + ceux du
        ``ProjectionReport`` éventuel).
    ignored_dimensions:
        Récapitulatif des dimensions ignorées par cette évaluation
        (combinaison de la vue + de la projection).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    view_name: str
    pipeline_name: str = Field(min_length=1, max_length=128)
    candidate_artifact_id: str
    ground_truth_artifact_id: str
    metric_values: dict[str, Any] = Field(default_factory=dict)
    failed_metrics: dict[str, str] = Field(default_factory=dict)
    projection_report: ProjectionReport | None = None
    warnings: tuple[str, ...] = Field(default_factory=tuple)
    ignored_dimensions: tuple[str, ...] = Field(default_factory=tuple)


@runtime_checkable
class EvaluationViewExecutor(Protocol):
    """Contrat de l'exécuteur de vues.

    Une implémentation prend en entrée :

    - une ``EvaluationView`` (déclarative),
    - un ``Artifact`` candidat (sortie d'un pipeline),
    - un ``Artifact`` GT (référence du corpus),

    et produit un ``ViewResult`` qui :

    1. applique la projection si la vue en spécifie une (et capture
       le ``ProjectionReport``),
    2. applique le profil de normalisation si spécifié,
    3. calcule chaque métrique listée dans
       ``view.metric_names`` (en propageant les erreurs dans
       ``failed_metrics`` plutôt que de planter),
    4. retourne un ``ViewResult`` immuable.

    Cas particulier : si l'artefact candidat n'est pas dans
    ``view.candidate_types``, l'executor lève ``ValueError`` —
    c'est au caller (typiquement le service applicatif) de filtrer
    en amont les pipelines qui ne produisent pas l'artefact attendu.
    """

    def evaluate(
        self,
        view: EvaluationView,
        candidate: Artifact,
        ground_truth: Artifact,
        *,
        pipeline_name: str,
    ) -> ViewResult: ...


__all__ = ["EvaluationViewExecutor", "ViewResult"]
