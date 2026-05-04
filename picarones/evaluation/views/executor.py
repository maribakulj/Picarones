"""``DefaultEvaluationViewExecutor`` — Sprint A14-S13.

Implémentation concrète du protocole ``EvaluationViewExecutor`` (S5).
Orchestration d'une vue d'évaluation sur une paire (candidat, GT) :

1. Vérifie que ``candidate.type`` est dans ``view.candidate_types``.
2. Si ``view.projection`` est défini, récupère le projecteur depuis
   ``ProjectorRegistry`` et applique la projection.  Capture le
   ``ProjectionReport``.
3. Charge les payloads (texte, ALTO parsé, etc.) via le
   ``payload_loader`` injecté au constructeur.
4. Applique optionnellement un profil de normalisation texte
   (``view.normalization_profile``) sur les payloads texte.
5. Calcule chaque métrique listée dans ``view.metric_names`` via
   ``MetricRegistry``.  Une métrique qui lève est enregistrée dans
   ``failed_metrics`` au lieu de planter le ViewResult complet.
6. Retourne un ``ViewResult`` agrégeant tout (metric_values,
   failed_metrics, projection_report, warnings,
   ignored_dimensions).

Le ``payload_loader`` est injecté pour découpler l'executor de la
manière dont les artefacts sont stockés (filesystem, in-memory,
remote).  Le service applicatif (S19) injectera un loader qui sait
gérer les workspaces sandboxés.

Anti-sur-ingénierie
-------------------
Pas de cache de payload chargé entre métriques (chaque métrique
relit l'artefact via le loader).  Si un caller veut éviter le coût
de re-lecture, il instancie un loader qui memo-ize lui-même.

Pas de gestion de batch (évaluer N paires en une seule passe).  À
ajouter quand un caller en a concrètement besoin.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from picarones.domain.artifacts import Artifact
from picarones.domain.errors import ProjectionError
from picarones.domain.evaluation_spec import EvaluationView
from picarones.evaluation.projectors.registry import (
    ProjectorNotFoundError,
    ProjectorRegistry,
)
from picarones.evaluation.registry import MetricRegistry, MetricNotFoundError
from picarones.evaluation.views.base import ViewResult

logger = logging.getLogger(__name__)


#: Sentinelle interne pour distinguer "pas de projection" de "projection
#: a retourné None comme payload" (cas pathologique mais théoriquement
#: possible).  Ne jamais comparer avec ``==`` — toujours ``is``.
_UNSET = object()


#: Type alias : un payload loader prend un Artifact et retourne le
#: contenu chargé (str pour RAW_TEXT, dict pour ENTITIES, etc.).
PayloadLoader = Callable[[Artifact], Any]


class DefaultEvaluationViewExecutor:
    """Implémentation par défaut de ``EvaluationViewExecutor``.

    Parameters
    ----------
    metric_registry:
        ``MetricRegistry`` contenant les métriques référencées par
        ``view.metric_names``.
    projector_registry:
        ``ProjectorRegistry`` contenant les projecteurs référencés
        par ``view.projection.projector_name``.
    payload_loader:
        Callable ``(Artifact) -> Any`` qui charge le contenu d'un
        artefact.  Pour les tests, typiquement un dict in-memory.
        En production (S19), un service applicatif qui sait gérer
        les workspaces.
    """

    def __init__(
        self,
        metric_registry: MetricRegistry,
        projector_registry: ProjectorRegistry,
        payload_loader: PayloadLoader,
    ) -> None:
        if not isinstance(metric_registry, MetricRegistry):
            raise TypeError(
                "metric_registry doit être un MetricRegistry."
            )
        if not isinstance(projector_registry, ProjectorRegistry):
            raise TypeError(
                "projector_registry doit être un ProjectorRegistry."
            )
        if not callable(payload_loader):
            raise TypeError("payload_loader doit être callable.")
        self._metrics = metric_registry
        self._projectors = projector_registry
        self._loader = payload_loader

    # ──────────────────────────────────────────────────────────────────
    # API publique
    # ──────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        view: EvaluationView,
        candidate: Artifact,
        ground_truth: Artifact,
    ) -> ViewResult:
        """Évalue la vue sur la paire (candidat, GT).

        Returns
        -------
        ViewResult
            Toujours retourné, jamais d'exception en sortie normale —
            les erreurs vont dans ``failed_metrics`` ou
            (pour les erreurs de projection) lèvent ``ProjectionError``
            qui est cohérente avec le contrat du S5.

        Raises
        ------
        ProjectionError
            Si la vue exige une projection que le projecteur ne peut
            pas réaliser (ex : type d'entrée incompatible avec le
            projecteur trouvé).
        ValueError
            Si ``candidate.type`` n'est pas dans
            ``view.candidate_types``.  Le caller (typiquement le
            service applicatif) doit filtrer les pipelines qui ne
            produisent pas le bon type avant d'appeler ``evaluate``.
        """
        # 1. Vérification du type d'entrée.
        if not view.accepts(candidate.type):
            raise ValueError(
                f"View {view.name!r} n'accepte pas l'artefact "
                f"{candidate.id!r} (type {candidate.type.value!r}). "
                f"Types acceptés : "
                f"{sorted(t.value for t in view.candidate_types)}."
            )

        # 2. Projection (optionnelle).  S14 — résolution par
        #    ``view.projection_for(candidate.type)`` qui supporte
        #    soit une projection unique (champ ``projection``), soit
        #    un mapping par type source (``projections_by_source_type``).
        # Sprint S25 : le projecteur retourne désormais
        # ``(Artifact, payload, report)`` — on conserve le payload
        # pour le passer aux métriques sans repasser par le loader.
        effective_candidate = candidate
        projection_report = None
        projected_payload: Any = _UNSET
        projection_spec = view.projection_for(candidate.type)
        if projection_spec is not None and not projection_spec.is_identity:
            try:
                projector = self._projectors.get(
                    projection_spec.projector_name,
                )
            except ProjectorNotFoundError as exc:
                raise ProjectionError(
                    f"View {view.name!r} référence le projecteur "
                    f"{projection_spec.projector_name!r} introuvable "
                    "dans le ProjectorRegistry."
                ) from exc
            try:
                (
                    effective_candidate,
                    projected_payload,
                    projection_report,
                ) = projector.project(
                    candidate, dict(projection_spec.params),
                )
            except ProjectionError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise ProjectionError(
                    f"Projecteur {projection_spec.projector_name!r} a "
                    f"levé sur l'artefact {candidate.id!r} : {exc}"
                ) from exc

        # 3. Chargement des payloads.
        # Échec de chargement = ViewResult avec une erreur globale
        # (pas de failed_metric par métrique — l'erreur est en amont).
        if projected_payload is not _UNSET:
            # Sprint S25 : payload calculé par le projecteur, pas
            # besoin de re-passer par le loader (l'artefact projeté
            # est intermédiaire et n'a typiquement pas d'URI).
            cand_payload = projected_payload
        else:
            try:
                cand_payload = self._loader(effective_candidate)
            except Exception as exc:  # noqa: BLE001
                return self._failed_view_result(
                    view=view,
                    candidate=candidate,
                    ground_truth=ground_truth,
                    projection_report=projection_report,
                    global_error=(
                        f"payload_loader a échoué sur le candidat "
                        f"{effective_candidate.id!r} : {exc}"
                    ),
                )
        try:
            gt_payload = self._loader(ground_truth)
        except Exception as exc:  # noqa: BLE001
            return self._failed_view_result(
                view=view,
                candidate=candidate,
                ground_truth=ground_truth,
                projection_report=projection_report,
                global_error=(
                    f"payload_loader a échoué sur la GT "
                    f"{ground_truth.id!r} : {exc}"
                ),
            )

        # 4. Normalisation texte (optionnelle).
        if view.normalization_profile is not None:
            cand_payload, gt_payload = self._apply_normalization(
                view.normalization_profile, cand_payload, gt_payload,
            )

        # 5. Calcul des métriques.  Une métrique qui lève va dans
        #    failed_metrics.  Une métrique non enregistrée va dans
        #    failed_metrics avec un message explicite.
        metric_values: dict[str, Any] = {}
        failed_metrics: dict[str, str] = {}
        for name in view.metric_names:
            try:
                value = self._metrics.compute(name, gt_payload, cand_payload)
                metric_values[name] = value
            except MetricNotFoundError as exc:
                failed_metrics[name] = (
                    f"métrique non enregistrée dans le MetricRegistry : "
                    f"{exc}"
                )
            except Exception as exc:  # noqa: BLE001
                failed_metrics[name] = (
                    f"{type(exc).__name__}: {exc}"
                )

        # 6. Construction du ViewResult.
        warnings = tuple(view.warnings)
        ignored = tuple(view.ignored_dimensions)
        if projection_report is not None:
            warnings = warnings + tuple(projection_report.warnings)
            # Déduplique les ignored_dimensions tout en préservant l'ordre.
            seen: set[str] = set(ignored)
            extra = tuple(
                d for d in projection_report.ignored_dimensions
                if d not in seen
            )
            ignored = ignored + extra

        return ViewResult(
            view_name=view.name,
            candidate_artifact_id=candidate.id,
            ground_truth_artifact_id=ground_truth.id,
            metric_values=metric_values,
            failed_metrics=failed_metrics,
            projection_report=projection_report,
            warnings=warnings,
            ignored_dimensions=ignored,
        )

    # ──────────────────────────────────────────────────────────────────
    # Helpers internes
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_normalization(
        profile_name: str,
        cand_payload: Any,
        gt_payload: Any,
    ) -> tuple[Any, Any]:
        """Applique un profil de normalisation aux deux payloads.

        Si l'un des deux n'est pas une string, on saute la
        normalisation pour ce payload (cas typique : ALTO non encore
        projeté en texte → on laisse passer).
        """
        from picarones.formats.text.normalization import get_builtin_profile
        try:
            profile = get_builtin_profile(profile_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[view_executor] profil normalisation %r introuvable : %s",
                profile_name, exc,
            )
            return cand_payload, gt_payload
        normalized_cand = (
            profile.normalize(cand_payload)
            if isinstance(cand_payload, str)
            else cand_payload
        )
        normalized_gt = (
            profile.normalize(gt_payload)
            if isinstance(gt_payload, str)
            else gt_payload
        )
        return normalized_cand, normalized_gt

    @staticmethod
    def _failed_view_result(
        *,
        view: EvaluationView,
        candidate: Artifact,
        ground_truth: Artifact,
        projection_report: Any,
        global_error: str,
    ) -> ViewResult:
        """Construit un ``ViewResult`` quand le payload n'a pas pu
        être chargé.  Toutes les métriques sont marquées en échec
        avec le même message d'erreur global."""
        failed = {name: global_error for name in view.metric_names}
        return ViewResult(
            view_name=view.name,
            candidate_artifact_id=candidate.id,
            ground_truth_artifact_id=ground_truth.id,
            metric_values={},
            failed_metrics=failed,
            projection_report=projection_report,
            warnings=tuple(view.warnings) + (global_error,),
            ignored_dimensions=tuple(view.ignored_dimensions),
        )


__all__ = [
    "DefaultEvaluationViewExecutor",
    "PayloadLoader",
]
