"""``DefaultEvaluationViewExecutor`` — Sprint A14-S13, refactoré au S27.

Implémentation concrète du protocole ``EvaluationViewExecutor`` (S5).
Orchestre une vue d'évaluation sur une paire (candidat, GT) en
**déléguant** la projection et l'évaluation à deux moteurs spécialisés
introduits au S27 :

- ``ProjectionEngine`` (cf. ``picarones/evaluation/projection_engine.py``)
  transforme l'artefact candidat selon la ``ProjectionSpec``.
- ``EvaluationEngine`` (cf. ``picarones/evaluation/evaluation_engine.py``)
  calcule les métriques sur les payloads.

Séquence d'orchestration
------------------------
1. Vérifie que ``candidate.type`` est dans ``view.candidate_types``.
2. ``ProjectionEngine.project(candidate, view.projection_for(candidate.type))``
   → retourne un ``ProjectionResult`` qui peut contenir un payload
   pré-calculé.
3. Charge les payloads (texte, ALTO parsé, etc.) via le
   ``payload_loader`` injecté.  Si la projection a produit un payload,
   l'utilise directement sans repasser par le loader.
4. Applique optionnellement un profil de normalisation texte
   (``view.normalization_profile``).
5. ``EvaluationEngine.evaluate(view.metric_names, gt_payload, cand_payload)``
   → retourne un ``EvaluationResult`` avec metric_values + failed_metrics.
6. Construit le ``ViewResult`` agrégeant tout (projection_report,
   metric_values, failed_metrics, warnings, ignored_dimensions).

Construction
------------
- ``__init__`` canonique prend ``(projection_engine, evaluation_engine,
  payload_loader)``.
- ``from_registries(metric_registry, projector_registry, payload_loader)``
  reste exposé comme classmethod ergonomique pour les callers qui
  n'ont pas envie de fabriquer eux-mêmes les deux moteurs (tests,
  scripts ad-hoc).  Aucune logique nouvelle — uniquement un appel
  composé ; l'API canonique reste l'injection des deux engines.

Anti-sur-ingénierie
-------------------
Pas de cache de payload chargé entre métriques (chaque appel à
``evaluate`` est indépendant).  Pas de batch (évaluer N paires en
une passe).  Pas de validation cross-métrique.  La complexité vit
dans les engines, pas dans l'executor.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from picarones.domain.artifacts import Artifact
from picarones.domain.evaluation_spec import EvaluationView
from picarones.evaluation.evaluation_engine import EvaluationEngine
from picarones.evaluation.projection_engine import ProjectionEngine
from picarones.evaluation.projectors.registry import ProjectorRegistry
from picarones.evaluation.registry import MetricRegistry
from picarones.evaluation.views.base import ViewResult

logger = logging.getLogger(__name__)


#: Type alias : un payload loader prend un Artifact et retourne le
#: contenu chargé (str pour RAW_TEXT, dict pour ENTITIES, etc.).
PayloadLoader = Callable[[Artifact], Any]


class DefaultEvaluationViewExecutor:
    """Orchestrateur de vue d'évaluation.

    Parameters
    ----------
    projection_engine:
        ``ProjectionEngine`` injecté.  Responsable de la
        transformation d'artefacts entre types via le registre de
        projecteurs.
    evaluation_engine:
        ``EvaluationEngine`` injecté.  Responsable du calcul des
        métriques nommées sur des payloads.
    payload_loader:
        Callable ``(Artifact) -> Any`` qui charge le contenu d'un
        artefact non encore résolu (typiquement la GT et le candidat
        s'il n'est pas projeté).  Pour les tests, un dict in-memory
        ; en production, un service applicatif qui sait gérer les
        workspaces sandboxés.
    """

    def __init__(
        self,
        projection_engine: ProjectionEngine,
        evaluation_engine: EvaluationEngine,
        payload_loader: PayloadLoader,
    ) -> None:
        if not isinstance(projection_engine, ProjectionEngine):
            raise TypeError(
                "projection_engine doit être un ProjectionEngine."
            )
        if not isinstance(evaluation_engine, EvaluationEngine):
            raise TypeError(
                "evaluation_engine doit être un EvaluationEngine."
            )
        if not callable(payload_loader):
            raise TypeError("payload_loader doit être callable.")
        self._projection = projection_engine
        self._evaluation = evaluation_engine
        self._loader = payload_loader

    # ──────────────────────────────────────────────────────────────────
    # Constructeur ergonomique
    # ──────────────────────────────────────────────────────────────────

    @classmethod
    def from_registries(
        cls,
        metric_registry: MetricRegistry,
        projector_registry: ProjectorRegistry,
        payload_loader: PayloadLoader,
    ) -> "DefaultEvaluationViewExecutor":
        """Construit l'executor à partir des registres bruts.

        Sucre syntaxique sur l'API canonique : un caller qui a déjà
        un ``MetricRegistry`` + ``ProjectorRegistry`` (cas typique :
        un test, ou un service qui n'a qu'un seul executor) gagne
        deux lignes.  Aucune logique nouvelle — instancie
        ``ProjectionEngine`` et ``EvaluationEngine`` puis délègue.
        """
        return cls(
            projection_engine=ProjectionEngine(projector_registry),
            evaluation_engine=EvaluationEngine(metric_registry),
            payload_loader=payload_loader,
        )

    # ──────────────────────────────────────────────────────────────────
    # API publique
    # ──────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        view: EvaluationView,
        candidate: Artifact,
        ground_truth: Artifact,
        *,
        pipeline_name: str,
    ) -> ViewResult:
        """Évalue la vue sur la paire (candidat, GT).

        Returns
        -------
        ViewResult
            Toujours retourné en sortie normale — les erreurs de
            métriques individuelles vont dans ``failed_metrics``,
            les erreurs de chargement de payload se traduisent en
            ``failed_metrics`` global.

        Raises
        ------
        ProjectionError
            Si la vue exige une projection que le projecteur ne
            peut pas réaliser (cohérent avec le contrat du S5).
        ValueError
            Si ``candidate.type`` n'est pas dans
            ``view.candidate_types``.  Le caller (typiquement le
            ``BenchmarkService``) doit filtrer les pipelines qui ne
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

        # 2. Projection (déléguée).  Lève ``ProjectionError`` si la
        #    projection est invalide — on laisse remonter (cohérence
        #    avec le contrat S5).
        projection_spec = view.projection_for(candidate.type)
        projection_result = self._projection.project(
            candidate, projection_spec,
        )

        # 3. Chargement des payloads.
        # Si la projection a fourni un payload, on l'utilise sans
        # repasser par le loader (typique S25 — l'artefact projeté
        # n'a pas d'URI).  Sinon, on charge le candidat via le loader.
        if projection_result.payload is not None:
            cand_payload = projection_result.payload
        else:
            try:
                cand_payload = self._loader(projection_result.artifact)
            except Exception as exc:  # noqa: BLE001
                return self._failed_view_result(
                    view=view,
                    pipeline_name=pipeline_name,
                    candidate=candidate,
                    ground_truth=ground_truth,
                    projection_report=projection_result.report,
                    global_error=(
                        f"payload_loader a échoué sur le candidat "
                        f"{projection_result.artifact.id!r} : {exc}"
                    ),
                )
        try:
            gt_payload = self._loader(ground_truth)
        except Exception as exc:  # noqa: BLE001
            return self._failed_view_result(
                view=view,
                pipeline_name=pipeline_name,
                candidate=candidate,
                ground_truth=ground_truth,
                projection_report=projection_result.report,
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

        # 4.bis. Phase B2.5 — filtrage des caractères (optionnel).
        # Appliqué APRÈS la normalisation pour que le profil de
        # normalisation puisse encore voir les caractères filtrés
        # (cohérent avec ``compute_metrics`` legacy qui exclut avant
        # tout calcul mais après les transformations de normalisation
        # implicites du flow).
        if view.char_exclude:
            cand_payload, gt_payload = self._apply_char_exclude(
                view.char_exclude, cand_payload, gt_payload,
            )

        # 5. Évaluation déléguée.  Une métrique cassée → failed_metrics.
        evaluation_result = self._evaluation.evaluate(
            view.metric_names, gt_payload, cand_payload,
        )

        # 6. Agrégation finale dans le ViewResult.
        warnings = tuple(view.warnings)
        ignored = tuple(view.ignored_dimensions)
        if projection_result.report is not None:
            warnings = warnings + tuple(projection_result.report.warnings)
            seen: set[str] = set(ignored)
            extra = tuple(
                d for d in projection_result.report.ignored_dimensions
                if d not in seen
            )
            ignored = ignored + extra

        return ViewResult(
            view_name=view.name,
            pipeline_name=pipeline_name,
            candidate_artifact_id=candidate.id,
            ground_truth_artifact_id=ground_truth.id,
            metric_values=evaluation_result.metric_values,
            failed_metrics=evaluation_result.failed_metrics,
            projection_report=projection_result.report,
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
    def _apply_char_exclude(
        char_exclude: str,
        cand_payload: Any,
        gt_payload: Any,
    ) -> tuple[Any, Any]:
        """Phase B2.5 — filtre les caractères de ``char_exclude`` des
        deux payloads avant comparaison.

        Sémantique strictement identique à ``compute_metrics``
        (legacy) : ``"".join(c for c in s if c not in char_exclude)``.

        Si un payload n'est pas une string (ex : ALTO non projeté),
        il est laissé tel quel — cohérent avec le filet
        ``_apply_normalization`` qui saute les non-strings.
        """
        exclude_set = frozenset(char_exclude)
        if not exclude_set:
            return cand_payload, gt_payload
        filtered_cand = (
            "".join(c for c in cand_payload if c not in exclude_set)
            if isinstance(cand_payload, str)
            else cand_payload
        )
        filtered_gt = (
            "".join(c for c in gt_payload if c not in exclude_set)
            if isinstance(gt_payload, str)
            else gt_payload
        )
        return filtered_cand, filtered_gt

    @staticmethod
    def _failed_view_result(
        *,
        view: EvaluationView,
        pipeline_name: str,
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
            pipeline_name=pipeline_name,
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
