"""``RunContext``, ``StepResult``, ``PipelineResult``

Types runtime du pipeline executor (à implémenter au Sprint S7).
Distincts des specs déclaratives (``picarones.pipeline.spec``) —
ces types portent les **résultats** de l'exécution, pas la
description du DAG.

Aucune logique métier ici : juste des dataclasses pydantic qu'un
service applicatif peut sérialiser dans le manifest d'un run.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from picarones.domain.artifacts import Artifact


class RunContext(BaseModel):
    """Contexte d'exécution passé à chaque ``StepExecutor.execute()``.

    Le caller (typiquement ``app/services/benchmark_service`` au
    S19) construit un ``RunContext`` par document et le passe à
    l'executor pour chaque étape.

    Attributs
    ---------
    document_id:
        ``DocumentRef.id`` du document en cours de traitement.
    code_version:
        Version du code (``picarones.__version__``) au moment du
        run.  Sert à étiqueter la ``ProvenanceRecord`` de chaque
        artefact produit.
    pipeline_name:
        Nom de la pipeline en cours.  Permet à un adapter de
        loguer ``[pipeline_x] step_y : ...`` plutôt que
        ``[unknown] ...``.
    workspace_uri:
        URI/chemin du workspace dans lequel l'adapter peut écrire
        ses artefacts intermédiaires.  ``None`` autorisé pour les
        adapters qui n'écrivent rien sur disque (mode in-memory).

    Anti-sur-ingénierie : pas de logger injecté, pas d'horloge
    abstraite, pas de cancellation token.  Ces extras viendront
    quand un caller en aura concrètement besoin (probablement S7
    pour la cancellation, S8 pour le timeout réel).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str = Field(min_length=1, max_length=256)
    code_version: str = Field(min_length=1, max_length=128)
    pipeline_name: str = Field(min_length=1, max_length=128)
    workspace_uri: str | None = Field(default=None, max_length=2048)


class StepResult(BaseModel):
    """Résultat de l'exécution d'une étape sur un document.

    Sérialisable JSON pour persistance dans le manifest du run.

    Attributs
    ---------
    step_id:
        Identifiant de l'étape (cf. ``PipelineStep.id``).
    succeeded:
        ``True`` si l'étape s'est exécutée sans lever d'exception
        et a produit tous les types déclarés dans
        ``output_types``.  ``False`` sinon.
    duration_seconds:
        Wall-clock time de ``execute()`` (du début effectif à la
        fin).  L'executor du S8 garantira que ce temps est mesuré
        depuis le démarrage réel (pas depuis la submission au pool).
    produced_artifacts:
        Map ``{ArtifactType: artifact_id}`` des artefacts produits.
        Vide en cas d'échec.
    error:
        ``None`` en cas de succès ; sinon message d'erreur.  Format
        libre (le caller décide de la structure dans son rapport).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    step_id: str = Field(min_length=1, max_length=128)
    succeeded: bool
    duration_seconds: float = Field(ge=0.0)
    produced_artifacts: dict[str, str] = Field(default_factory=dict)
    """Map ``{ArtifactType.value: Artifact.id}``.

    Sérialisée avec la valeur string de l'enum (``"raw_text"``,
    ``"alto_xml"``) pour faciliter la lecture humaine du JSON.
    """
    error: str | None = None


class PipelineResult(BaseModel):
    """Résultat complet d'une exécution de pipeline sur un document.

    Attributs
    ---------
    pipeline_name:
        Nom de la pipeline qui a produit ce résultat.
    document_id:
        Document traité.
    step_results:
        Résultats de chaque étape, dans l'ordre d'exécution.
    succeeded:
        ``True`` ssi tous les ``step_results`` sont des succès.
        Si ``False``, un ou plusieurs ``StepResult.error`` sont
        non-None.
    duration_seconds:
        Wall-clock total (somme des étapes + overhead orchestration).
    artifacts:
        Liste **plate** de tous les artefacts produits par la
        pipeline.  Permet à un consommateur (rapport, vue
        d'évaluation) d'accéder directement à un artefact par son
        id sans parcourir l'arborescence des étapes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pipeline_name: str
    document_id: str
    step_results: tuple[StepResult, ...] = Field(default_factory=tuple)
    succeeded: bool = False
    duration_seconds: float = Field(default=0.0, ge=0.0)
    artifacts: tuple[Artifact, ...] = Field(default_factory=tuple)

    def step_result_by_id(self, step_id: str) -> StepResult | None:
        for r in self.step_results:
            if r.step_id == step_id:
                return r
        return None

    def artifacts_of_type(self, artifact_type: Any) -> tuple[Artifact, ...]:
        """Retourne tous les artefacts du type donné dans l'ordre
        de production."""
        return tuple(a for a in self.artifacts if a.type == artifact_type)


__all__ = ["RunContext", "StepResult", "PipelineResult"]
