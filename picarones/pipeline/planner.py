"""``PipelinePlanner`` — Sprint A14-S28.

Le S6 livrait ``validate_spec`` (validation statique : types
cohérents, IDs uniques, ``inputs_from`` valides, adapters connus).
Le S7 livrait ``PipelineExecutor`` qui résolvait les bindings
**au runtime** (bag versionné consulté à chaque step).

S28 introduit une couche de **planification** qui transforme une
``PipelineSpec`` en ``ExecutionPlan`` immuable :

1. Validation statique (délègue à ``validate_spec``).
2. Résolution explicite de chaque binding d'entrée — fini la
   résolution implicite « dernier producteur » au runtime.
3. Détection des **jonctions de métriques** : pour chaque sortie
   de step, le planner interroge le ``MetricRegistry`` pour les
   métriques applicables sur la signature ``(T, T)`` — base
   pour l'auto-évaluation contre la GT du même niveau.
4. Calcul d'un ordre topologique déterministe (les steps
   ``inputs_from`` peuvent référencer n'importe quelle étape
   antérieure ; le planner s'assure que la séquence est cohérente).

Pourquoi cette séparation
-------------------------
- **Contrat explicite** : l'executor consomme un ``ExecutionPlan``
  immuable plutôt que de dériver les bindings au runtime — moins
  de surprises, debug plus simple.
- **Réutilisabilité** : le ``CorpusRunner`` planifie **une fois**
  pour la spec, exécute N fois (un par document) — économie marginale
  mais clarté garantie.
- **Diagnostic** : un ``PlanningError`` capture toutes les erreurs
  d'un coup (pas de short-circuit à la première erreur).
- **Métriques de jonction** : le planner liste les métriques
  applicables à chaque sortie ; un service applicatif (S29+) peut
  pré-calculer où l'évaluation est possible.

Anti-sur-ingénierie
-------------------
- Pas de cache de plan inter-spec (le coût de planification est
  O(steps) et négligeable face à l'OCR).
- Pas d'optimisation de DAG (parallélisation, fusion, etc.) — le
  plan reste séquentiel et correspond exactement à l'ordre des
  steps.
- Pas de validation runtime additionnelle (artefacts effectivement
  produits, etc.) — c'est la responsabilité de l'executor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from picarones.domain.artifacts import ArtifactType
from picarones.domain.errors import PicaronesError
from picarones.evaluation.registry import MetricRegistry
from picarones.pipeline.spec import (
    INITIAL_STEP_ID,
    PipelineSpec,
    PipelineStep,
)
from picarones.pipeline.validation import ValidationError, validate_spec


# ──────────────────────────────────────────────────────────────────────
# Erreur dédiée
# ──────────────────────────────────────────────────────────────────────


class PlanningError(PicaronesError):
    """La spec n'a pas pu être planifiée — typiquement parce qu'elle
    contient des erreurs de validation détectées par
    ``validate_spec``.

    Attributes
    ----------
    errors:
        Liste des ``ValidationError`` produites par ``validate_spec``.
        Le caller peut les rendre dans son rapport (CLI, JSON, HTML)
        sans avoir à parser le message.
    """

    def __init__(
        self, message: str, errors: list[ValidationError] | None = None,
    ) -> None:
        super().__init__(message)
        self.errors: tuple[ValidationError, ...] = tuple(errors or ())


# ──────────────────────────────────────────────────────────────────────
# Modèles immuables du plan
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StepInputBinding:
    """Binding explicite d'une entrée de step à sa source.

    Attributes
    ----------
    input_type:
        Type d'artefact consommé.
    source_step_id:
        ID de l'étape source, ou ``INITIAL_STEP_ID`` pour les
        entrées initiales fournies au runner.

    Notes
    -----
    Frozen — le caller doit considérer le binding comme un fait
    figé du plan.  Toute mutation invaliderait l'``ExecutionPlan``.
    """

    input_type: ArtifactType
    source_step_id: str


@dataclass(frozen=True)
class ResolvedStep:
    """Étape avec tous ses bindings d'entrée résolus.

    Attributes
    ----------
    step:
        Le ``PipelineStep`` original (frozen pydantic).
    input_bindings:
        Bindings explicites — un par ``input_type``.  Préserve
        l'ordre de ``step.input_types``.

    Notes
    -----
    Le runner peut directement consommer ``input_bindings`` sans
    refaire la résolution : pour chaque binding, il sait quelle
    version de quel artefact aller chercher dans son bag.
    """

    step: PipelineStep
    input_bindings: tuple[StepInputBinding, ...] = field(default_factory=tuple)

    @property
    def id(self) -> str:
        return self.step.id

    @property
    def adapter_name(self) -> str:
        return self.step.adapter_name


@dataclass(frozen=True)
class MetricJunction:
    """Jonction de métriques détectée à la sortie d'un step.

    Pour chaque sortie ``T`` d'un step, le planner interroge le
    ``MetricRegistry`` pour les métriques de signature ``(T, T)``
    — celles qui peuvent comparer la sortie du step à une GT
    du même niveau.  Un service applicatif (S29+) consomme cette
    liste pour décider où auto-évaluer.

    Attributes
    ----------
    step_id:
        Step qui produit l'artefact évaluable.
    artifact_type:
        Type de l'artefact produit.
    candidate_metrics:
        Noms des métriques applicables, triés alphabétiquement
        pour déterminisme.

    Notes
    -----
    « Candidate » : la jonction est *applicable*, pas *exigée*.  Le
    caller décide selon la GT disponible et la stratégie d'évaluation.
    """

    step_id: str
    artifact_type: ArtifactType
    candidate_metrics: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ExecutionPlan:
    """Plan d'exécution immuable consommable par le ``PipelineExecutor``.

    Construit par ``PipelinePlanner.plan(spec)``.  Garantit que :

    - La spec est statiquement valide (toutes les ``ValidationError``
      sont nulles).
    - Chaque step a ses bindings résolus (``input_bindings`` non vide
      pour chaque ``input_type`` déclaré).
    - L'ordre topologique est respecté (``resolved_steps`` suit
      l'ordre de ``spec.steps``, qui doit déjà être topologique).
    - Les jonctions de métriques sont indexées par step.

    Attributes
    ----------
    spec:
        La ``PipelineSpec`` source (référence, pas copie).
    resolved_steps:
        Steps avec bindings résolus, dans l'ordre topologique
        d'exécution.
    metric_junctions:
        Jonctions auto-détectées si un ``MetricRegistry`` était
        fourni au planner ; tuple vide sinon.

        Sprint S54 — note honnête (audit #14) : à ce jour, le
        ``PipelineExecutor`` ne consomme pas ces jonctions au runtime
        (le calcul des métriques aux jonctions intra-pipeline est
        prévu dans un sprint dédié de l'axe « auto-évaluation »).
        Le champ est livré dès maintenant pour fixer le contrat —
        un caller peut déjà l'utiliser pour de l'introspection
        (rapport, diagnostic).  Pas de risque de breaking change
        quand l'auto-évaluation arrivera.
    """

    spec: PipelineSpec
    resolved_steps: tuple[ResolvedStep, ...] = field(default_factory=tuple)
    metric_junctions: tuple[MetricJunction, ...] = field(default_factory=tuple)

    def step_by_id(self, step_id: str) -> ResolvedStep | None:
        """Retourne le step résolu par son id, ou ``None``."""
        for rs in self.resolved_steps:
            if rs.id == step_id:
                return rs
        return None

    def junctions_for_step(self, step_id: str) -> tuple[MetricJunction, ...]:
        """Retourne les jonctions de métriques associées à un step."""
        return tuple(
            j for j in self.metric_junctions if j.step_id == step_id
        )


# ──────────────────────────────────────────────────────────────────────
# Planificateur
# ──────────────────────────────────────────────────────────────────────


class PipelinePlanner:
    """Planificateur d'une ``PipelineSpec`` en ``ExecutionPlan``.

    Parameters
    ----------
    metric_registry:
        Optionnel — si fourni, les jonctions de métriques sont
        détectées pour chaque sortie de step.  Sinon, le plan a
        ``metric_junctions=()``.
    available_adapters:
        Optionnel — set des noms d'adapters connus.  Si fourni, la
        validation rejette les ``adapter_name`` inconnus.  Sinon,
        cette validation est sautée (utile pour les YAML qui
        peuvent référencer des adapters tiers absents en CI).

    Notes
    -----
    Stateless : le planner ne mémorise aucun état entre appels.
    Thread-safe en lecture/écriture.
    """

    def __init__(
        self,
        metric_registry: MetricRegistry | None = None,
        available_adapters: set[str] | None = None,
    ) -> None:
        if metric_registry is not None and not isinstance(
            metric_registry, MetricRegistry,
        ):
            raise TypeError(
                "metric_registry doit être un MetricRegistry ou None."
            )
        self._metrics = metric_registry
        self._adapters = (
            frozenset(available_adapters)
            if available_adapters is not None
            else None
        )

    def plan(self, spec: PipelineSpec) -> ExecutionPlan:
        """Construit un ``ExecutionPlan`` à partir d'une ``PipelineSpec``.

        Étapes :

        1. ``validate_spec(spec, available_adapters)`` — récolte
           toutes les erreurs structurelles.
        2. Si erreurs → ``PlanningError`` avec la liste complète.
        3. Sinon, résout les bindings step par step en simulant le
           bag versionné.
        4. Si un registre de métriques est disponible, détecte les
           jonctions pour chaque sortie de step.

        Raises
        ------
        PlanningError
            Si la validation statique échoue.  Le caller peut
            inspecter ``error.errors`` pour rendre un rapport.
        """
        # 1. Validation statique.
        errors = validate_spec(
            spec,
            available_adapters=set(self._adapters) if self._adapters else None,
        )
        if errors:
            n = len(errors)
            preview = "; ".join(
                f"{e.step_id or '<global>'}:{e.code}"
                for e in errors[:3]
            )
            suffix = f" (+{n - 3} de plus)" if n > 3 else ""
            raise PlanningError(
                f"PipelineSpec {spec.name!r} a {n} erreur(s) de "
                f"validation : {preview}{suffix}",
                errors=errors,
            )

        # 2. Résolution des bindings.
        resolved_steps = self._resolve_steps(spec)

        # 3. Détection des jonctions de métriques.
        metric_junctions = (
            self._detect_junctions(spec)
            if self._metrics is not None
            else ()
        )

        return ExecutionPlan(
            spec=spec,
            resolved_steps=resolved_steps,
            metric_junctions=metric_junctions,
        )

    # ──────────────────────────────────────────────────────────────────
    # Helpers internes
    # ──────────────────────────────────────────────────────────────────

    def _resolve_steps(
        self, spec: PipelineSpec,
    ) -> tuple[ResolvedStep, ...]:
        """Résout les bindings de chaque step en simulant le bag.

        Pour chaque ``input_type`` d'un step :

        - Si ``inputs_from[input_type]`` est défini → ce step est la
          source explicite.
        - Sinon → la source est le **dernier producteur** du type
          dans l'ordre topologique (équivalent au comportement
          historique de l'executor S7).

        ``validate_spec`` garantit que ces résolutions sont valides
        (pas de référence pendante, type produit par la source).
        """
        latest_producer: dict[ArtifactType, str] = {
            t: INITIAL_STEP_ID for t in spec.initial_inputs
        }
        resolved: list[ResolvedStep] = []

        for step in spec.steps:
            bindings: list[StepInputBinding] = []
            for input_type in step.input_types:
                source = step.inputs_from.get(input_type)
                if source is None:
                    # validate_spec a vérifié que latest_producer[t]
                    # existe → on peut indexer sans garde.
                    source = latest_producer[input_type]
                bindings.append(StepInputBinding(
                    input_type=input_type,
                    source_step_id=source,
                ))
            resolved.append(ResolvedStep(
                step=step,
                input_bindings=tuple(bindings),
            ))
            # Mise à jour de l'état pour les steps suivants.
            for output_type in step.output_types:
                latest_producer[output_type] = step.id

        return tuple(resolved)

    def _detect_junctions(
        self, spec: PipelineSpec,
    ) -> tuple[MetricJunction, ...]:
        """Détecte les jonctions de métriques pour chaque sortie.

        Pour chaque ``output_type`` ``T`` d'un step, interroge le
        ``MetricRegistry`` pour les métriques de signature ``(T, T)``
        — métriques applicables à la comparaison ``GT[T]`` vs
        ``step.outputs[T]``.

        Si aucune métrique n'est applicable, la jonction est tout
        de même listée avec ``candidate_metrics=()`` — un caller
        peut ainsi détecter qu'un step produit un type non
        évaluable et décider de la suite (warning, registre étendu,
        omission).
        """
        # Garde-fou : devrait être garanti par le check dans plan().
        if self._metrics is None:  # pragma: no cover
            return ()
        junctions: list[MetricJunction] = []
        for step in spec.steps:
            for output_type in step.output_types:
                specs = self._metrics.select(output_type, output_type)
                names = tuple(sorted(s.name for s in specs))
                junctions.append(MetricJunction(
                    step_id=step.id,
                    artifact_type=output_type,
                    candidate_metrics=names,
                ))
        return tuple(junctions)


__all__ = [
    "ExecutionPlan",
    "MetricJunction",
    "PipelinePlanner",
    "PlanningError",
    "ResolvedStep",
    "StepInputBinding",
]
