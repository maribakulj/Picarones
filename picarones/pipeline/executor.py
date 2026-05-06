"""``PipelineExecutor`` mono-document — Sprints A14-S7 / S28.

Exécuteur séquentiel d'une pipeline composée sur un document.

Sprint S7 livrait ``run(spec, document, initial_inputs, context)``
qui validait la spec en interne et résolvait les bindings au
runtime via un bag versionné.

Sprint S28 introduit le ``PipelinePlanner`` qui transforme une
``PipelineSpec`` en ``ExecutionPlan`` immuable (validations +
bindings résolus + jonctions de métriques détectées).  L'executor
consomme désormais soit :

- Un ``ExecutionPlan`` pré-calculé via ``run_plan(plan, ...)`` —
  signature canonique, contrat explicite.
- Une ``PipelineSpec`` brute via ``run(spec, ...)`` — sucre
  ergonomique qui appelle le planner en interne (planification
  systématique, pas de cache implicite).

Contrat
-------
Le caller (typiquement ``BenchmarkService`` ou ``CorpusRunner``)
fournit :

- un ``ExecutionPlan`` (canonique) ou ``PipelineSpec`` (sucre),
- un ``DocumentRef`` du document à traiter,
- un dict ``{ArtifactType: Artifact}`` des entrées initiales
  (typiquement ``{IMAGE: Artifact(...)}``),
- un ``RunContext`` (``document_id``, ``code_version``,
  ``pipeline_name``, éventuel ``workspace_uri``),
- un ``adapter_resolver: Callable[[str], StepExecutor]`` injecté
  au constructeur.

L'executor garantit :

- Les étapes sont exécutées dans l'ordre du plan
  (``resolved_steps``).
- Chaque entrée d'une étape est résolue depuis les
  ``StepInputBinding`` du plan — fini la résolution implicite
  « dernier producteur » au runtime.
- Toute exception levée par un adapter est capturée — le step
  est marqué ``succeeded=False`` avec ``error=str(exc)``, et le
  pipeline continue (les étapes en aval pourront échouer si elles
  dépendaient des outputs de ce step, ce qui est explicite).
- Les ``output_types`` déclarés par l'adapter sont validés au
  retour : un type promis manquant marque le step en échec avec
  ``error="missing_output: <type>"``.

L'executor ne garantit PAS (reportés à des sprints suivants) :

- Cache d'artefacts inter-runs (S29 livre ``ArtifactStore``).
- Parallélisation inter-documents ou inter-étapes (cf. S8 pour
  inter-doc via ``CorpusRunner``).

Compat S7
---------
La signature historique ``run(spec, document, ...)`` reste
exposée — elle planifie la spec systématiquement à chaque appel
et délègue à ``run_plan``.  Aucune logique nouvelle n'y vit.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.domain.documents import DocumentRef
from picarones.domain.errors import PicaronesError
from picarones.pipeline.cache_helpers import (
    compute_step_artifact_key,
    read_cached_outputs,
    write_outputs_to_cache,
)
from picarones.pipeline.cache_protocol import ArtifactCachePort
from picarones.pipeline.planner import (
    ExecutionPlan,
    PipelinePlanner,
    PlanningError,
    ResolvedStep,
)
from picarones.pipeline.protocols import StepExecutor
from picarones.domain.pipeline_spec import INITIAL_STEP_ID, PipelineSpec
from picarones.pipeline.types import PipelineResult, RunContext, StepResult

logger = logging.getLogger(__name__)


class PipelineSpecInvalid(PicaronesError):
    """``PipelineSpec`` mal formée — l'executor refuse de démarrer.

    Wrappe le ``PlanningError`` produit par ``PipelinePlanner`` pour
    préserver la sémantique historique : un caller qui catchait
    ``PipelineSpecInvalid`` continue de fonctionner.
    """


#: Type alias pour le resolver d'adapters.  Une fonction qui
#: prend un ``adapter_name`` (str) et retourne une instance
#: ``StepExecutor`` prête à l'emploi.  Si le resolver lève
#: ``KeyError``, l'executor traduit en step en échec avec
#: ``error="adapter_not_found: ..."``.
AdapterResolver = Callable[[str], StepExecutor]


class PipelineExecutor:
    """Exécuteur séquentiel mono-document.

    Une instance peut traiter plusieurs documents (l'état est
    porté par les paramètres de ``run()``, pas par le constructeur).
    L'instance est thread-safe en lecture (rien n'est muté après
    construction).

    Parameters
    ----------
    adapter_resolver:
        Callable qui résout un ``adapter_name`` en instance
        ``StepExecutor``.  Typiquement
        ``lambda name: registry[name]`` en test, ou un service
        applicatif qui injecte les bonnes dépendances en prod.
    planner:
        ``PipelinePlanner`` injecté (S28).  Si ``None``, un planner
        par défaut sans ``MetricRegistry`` est instancié.
    artifact_store:
        ``ArtifactStore`` optionnel (S29 + S47) pour la **reprise par
        hash**.  Si fourni, l'executor :

        - **avant** chaque step, calcule la clé du step via
          ``compute_step_artifact_key`` et interroge le store ; si
          toutes les sorties attendues sont présentes ET valides
          (URIs accessibles), saute l'exécution et retourne les
          artefacts cachés (``StepResult.duration_seconds=0.0``) ;
        - **après** chaque step réussi, persiste les outputs dans
          le store sous la clé dérivée.

        Si ``None`` (défaut), aucun cache n'est consulté ni écrit.
        Le comportement est strictement identique à l'avant-S47.
    """

    def __init__(
        self,
        adapter_resolver: AdapterResolver,
        planner: PipelinePlanner | None = None,
        artifact_store: ArtifactCachePort | None = None,
    ) -> None:
        if not callable(adapter_resolver):
            raise PicaronesError(
                "PipelineExecutor : adapter_resolver doit être callable."
            )
        if planner is not None and not isinstance(planner, PipelinePlanner):
            raise PicaronesError(
                "PipelineExecutor : planner doit être un PipelinePlanner ou None."
            )
        # ``isinstance(artifact_store, ArtifactCachePort)`` est un duck
        # typing check (Protocol @runtime_checkable) — valide get/put/
        # __contains__ par leur seule présence.  Permet à un caller
        # tiers (Redis, S3) de fournir un store custom satisfaisant
        # le protocol sans hériter de la classe ABC ``ArtifactStore``.
        if artifact_store is not None and not isinstance(
            artifact_store, ArtifactCachePort,
        ):
            raise PicaronesError(
                "PipelineExecutor : artifact_store doit satisfaire le "
                "protocole ArtifactCachePort (get / put / __contains__) "
                "ou être None.",
            )
        self._resolver = adapter_resolver
        # Si pas de planner injecté, on en fabrique un sans MetricRegistry —
        # les jonctions seront vides mais la planification reste correcte.
        self._planner = planner if planner is not None else PipelinePlanner()
        self._artifact_store = artifact_store

    def plan(self, spec: PipelineSpec) -> ExecutionPlan:
        """Planifie une ``PipelineSpec`` en ``ExecutionPlan``.

        Sucre exposant le planner injecté.  Permet aux callers
        (typiquement ``CorpusRunner`` qui exécute la même spec sur
        N documents) de planifier **une fois** puis appeler
        ``run_plan`` N fois — économisant N-1 validations.

        Raises
        ------
        PipelineSpecInvalid
            Si la planification échoue (validations statiques).
        """
        try:
            return self._planner.plan(spec)
        except PlanningError as exc:
            messages = "; ".join(
                f"{e.step_id or '<global>'}: {e.message}"
                for e in exc.errors
            )
            raise PipelineSpecInvalid(
                f"Spec {spec.name!r} invalide : {messages}"
            ) from exc

    def run(
        self,
        spec: PipelineSpec,
        document: DocumentRef,
        initial_inputs: dict[ArtifactType, Artifact],
        context: RunContext,
    ) -> PipelineResult:
        """Exécute une pipeline complète sur un document (sucre).

        Sucre ergonomique sur ``run_plan`` : appelle
        ``self._planner.plan(spec)`` puis ``run_plan(plan, ...)``.
        Aucune logique nouvelle n'y vit — l'API canonique est
        ``run_plan(plan, document, initial_inputs, context)`` qui
        accepte un ``ExecutionPlan`` pré-calculé.

        Returns
        -------
        PipelineResult
            ``succeeded`` global = True ssi toutes les étapes ont
            réussi.  Une étape en échec n'arrête PAS l'exécution —
            les étapes suivantes peuvent quand même tourner si
            leurs entrées ne dépendent pas du step en échec.

        Raises
        ------
        PipelineSpecInvalid
            Si la planification échoue (validations statiques).
            L'executor ne masque pas ce type d'erreur : c'est un
            bug de programmation, pas un problème runtime.
        """
        plan = self.plan(spec)
        return self.run_plan(plan, document, initial_inputs, context)

    def run_plan(
        self,
        plan: ExecutionPlan,
        document: DocumentRef,
        initial_inputs: dict[ArtifactType, Artifact],
        context: RunContext,
    ) -> PipelineResult:
        """Exécute un ``ExecutionPlan`` pré-calculé sur un document.

        Signature canonique du S28.  Le caller a déjà appelé
        ``planner.plan(spec)`` (typiquement ``CorpusRunner`` qui
        planifie une fois pour N documents).  L'executor consomme
        directement ``plan.resolved_steps`` sans re-valider la
        spec ni re-résoudre les bindings.

        Toute la logique d'exécution vit ici ; ``run`` n'est qu'un
        sucre.
        """
        if not isinstance(plan, ExecutionPlan):
            raise PicaronesError(
                f"run_plan : plan doit être un ExecutionPlan, "
                f"reçu {type(plan).__name__}"
            )

        # 1. Bag versionné : map (type, step_id) → Artifact.
        versioned: dict[tuple[ArtifactType, str], Artifact] = {}
        for art_type, art in initial_inputs.items():
            versioned[(art_type, INITIAL_STEP_ID)] = art

        # 2. Exécution séquentielle des steps résolus.
        step_results: list[StepResult] = []
        all_artifacts: list[Artifact] = list(initial_inputs.values())
        run_started = time.perf_counter()

        for resolved_step in plan.resolved_steps:
            result, produced = self._run_step(
                resolved_step=resolved_step,
                versioned=versioned,
                context=context,
            )
            step_results.append(result)
            for art_type, art in produced.items():
                versioned[(art_type, resolved_step.id)] = art
                all_artifacts.append(art)

        run_duration = time.perf_counter() - run_started
        succeeded = all(r.succeeded for r in step_results)

        return PipelineResult(
            pipeline_name=plan.spec.name,
            document_id=document.id,
            step_results=tuple(step_results),
            succeeded=succeeded,
            duration_seconds=run_duration,
            artifacts=tuple(all_artifacts),
        )

    # ──────────────────────────────────────────────────────────────────
    # Helpers internes
    # ──────────────────────────────────────────────────────────────────

    def _run_step(
        self,
        *,
        resolved_step: ResolvedStep,
        versioned: dict[tuple[ArtifactType, str], Artifact],
        context: RunContext,
    ) -> tuple[StepResult, dict[ArtifactType, Artifact]]:
        """Exécute une étape résolue, retourne (result, artefacts produits).

        Le tuple est important : si le step échoue, on retourne quand
        même un dict vide pour les artefacts → le caller peut
        continuer la boucle proprement.
        """
        step = resolved_step.step
        step_started = time.perf_counter()

        # 1. Résoudre les inputs depuis le bag en suivant les bindings
        #    explicites du plan.
        try:
            inputs = self._inputs_from_bindings(
                resolved_step=resolved_step,
                versioned=versioned,
            )
        except _InputResolutionError as exc:
            duration = time.perf_counter() - step_started
            return (
                StepResult(
                    step_id=step.id,
                    succeeded=False,
                    duration_seconds=duration,
                    error=str(exc),
                ),
                {},
            )

        # 1bis. S47 — Reprise par hash via ArtifactStore.
        # Si un store est injecté et que tous les inputs ont un
        # ``content_hash``, on calcule la clé du step et on interroge
        # le store.  Hit complet → on saute l'exécution (durée 0,
        # même artefacts que la dernière exécution réussie).  Miss
        # ou cache partiel → on tombe dans l'exécution normale.
        if self._artifact_store is not None:
            cached_outputs = self._try_resume_from_cache(
                step=step, inputs=inputs, context=context,
            )
            if cached_outputs is not None:
                logger.info(
                    "[pipeline:%s] step '%s' : hit cache "
                    "(reprise par hash, exécution sautée).",
                    context.pipeline_name, step.id,
                )
                return (
                    StepResult(
                        step_id=step.id,
                        succeeded=True,
                        duration_seconds=0.0,
                        produced_artifacts={
                            t.value: a.id
                            for t, a in cached_outputs.items()
                        },
                    ),
                    cached_outputs,
                )

        # 2. Résoudre l'adapter.
        try:
            adapter = self._resolver(step.adapter_name)
        except KeyError:
            duration = time.perf_counter() - step_started
            return (
                StepResult(
                    step_id=step.id,
                    succeeded=False,
                    duration_seconds=duration,
                    error=f"adapter_not_found: {step.adapter_name}",
                ),
                {},
            )
        except Exception as exc:  # noqa: BLE001
            duration = time.perf_counter() - step_started
            return (
                StepResult(
                    step_id=step.id,
                    succeeded=False,
                    duration_seconds=duration,
                    error=f"adapter_resolver_failed: {exc}",
                ),
                {},
            )

        # 3. Exécuter.  Toute exception est capturée → step en échec.
        try:
            outputs = adapter.execute(inputs, dict(step.params), context)
        except Exception as exc:  # noqa: BLE001
            duration = time.perf_counter() - step_started
            logger.warning(
                "[pipeline:%s] step '%s' a levé : %s",
                context.pipeline_name, step.id, exc,
            )
            return (
                StepResult(
                    step_id=step.id,
                    succeeded=False,
                    duration_seconds=duration,
                    error=f"adapter_raised: {type(exc).__name__}: {exc}",
                ),
                {},
            )

        # 4. Valider les outputs déclarés.
        missing = [
            t for t in step.output_types
            if t not in outputs
        ]
        duration = time.perf_counter() - step_started
        if missing:
            return (
                StepResult(
                    step_id=step.id,
                    succeeded=False,
                    duration_seconds=duration,
                    error=(
                        "missing_output: "
                        f"{[t.value for t in missing]}"
                    ),
                ),
                # On garde quand même les outputs qui ont été produits,
                # pour que les éventuels steps en aval puissent les
                # utiliser si la pipeline est résiliente.
                outputs,
            )

        # 5. Succès.
        # S47 — persiste les outputs dans le store si fourni.  La
        # méthode interne sait gérer le cas content_hash manquant
        # (skip silencieux) — on lui passe la responsabilité.
        if self._artifact_store is not None:
            self._persist_to_cache(
                step=step, inputs=inputs, context=context, outputs=outputs,
            )
        produced_map = {
            t.value: a.id for t, a in outputs.items()
        }
        return (
            StepResult(
                step_id=step.id,
                succeeded=True,
                duration_seconds=duration,
                produced_artifacts=produced_map,
            ),
            outputs,
        )

    # ──────────────────────────────────────────────────────────────────
    # S47 — Reprise par hash via ArtifactStore
    # ──────────────────────────────────────────────────────────────────

    def _try_resume_from_cache(
        self,
        *,
        step,
        inputs: dict[ArtifactType, Artifact],
        context: RunContext,
    ) -> dict[ArtifactType, Artifact] | None:
        """Tente de retrouver les outputs cachés du step.

        Retourne ``None`` (cache miss) dans 3 cas :

        1. Un input n'a pas de ``content_hash`` → la clé n'est pas
           calculable (cf. ``ArtifactKey.hash_hex``).
        2. Le store ne contient pas TOUS les ``output_types`` du step.
        3. Une URI cachée pointe vers un fichier qui n'existe plus.
        """
        # Nécessairement non-None ici (vérifié par le caller), mais on
        # défend en profondeur.
        if self._artifact_store is None:
            return None
        key = compute_step_artifact_key(step, inputs, context)
        step_hash = key.hash_hex()
        if step_hash is None:
            return None
        return read_cached_outputs(
            store=self._artifact_store,
            step=step,
            step_hash=step_hash,
        )

    def _persist_to_cache(
        self,
        *,
        step,
        inputs: dict[ArtifactType, Artifact],
        context: RunContext,
        outputs: dict[ArtifactType, Artifact],
    ) -> None:
        """Persiste les outputs d'un step réussi dans le store.

        Skip silencieux si la clé n'est pas calculable (un input sans
        ``content_hash``).
        """
        if self._artifact_store is None:
            return
        key = compute_step_artifact_key(step, inputs, context)
        step_hash = key.hash_hex()
        if step_hash is None:
            return
        write_outputs_to_cache(
            store=self._artifact_store,
            step=step,
            step_hash=step_hash,
            outputs=outputs,
        )

    def _inputs_from_bindings(
        self,
        *,
        resolved_step: ResolvedStep,
        versioned: dict[tuple[ArtifactType, str], Artifact],
    ) -> dict[ArtifactType, Artifact]:
        """Construit le dict ``{ArtifactType: Artifact}`` à passer
        à l'adapter à partir des bindings explicites du plan.

        Le plan a déjà résolu chaque ``input_type`` à une
        ``source_step_id`` (soit ``INITIAL_STEP_ID``, soit l'ID
        d'une étape antérieure).  L'executor n'a plus qu'à indexer
        le bag par ``(input_type, source_step_id)``.

        Lève ``_InputResolutionError`` si l'artefact attendu
        n'est pas dans le bag — typiquement parce qu'une étape
        antérieure a échoué et n'a pas produit son output.
        """
        inputs: dict[ArtifactType, Artifact] = {}
        for binding in resolved_step.input_bindings:
            key = (binding.input_type, binding.source_step_id)
            if key not in versioned:
                raise _InputResolutionError(
                    f"missing_input: {binding.input_type.value}"
                    f"@{binding.source_step_id}"
                )
            inputs[binding.input_type] = versioned[key]
        return inputs


class _InputResolutionError(Exception):
    """Erreur interne signalant qu'un input n'a pas pu être résolu.

    Capturée par ``_run_step`` qui la traduit en ``StepResult``
    en échec avec ``error="missing_input: ..."``.
    """


__all__ = [
    "AdapterResolver",
    "PipelineExecutor",
    "PipelineSpecInvalid",
]
