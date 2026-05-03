"""``PipelineExecutor`` mono-document — Sprint A14-S7.

Première version réelle de l'exécuteur du nouveau pipeline.
Mono-document, séquentiel, capture gracieuse des erreurs par
étape.  L'orchestration corpus-wide (backpressure, timeout réel,
annulation propre) arrive au Sprint S8.

Contrat
-------
Le caller (typiquement un service applicatif au S19) fournit :

- une ``PipelineSpec`` validée (le caller doit avoir appelé
  ``validate_spec`` en amont — l'executor re-valide quand même
  pour défendre en profondeur),
- un ``DocumentRef`` du document à traiter,
- un dict ``{ArtifactType: Artifact}`` des entrées initiales
  (typiquement ``{IMAGE: Artifact(...)}``),
- un ``RunContext`` qui porte ``document_id``, ``code_version``,
  ``pipeline_name`` et un éventuel ``workspace_uri``,
- un ``adapter_resolver: Callable[[str], StepExecutor]`` qui
  résout ``adapter_name`` → instance d'adapter.  Au S19, ce
  resolver sera fourni par ``app/services/adapter_registry``.

L'executor garantit :

- Les étapes sont exécutées dans l'ordre de ``spec.steps``.
- Chaque entrée d'une étape est résolue depuis le **bag versionné** :
  si ``inputs_from[type] = "step_x"``, on prend la version
  produite par ``step_x`` ; sinon, on prend la dernière version
  disponible (comportement Sprint 66 historique).
- Toute exception levée par un adapter est capturée — le step
  est marqué ``succeeded=False`` avec ``error=str(exc)``, et le
  pipeline continue (les étapes en aval pourront échouer si
  elles dépendaient des outputs de ce step, ce qui est explicite).
- Les ``output_types`` déclarés par l'adapter sont validés au
  retour : si un type promis est manquant, le step est marqué
  en échec avec ``error="missing_output: <type>"``.

L'executor ne garantit PAS (reportés à des sprints suivants) :

- Mesure du temps depuis le début d'exécution réelle (S8 — pour
  l'instant, ``time.perf_counter()`` autour de ``execute()``).
- Annulation propre par signal aux workers en cours (S8).
- Cache d'artefacts inter-runs (S7 livre ``ArtifactCache`` mais
  l'executor ne s'y branche pas encore — ça vient quand on aura
  un cas d'usage concret de réutilisation).
- Parallélisation inter-documents ou inter-étapes (S8).

Définition de done du S7
------------------------
``PipelineExecutor.run(spec, document, initial_inputs, context)``
exécute une pipeline mock en moins de 100 ms et produit un
``PipelineResult`` complet (durées par étape, artefacts produits,
``succeeded`` agrégé).
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.domain.documents import DocumentRef
from picarones.domain.errors import PicaronesError
from picarones.pipeline.protocols import StepExecutor
from picarones.pipeline.spec import INITIAL_STEP_ID, PipelineSpec, PipelineStep
from picarones.pipeline.types import PipelineResult, RunContext, StepResult
from picarones.pipeline.validation import validate_spec

logger = logging.getLogger(__name__)


class PipelineSpecInvalid(PicaronesError):
    """``PipelineSpec`` mal formée — l'executor refuse de démarrer."""


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
    """

    def __init__(self, adapter_resolver: AdapterResolver) -> None:
        if not callable(adapter_resolver):
            raise PicaronesError(
                "PipelineExecutor : adapter_resolver doit être callable."
            )
        self._resolver = adapter_resolver

    def run(
        self,
        spec: PipelineSpec,
        document: DocumentRef,
        initial_inputs: dict[ArtifactType, Artifact],
        context: RunContext,
    ) -> PipelineResult:
        """Exécute une pipeline complète sur un document.

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
            Si ``validate_spec`` détecte des erreurs de
            cohérence.  L'executor ne masque pas ce type d'erreur :
            c'est un bug de programmation, pas un problème runtime.
        """
        # 1. Validation défensive.
        errors = validate_spec(spec)
        if errors:
            messages = "; ".join(
                f"{e.step_id or '<global>'}: {e.message}" for e in errors
            )
            raise PipelineSpecInvalid(
                f"Spec '{spec.name}' invalide : {messages}"
            )

        # 2. Bag versionné : map (type, step_id) → Artifact.
        #    Plus une map type → step_id "le plus récent" pour le
        #    fallback quand inputs_from ne précise pas la source.
        versioned: dict[tuple[ArtifactType, str], Artifact] = {}
        latest_producer: dict[ArtifactType, str] = {}

        for art_type, art in initial_inputs.items():
            versioned[(art_type, INITIAL_STEP_ID)] = art
            latest_producer[art_type] = INITIAL_STEP_ID

        # 3. Exécution séquentielle.
        step_results: list[StepResult] = []
        all_artifacts: list[Artifact] = list(initial_inputs.values())
        run_started = time.perf_counter()

        for step in spec.steps:
            result, produced = self._run_step(
                step=step,
                versioned=versioned,
                latest_producer=latest_producer,
                context=context,
            )
            step_results.append(result)
            for art_type, art in produced.items():
                versioned[(art_type, step.id)] = art
                latest_producer[art_type] = step.id
                all_artifacts.append(art)

        run_duration = time.perf_counter() - run_started
        succeeded = all(r.succeeded for r in step_results)

        return PipelineResult(
            pipeline_name=spec.name,
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
        step: PipelineStep,
        versioned: dict[tuple[ArtifactType, str], Artifact],
        latest_producer: dict[ArtifactType, str],
        context: RunContext,
    ) -> tuple[StepResult, dict[ArtifactType, Artifact]]:
        """Exécute une étape, retourne (result, artefacts produits).

        Le tuple est important : si le step échoue, on retourne quand
        même un dict vide pour les artefacts → le caller peut
        continuer la boucle proprement.
        """
        step_started = time.perf_counter()

        # 1. Résoudre les inputs depuis le bag.
        try:
            inputs = self._resolve_inputs(
                step=step,
                versioned=versioned,
                latest_producer=latest_producer,
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

    def _resolve_inputs(
        self,
        *,
        step: PipelineStep,
        versioned: dict[tuple[ArtifactType, str], Artifact],
        latest_producer: dict[ArtifactType, str],
    ) -> dict[ArtifactType, Artifact]:
        """Construit le dict ``{ArtifactType: Artifact}`` à passer
        à l'adapter, en respectant ``step.inputs_from``.

        Algorithme :

        - Pour chaque type dans ``step.input_types`` :
          - si ``step.inputs_from[type]`` est défini : exiger la
            version produite par cette étape, lever sinon ;
          - sinon : prendre la dernière version disponible
            (``latest_producer[type]``), lever si aucune.
        """
        inputs: dict[ArtifactType, Artifact] = {}
        for input_type in step.input_types:
            source_step = step.inputs_from.get(input_type)
            if source_step is None:
                source_step = latest_producer.get(input_type)
                if source_step is None:
                    raise _InputResolutionError(
                        f"missing_input: {input_type.value} "
                        "non disponible dans le bag d'artefacts"
                    )
            key = (input_type, source_step)
            if key not in versioned:
                raise _InputResolutionError(
                    f"missing_input: {input_type.value}"
                    f"@{source_step}"
                )
            inputs[input_type] = versioned[key]
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
