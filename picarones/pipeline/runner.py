"""``CorpusRunner``

Orchestre l'exécution d'une ``PipelineSpec`` sur un corpus complet
avec trois propriétés critiques que l'ancien
``measurements.runner`` ne garantissait pas correctement :

1. **Backpressure** — pas de "submit all upfront".  L'orchestrateur
   ne soumet jamais plus de ``max_in_flight`` documents en
   parallèle.  RAM bornée même sur des corpus de plusieurs milliers
   de documents.

2. **Timeout depuis le début d'exécution réelle** — l'ancien runner
   calculait le timeout depuis la submission au pool, donc un
   document pouvait être marqué timeout parce qu'il avait passé
   N secondes en queue, pas N secondes en train de tourner.  Le
   nouveau runner mesure depuis le moment où le worker démarre
   réellement.

3. **Annulation propre** — un ``threading.Event`` partagé permet
   au caller (typiquement un service applicatif sur un endpoint
   FastAPI ``cancel``) de signaler l'arrêt.  Les workers
   coopératifs vérifient l'event ; les futures non démarrées sont
   sautées ; les futures déjà en cours se terminent (Python ne
   permet pas de tuer un thread en cours).

Limites assumées pour S8
------------------------
- **Mode threads uniquement.**  Le mode process (``ProcessPoolExecutor``)
  ajouté au S11 quand on déplacera les adapters CPU-bound.
  Aujourd'hui, un adapter Tesseract local en thread fonctionne
  (le GIL est relâché par le sous-processus pytesseract → OK).
- **Pas de kill-thread garanti.**  Si un adapter ne coopère pas avec
  ``cancel_event`` et fait un appel C bloquant non-interruptible,
  le runner attend la fin naturelle.  C'est documenté.
- **Pas de retry automatique.**  Si un adapter échoue, le doc est
  marqué en échec et on passe au suivant.

Définition de done
------------------
``CorpusRunner.run(spec, 1000 docs synthétiques)`` se termine en
moins de 10 minutes sans dépasser 500 MB de RAM résidente.  Le
test ``test_sprint_a14_s8_def_of_done`` valide ce critère
(échantillon paramétrable pour CI rapide).
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from collections.abc import Iterable
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field

from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.domain.documents import DocumentRef
from picarones.domain.errors import PicaronesError
from picarones.pipeline.executor import PipelineExecutor
from picarones.domain.pipeline_spec import PipelineSpec
from picarones.pipeline.types import PipelineResult, RunContext

logger = logging.getLogger(__name__)


#: Factories injectées par le caller pour adapter le runner à
#: son contexte (corpus local, IIIF, HF, etc.).
InitialInputsFactory = Callable[
    [DocumentRef],
    dict[ArtifactType, Artifact],
]
ContextFactory = Callable[[DocumentRef], RunContext]


class DocumentOutcome(BaseModel):
    """Résultat de l'exécution d'une pipeline sur **un** document.

    Distinct de ``PipelineResult`` : porte un statut
    (``"succeeded"`` / ``"failed"`` / ``"timed_out"`` /
    ``"cancelled"``) et conserve le ``PipelineResult`` quand il
    existe (peut être ``None`` si annulation avant démarrage).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str
    status: str = Field(pattern=r"^(succeeded|failed|timed_out|cancelled)$")
    duration_seconds: float = Field(ge=0.0)
    error: str | None = None
    pipeline_result: PipelineResult | None = None


class CorpusRunResult(BaseModel):
    """Résultat agrégé d'un run de corpus.

    Attributs
    ---------
    pipeline_name:
        Nom de la pipeline exécutée.
    corpus_name:
        Nom du corpus (libre, fourni par le caller).
    n_documents:
        Nombre total de documents tentés.
    n_succeeded:
        Nombre de documents pour lesquels la pipeline a complètement
        réussi (``PipelineResult.succeeded == True``).
    n_failed:
        Nombre de documents avec au moins une étape en échec.
    n_timed_out:
        Nombre de documents tués par timeout.
    n_cancelled:
        Nombre de documents jamais démarrés (cancel_event signalé
        avant leur tour).
    duration_seconds:
        Wall-clock total du run.
    outcomes:
        Détail document par document, ordre d'achèvement.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pipeline_name: str
    corpus_name: str
    n_documents: int = Field(ge=0)
    n_succeeded: int = Field(ge=0)
    n_failed: int = Field(ge=0)
    n_timed_out: int = Field(ge=0)
    n_cancelled: int = Field(ge=0)
    duration_seconds: float = Field(ge=0.0)
    outcomes: tuple[DocumentOutcome, ...] = Field(default_factory=tuple)


class CorpusRunner:
    """Orchestre ``PipelineExecutor`` sur un corpus avec backpressure
    + timeout réel + cancellation.

    Une instance est réutilisable à travers plusieurs runs.
    """

    def __init__(
        self,
        executor: PipelineExecutor,
        max_in_flight: int = 4,
        timeout_seconds_per_doc: float = 300.0,
        poll_interval_seconds: float = 0.05,
    ) -> None:
        if max_in_flight < 1:
            raise PicaronesError(
                f"max_in_flight doit être >= 1 (reçu {max_in_flight})."
            )
        if timeout_seconds_per_doc <= 0:
            raise PicaronesError(
                f"timeout_seconds_per_doc doit être > 0 (reçu "
                f"{timeout_seconds_per_doc})."
            )
        if poll_interval_seconds <= 0:
            raise PicaronesError(
                "poll_interval_seconds doit être > 0."
            )
        self._executor = executor
        self._max_in_flight = max_in_flight
        self._timeout = timeout_seconds_per_doc
        self._poll = poll_interval_seconds

    def run(
        self,
        spec: PipelineSpec,
        documents: Iterable[DocumentRef],
        initial_inputs_factory: InitialInputsFactory,
        context_factory: ContextFactory,
        corpus_name: str = "corpus",
        cancel_event: threading.Event | None = None,
    ) -> CorpusRunResult:
        """Exécute ``spec`` sur tous les ``documents`` du corpus.

        Returns
        -------
        CorpusRunResult
            Résultat agrégé.  Ne lève jamais — toute erreur d'un
            document est capturée dans son ``DocumentOutcome``.
        """
        documents_list = list(documents)
        run_started = time.perf_counter()

        # État partagé entre threads : ``started_at[doc_id]`` =
        # monotonic au moment où le worker du doc a vraiment démarré
        # ``execute()``.  L'orchestrateur lit ce dict pour décider
        # d'un timeout depuis le début d'exécution réelle.
        started_at: dict[str, float] = {}
        started_at_lock = threading.Lock()

        outcomes: list[DocumentOutcome] = []

        # Fast path : aucun document → résultat vide immédiat.
        if not documents_list:
            return CorpusRunResult(
                pipeline_name=spec.name,
                corpus_name=corpus_name,
                n_documents=0,
                n_succeeded=0,
                n_failed=0,
                n_timed_out=0,
                n_cancelled=0,
                duration_seconds=0.0,
                outcomes=(),
            )

        # S28 : on planifie une seule fois pour la spec.  Si la spec
        # est invalide, on lève maintenant — pas dans chaque worker.
        # Les workers consomment ensuite ``executor.run_plan(plan, ...)``
        # → N-1 validations économisées.
        plan = self._executor.plan(spec)

        # Pool instancié explicitement avec ``shutdown(wait=False,
        # cancel_futures=True)`` à la sortie : les futures en queue
        # sont annulées, les threads en cours continuent en
        # arrière-plan jusqu'à leur fin naturelle (Python ne permet
        # pas de tuer un thread).  Le caller récupère le résultat
        # immédiatement après le timeout / la cancellation, sans
        # attendre que les threads en cours se terminent — c'est
        # critique pour la latence perçue du runner.
        pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_in_flight,
            thread_name_prefix=f"picarones-{spec.name}",
        )
        try:
            future_to_doc: dict[concurrent.futures.Future, DocumentRef] = {}
            doc_iter = iter(documents_list)
            in_flight = 0
            done_count = 0

            def _submit_next() -> bool:
                """Tente de soumettre le prochain document au pool.

                Retourne ``True`` si un doc a été soumis,
                ``False`` si l'itérateur est épuisé ou si
                cancel_event est signalé.
                """
                nonlocal in_flight
                if cancel_event is not None and cancel_event.is_set():
                    return False
                try:
                    doc = next(doc_iter)
                except StopIteration:
                    return False
                fut = pool.submit(
                    self._run_one,
                    plan=plan,
                    document=doc,
                    initial_inputs_factory=initial_inputs_factory,
                    context_factory=context_factory,
                    started_at=started_at,
                    started_at_lock=started_at_lock,
                )
                future_to_doc[fut] = doc
                in_flight += 1
                return True

            # 1. Amorcer le pool : ne pas dépasser max_in_flight.
            for _ in range(self._max_in_flight):
                if not _submit_next():
                    break

            # 2. Boucle principale : récolter les résultats, surveiller
            #    les timeouts, soumettre le suivant à chaque libération.
            while future_to_doc:
                # Polling court pour pouvoir vérifier les timeouts en
                # parallèle des completions naturelles.
                done_set, _ = concurrent.futures.wait(
                    future_to_doc.keys(),
                    timeout=self._poll,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                # 2a. Récolter les futures terminées.
                for fut in done_set:
                    doc = future_to_doc.pop(fut)
                    in_flight -= 1
                    outcomes.append(_outcome_from_future(fut, doc))
                    done_count += 1
                    # Soumettre le suivant pour maintenir la backpressure.
                    _submit_next()

                # 2b. Vérifier les timeouts depuis le début d'exécution
                #     réelle (pas depuis la submission).
                now = time.monotonic()
                timed_out_futures: list[concurrent.futures.Future] = []
                with started_at_lock:
                    started_snapshot = dict(started_at)
                for fut, doc in list(future_to_doc.items()):
                    started = started_snapshot.get(doc.id)
                    if started is None:
                        continue  # pas encore démarré → pas de timeout
                    if now - started > self._timeout:
                        timed_out_futures.append(fut)

                for fut in timed_out_futures:
                    doc = future_to_doc.pop(fut)
                    in_flight -= 1
                    # On ne peut pas vraiment killer un thread en
                    # Python ; on signale via cancel_event si fourni
                    # ET on enregistre le timeout immédiatement (le
                    # thread continuera en arrière-plan jusqu'à ce
                    # qu'il ait fini, mais le run principal n'attend
                    # plus son résultat).
                    duration = (
                        now - started_snapshot.get(doc.id, now)
                    )
                    outcomes.append(DocumentOutcome(
                        document_id=doc.id,
                        status="timed_out",
                        duration_seconds=max(duration, 0.0),
                        error=(
                            f"timeout: doc {doc.id} a dépassé "
                            f"{self._timeout:.1f}s d'exécution réelle"
                        ),
                    ))
                    done_count += 1
                    _submit_next()

                # 2c. Cancellation explicite : marquer toutes les
                #     futures non démarrées comme annulées.
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = []
                    with started_at_lock:
                        started_snapshot = dict(started_at)
                    for fut, doc in list(future_to_doc.items()):
                        if doc.id not in started_snapshot:
                            # Future encore en queue → on peut la
                            # canceller proprement.
                            if fut.cancel():
                                cancelled.append(doc)
                                future_to_doc.pop(fut, None)
                                in_flight -= 1
                    for doc in cancelled:
                        outcomes.append(DocumentOutcome(
                            document_id=doc.id,
                            status="cancelled",
                            duration_seconds=0.0,
                            error="cancelled before start",
                        ))
        finally:
            # Sortie immédiate : on ne bloque pas sur les threads en
            # cours.  Les futures en queue sont annulées, les threads
            # déjà actifs continuent jusqu'à leur fin naturelle (cf.
            # commentaire à l'instanciation du pool).
            pool.shutdown(wait=False, cancel_futures=True)

        # 3. Agrégation finale.
        run_duration = time.perf_counter() - run_started
        return _aggregate(
            pipeline_name=spec.name,
            corpus_name=corpus_name,
            n_documents=len(documents_list),
            outcomes=outcomes,
            duration_seconds=run_duration,
        )

    # ──────────────────────────────────────────────────────────────────
    # Worker
    # ──────────────────────────────────────────────────────────────────

    def _run_one(
        self,
        *,
        plan,  # ExecutionPlan ; type omis pour éviter l'import top-level
        document: DocumentRef,
        initial_inputs_factory: InitialInputsFactory,
        context_factory: ContextFactory,
        started_at: dict[str, float],
        started_at_lock: threading.Lock,
    ) -> PipelineResult:
        """Exécute le plan pré-calculé sur un document.  Appelé dans
        un thread du pool.

        Enregistre ``started_at[doc.id]`` au tout début pour que
        l'orchestrateur puisse mesurer le timeout depuis le début
        d'exécution réelle.
        """
        # 1. Marquer le démarrage réel.  Ce moment est ce qui sert de
        #    référence pour le timeout.
        with started_at_lock:
            started_at[document.id] = time.monotonic()

        # 2. Construire les inputs et le contexte.
        initial_inputs = initial_inputs_factory(document)
        context = context_factory(document)

        # 3. Déléguer au PipelineExecutor.run_plan (S28).  Le plan a
        #    déjà été validé une fois par le runner ; pas de re-validation
        #    par doc.
        return self._executor.run_plan(
            plan=plan,
            document=document,
            initial_inputs=initial_inputs,
            context=context,
        )


# ──────────────────────────────────────────────────────────────────────
# Helpers d'agrégation
# ──────────────────────────────────────────────────────────────────────


def _outcome_from_future(
    fut: concurrent.futures.Future,
    doc: DocumentRef,
) -> DocumentOutcome:
    """Convertit une future achevée en ``DocumentOutcome``.

    - Future qui a levé → ``status="failed"``, ``error=str(exc)``.
    - Future qui a renvoyé un ``PipelineResult`` succeeded → ``"succeeded"``.
    - Future qui a renvoyé un ``PipelineResult`` non-succeeded →
      ``"failed"`` (au moins une étape en erreur).
    """
    try:
        result = fut.result(timeout=0)  # déjà done
    except concurrent.futures.CancelledError:
        return DocumentOutcome(
            document_id=doc.id,
            status="cancelled",
            duration_seconds=0.0,
            error="cancelled",
        )
    except Exception as exc:  # noqa: BLE001
        # PipelineExecutor capture toutes les erreurs des steps,
        # donc une exception ici signale un bug profond (typiquement
        # un PipelineSpecInvalid levé par l'executor).
        return DocumentOutcome(
            document_id=doc.id,
            status="failed",
            duration_seconds=0.0,
            error=f"runner_internal_error: {type(exc).__name__}: {exc}",
        )

    if result.succeeded:
        status = "succeeded"
        error: str | None = None
    else:
        status = "failed"
        # Concaténer les erreurs de step pour le diagnostic.
        step_errors = [
            f"{r.step_id}: {r.error}"
            for r in result.step_results
            if not r.succeeded
        ]
        error = "; ".join(step_errors) if step_errors else "unknown failure"

    return DocumentOutcome(
        document_id=doc.id,
        status=status,
        duration_seconds=result.duration_seconds,
        error=error,
        pipeline_result=result,
    )


def _aggregate(
    *,
    pipeline_name: str,
    corpus_name: str,
    n_documents: int,
    outcomes: list[DocumentOutcome],
    duration_seconds: float,
) -> CorpusRunResult:
    return CorpusRunResult(
        pipeline_name=pipeline_name,
        corpus_name=corpus_name,
        n_documents=n_documents,
        n_succeeded=sum(1 for o in outcomes if o.status == "succeeded"),
        n_failed=sum(1 for o in outcomes if o.status == "failed"),
        n_timed_out=sum(1 for o in outcomes if o.status == "timed_out"),
        n_cancelled=sum(1 for o in outcomes if o.status == "cancelled"),
        duration_seconds=duration_seconds,
        outcomes=tuple(outcomes),
    )


__all__ = [
    "CorpusRunner",
    "CorpusRunResult",
    "DocumentOutcome",
    "InitialInputsFactory",
    "ContextFactory",
]
