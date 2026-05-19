"""Sprint A14-S8 — annulation propre du ``CorpusRunner``.

Vérifie qu'un ``threading.Event`` partagé permet au caller
(typiquement un endpoint FastAPI ``cancel``) de signaler l'arrêt.
Les futures non démarrées sont annulées proprement, les futures
en cours se terminent (Python ne permet pas de tuer un thread).
"""

from __future__ import annotations

import threading
import time

from picarones.domain import Artifact, ArtifactType, DocumentRef
from picarones.pipeline import (
    CorpusRunner,
    PipelineExecutor,
    PipelineSpec,
    PipelineStep,
    RunContext,
)


class _EventAwareAdapter:
    """Adapter qui dort par petites tranches et signale qu'il a démarré."""

    name = "event"
    input_types = frozenset({ArtifactType.IMAGE})
    output_types = frozenset({ArtifactType.RAW_TEXT})
    execution_mode = "io"

    def __init__(
        self,
        sleep_seconds: float,
        started_event: threading.Event | None = None,
    ) -> None:
        self._sleep = sleep_seconds
        self._started = started_event

    def execute(self, inputs, params, context):
        if self._started is not None:
            self._started.set()
        time.sleep(self._sleep)
        return {
            ArtifactType.RAW_TEXT: Artifact(
                id=f"{context.document_id}:raw_text",
                document_id=context.document_id,
                type=ArtifactType.RAW_TEXT,
            ),
        }


def _build(adapter, max_in_flight: int = 1):
    registry = {"event": adapter}
    exe = PipelineExecutor(adapter_resolver=lambda n: registry[n])
    runner = CorpusRunner(
        exe,
        max_in_flight=max_in_flight,
        timeout_seconds_per_doc=10.0,
        poll_interval_seconds=0.01,
    )
    spec = PipelineSpec(
        name="c", initial_inputs=(ArtifactType.IMAGE,),
        steps=(PipelineStep(
            id="s", kind="ocr", adapter_name="event",
            input_types=(ArtifactType.IMAGE,),
            output_types=(ArtifactType.RAW_TEXT,),
        ),),
    )
    return runner, spec


def _factories():
    def inputs(doc):
        return {ArtifactType.IMAGE: Artifact(
            id=f"{doc.id}:image",
            document_id=doc.id,
            type=ArtifactType.IMAGE,
        )}

    def ctx(doc):
        return RunContext(
            document_id=doc.id, code_version="1.0.0", pipeline_name="c",
        )
    return inputs, ctx


def test_cancel_before_run_yields_zero_progress() -> None:
    """Cancel signalé avant le run → aucun doc ne démarre."""
    adapter = _EventAwareAdapter(sleep_seconds=1.0)
    runner, spec = _build(adapter, max_in_flight=1)
    inputs, ctx = _factories()
    docs = [DocumentRef(id=f"d{i}") for i in range(10)]

    cancel_event = threading.Event()
    cancel_event.set()  # déjà signalé

    result = runner.run(
        spec, docs, inputs, ctx, cancel_event=cancel_event,
    )
    # Tous les docs sont cancelled (ou en partie cancelled si
    # quelques-uns ont eu le temps d'être amorcés avant la
    # première itération de la boucle).
    assert result.n_succeeded == 0


def test_cancel_during_run_stops_pending_docs() -> None:
    """Cancel signalé pendant l'exécution → les docs en attente sont
    annulés, ceux en cours se terminent."""
    started = threading.Event()
    adapter = _EventAwareAdapter(sleep_seconds=0.1, started_event=started)
    runner, spec = _build(adapter, max_in_flight=1)
    inputs, ctx = _factories()
    docs = [DocumentRef(id=f"d{i}") for i in range(20)]

    cancel_event = threading.Event()

    def _trigger_cancel():
        # Attendre que le premier doc démarre, puis annuler.
        started.wait(timeout=2.0)
        cancel_event.set()

    canceller = threading.Thread(target=_trigger_cancel, daemon=True)
    canceller.start()

    t0 = time.perf_counter()
    result = runner.run(
        spec, docs, inputs, ctx, cancel_event=cancel_event,
    )
    elapsed = time.perf_counter() - t0

    canceller.join(timeout=1.0)

    # On a au plus quelques docs réussis (ceux qui ont démarré avant
    # la cancellation), et le reste cancellé.  Pas tous succeeded.
    assert result.n_succeeded < len(docs)
    # Le run ne dure pas 20 * 0.1 = 2s ; il s'arrête bien plus tôt
    # grâce à la cancellation.
    assert elapsed < 1.5, f"cancellation trop lente : {elapsed:.2f}s"


def test_cancel_returns_well_formed_result() -> None:
    """Même en cas de cancel, le ``CorpusRunResult`` reste cohérent
    (n_succeeded + n_failed + n_timed_out + n_cancelled <=
    n_documents, outcomes correspondants)."""
    adapter = _EventAwareAdapter(sleep_seconds=0.5)
    runner, spec = _build(adapter, max_in_flight=2)
    inputs, ctx = _factories()
    docs = [DocumentRef(id=f"d{i}") for i in range(10)]

    cancel_event = threading.Event()
    cancel_event.set()

    result = runner.run(
        spec, docs, inputs, ctx, cancel_event=cancel_event,
    )
    total = (
        result.n_succeeded + result.n_failed
        + result.n_timed_out + result.n_cancelled
    )
    assert total <= result.n_documents
    assert len(result.outcomes) == total
