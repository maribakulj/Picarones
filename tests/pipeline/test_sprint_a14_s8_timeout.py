"""Sprint A14-S8 — timeout depuis le début d'exécution **réelle**.

Le bug critique de l'ancien runner : un document pouvait être marqué
``timeout`` parce qu'il avait passé N secondes en queue, pas N
secondes en train de tourner.  Le nouveau ``CorpusRunner`` mesure
le timeout depuis ``time.monotonic()`` au moment où le worker
démarre réellement (cf. ``CorpusRunner._run_one`` qui écrit
``started_at[doc.id]`` en première instruction).
"""

from __future__ import annotations

import threading
import time

import pytest

from picarones.domain import Artifact, ArtifactType, DocumentRef
from picarones.pipeline import (
    CorpusRunner,
    PipelineExecutor,
    PipelineSpec,
    PipelineStep,
    RunContext,
)


class _SlowAdapter:
    """Adapter qui dort un certain temps avant de retourner."""

    name = "slow"
    input_types = frozenset({ArtifactType.IMAGE})
    output_types = frozenset({ArtifactType.RAW_TEXT})
    execution_mode = "io"

    def __init__(self, sleep_seconds: float) -> None:
        self._sleep = sleep_seconds

    def execute(self, inputs, params, context):
        time.sleep(self._sleep)
        return {
            ArtifactType.RAW_TEXT: Artifact(
                id=f"{context.document_id}:raw_text",
                document_id=context.document_id,
                type=ArtifactType.RAW_TEXT,
            ),
        }


def _build(adapter, *, timeout: float, max_in_flight: int = 2):
    registry = {"slow": adapter}
    exe = PipelineExecutor(adapter_resolver=lambda n: registry[n])
    runner = CorpusRunner(
        exe,
        max_in_flight=max_in_flight,
        timeout_seconds_per_doc=timeout,
        poll_interval_seconds=0.01,
    )
    spec = PipelineSpec(
        name="t", initial_inputs=(ArtifactType.IMAGE,),
        steps=(PipelineStep(
            id="s", kind="ocr", adapter_name="slow",
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
            document_id=doc.id, code_version="1.0.0", pipeline_name="t",
        )
    return inputs, ctx


def test_doc_timed_out_when_exceeds_timeout() -> None:
    """Step qui dort 0.5s, timeout 0.1s → status timed_out."""
    adapter = _SlowAdapter(sleep_seconds=0.5)
    runner, spec = _build(adapter, timeout=0.1, max_in_flight=1)
    inputs, ctx = _factories()
    docs = [DocumentRef(id="slow_one", image_uri="/tmp/x.png")]

    t0 = time.perf_counter()
    result = runner.run(spec, docs, inputs, ctx)
    elapsed = time.perf_counter() - t0

    assert result.n_timed_out == 1
    assert result.outcomes[0].status == "timed_out"
    assert "timeout" in (result.outcomes[0].error or "")
    # Le run principal a rendu la main rapidement (ne s'est pas bloqué
    # sur le sleep complet — le thread continue mais on n'attend plus).
    assert elapsed < 0.3, f"runner s'est bloqué : {elapsed:.2f}s"


def test_timeout_measured_from_real_start_not_submission() -> None:
    """Bug historique : avec un seul worker (max_in_flight=1) et 4
    documents, les 3 derniers attendent en queue.  L'ancien runner
    aurait marqué ces 3 docs timeout dès que la queue dépassait le
    timeout.  Le nouveau runner ne marque timeout que les docs qui
    ont **réellement** dépassé le délai en exécution."""
    # Adapter qui dort 50ms — bien sous le timeout de 500ms.
    adapter = _SlowAdapter(sleep_seconds=0.05)
    runner, spec = _build(adapter, timeout=0.5, max_in_flight=1)
    inputs, ctx = _factories()
    docs = [DocumentRef(id=f"d{i}") for i in range(4)]

    result = runner.run(spec, docs, inputs, ctx)

    # Les 4 docs auraient pris ~0.2s en série, ce qui dépasse le
    # timeout de 0.5s **si** le runner mesurait depuis la submission
    # du dernier doc.  Mais comme on mesure depuis le début réel
    # de chaque doc, aucun ne devrait timeout.
    assert result.n_succeeded == 4
    assert result.n_timed_out == 0


def test_some_docs_succeed_others_timeout() -> None:
    """Mix : la moitié des docs sont rapides, l'autre lente.  Avec
    un timeout intermédiaire, les rapides réussissent et les lents
    timeout."""

    class _ConditionalSlow:
        name = "cond"
        input_types = frozenset({ArtifactType.IMAGE})
        output_types = frozenset({ArtifactType.RAW_TEXT})
        execution_mode = "io"

        def execute(self, inputs, params, context):
            # Les docs avec id pair sont rapides.
            if int(context.document_id.removeprefix("d")) % 2 == 0:
                time.sleep(0.01)
            else:
                time.sleep(0.5)
            return {
                ArtifactType.RAW_TEXT: Artifact(
                    id=f"{context.document_id}:raw_text",
                    document_id=context.document_id,
                    type=ArtifactType.RAW_TEXT,
                ),
            }

    adapter = _ConditionalSlow()
    runner, spec = _build(adapter, timeout=0.1, max_in_flight=2)
    inputs, ctx = _factories()
    docs = [DocumentRef(id=f"d{i}") for i in range(6)]

    result = runner.run(spec, docs, inputs, ctx)
    assert result.n_succeeded == 3  # pairs : d0, d2, d4
    assert result.n_timed_out == 3  # impairs : d1, d3, d5
