"""Sprint A14-S8 — backpressure du ``CorpusRunner``.

Vérifie que ``max_in_flight`` est respecté à tout instant : il n'y
a jamais plus de N adapters qui tournent en parallèle, même sur
des corpus de plusieurs centaines de documents.

Stratégie : un stub d'adapter incrémente un compteur partagé au
début de ``execute()``, le décrémente à la fin, et capture le
maximum atteint.  À la fin du run, on vérifie ``max_observed
<= max_in_flight``.
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


class _ConcurrencyTrackingAdapter:
    """Adapter qui mesure la concurrence observée pendant son exécution."""

    name = "tracking"
    input_types = frozenset({ArtifactType.IMAGE})
    output_types = frozenset({ArtifactType.RAW_TEXT})
    execution_mode = "io"

    def __init__(self, sleep_seconds: float = 0.01) -> None:
        self._sleep = sleep_seconds
        self._lock = threading.Lock()
        self._current = 0
        self.max_observed = 0

    def execute(self, inputs, params, context):
        with self._lock:
            self._current += 1
            if self._current > self.max_observed:
                self.max_observed = self._current
        try:
            time.sleep(self._sleep)
            return {
                ArtifactType.RAW_TEXT: Artifact(
                    id=f"{context.document_id}:raw_text",
                    document_id=context.document_id,
                    type=ArtifactType.RAW_TEXT,
                ),
            }
        finally:
            with self._lock:
                self._current -= 1


def _build(adapter, max_in_flight: int):
    registry = {"tracking": adapter}
    exe = PipelineExecutor(adapter_resolver=lambda n: registry[n])
    runner = CorpusRunner(
        exe,
        max_in_flight=max_in_flight,
        timeout_seconds_per_doc=10.0,
        poll_interval_seconds=0.005,
    )
    spec = PipelineSpec(
        name="bp", initial_inputs=(ArtifactType.IMAGE,),
        steps=(PipelineStep(
            id="s", kind="ocr", adapter_name="tracking",
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
            uri=doc.image_uri,
        )}

    def ctx(doc):
        return RunContext(
            document_id=doc.id,
            code_version="1.0.0",
            pipeline_name="bp",
        )
    return inputs, ctx


@pytest.mark.parametrize("max_in_flight", [1, 2, 4])
def test_max_in_flight_respected(max_in_flight: int) -> None:
    adapter = _ConcurrencyTrackingAdapter(sleep_seconds=0.02)
    runner, spec = _build(adapter, max_in_flight=max_in_flight)
    inputs, ctx = _factories()
    docs = [DocumentRef(id=f"d{i}", image_uri=f"/tmp/{i}.png") for i in range(40)]

    result = runner.run(spec, docs, inputs, ctx, corpus_name="bp")

    assert result.n_documents == 40
    assert result.n_succeeded == 40
    # Garantie de backpressure : la concurrence n'a jamais excédé max.
    assert adapter.max_observed <= max_in_flight, (
        f"max observed = {adapter.max_observed}, attendu <= {max_in_flight}"
    )
    # Et la backpressure a effectivement saturé : on a bien atteint le
    # plafond (preuve qu'on parallélise vraiment).
    assert adapter.max_observed == max_in_flight, (
        f"on aurait dû saturer à {max_in_flight}, observed "
        f"{adapter.max_observed}"
    )


def test_max_in_flight_one_means_sequential() -> None:
    adapter = _ConcurrencyTrackingAdapter(sleep_seconds=0.005)
    runner, spec = _build(adapter, max_in_flight=1)
    inputs, ctx = _factories()
    docs = [DocumentRef(id=f"d{i}") for i in range(20)]

    runner.run(spec, docs, inputs, ctx)
    assert adapter.max_observed == 1


def test_empty_corpus_returns_zero_outcomes() -> None:
    adapter = _ConcurrencyTrackingAdapter()
    runner, spec = _build(adapter, max_in_flight=4)
    inputs, ctx = _factories()

    result = runner.run(spec, [], inputs, ctx)
    assert result.n_documents == 0
    assert result.outcomes == ()
    assert adapter.max_observed == 0


def test_max_in_flight_zero_rejected() -> None:
    from picarones.domain import PicaronesError
    exe = PipelineExecutor(adapter_resolver=lambda n: None)
    with pytest.raises(PicaronesError, match="max_in_flight"):
        CorpusRunner(exe, max_in_flight=0)


def test_negative_timeout_rejected() -> None:
    from picarones.domain import PicaronesError
    exe = PipelineExecutor(adapter_resolver=lambda n: None)
    with pytest.raises(PicaronesError, match="timeout"):
        CorpusRunner(exe, timeout_seconds_per_doc=0)
