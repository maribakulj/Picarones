"""Orchestration ``BenchmarkService`` — module extrait du god-module
``benchmark_runner.py`` lors de la Phase 6 (round 4) de l'audit
code-quality (2026-05).

Surface publique (rééxportée par ``benchmark_runner.py`` pour
préserver les imports internes existants) :

- :func:`execute_via_benchmark_service` — lance
  ``BenchmarkService.run`` sur les specs converties.  Wrappe la
  factory d'inputs + GT + RunContext + cancel_event.

Les fonctions ``_run_benchmark_unified`` et
``_run_benchmark_with_partial`` (qui consomment le ``BenchmarkResult``
final) restent dans ``benchmark_runner.py`` car elles dépendent
d'un grand nombre d'helpers internes (NER attach, fingerprint,
partial store, etc.).  Leur extraction nécessiterait d'extraire
aussi tous ces helpers — chantier reporté.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, Callable

from picarones.domain.artifacts import ArtifactType
from picarones.domain.corpus import CorpusSpec
from picarones.domain.documents import DocumentRef
from picarones.domain.errors import PicaronesError
from picarones.domain.pipeline_spec import PipelineSpec

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def execute_via_benchmark_service(
    *,
    corpus_spec: CorpusSpec,
    pipeline_specs: list[PipelineSpec],
    adapter_resolver: Callable[[str], Any],
    workspace_uri: str,
    code_version: str,
    timeout_seconds: float,
    progress_callback: Callable[[str, int, str], None] | None = None,
    cancel_event: Any | None = None,
    pipeline_to_engine_name: dict[str, str] | None = None,
) -> Any:
    """Lance ``BenchmarkService.run`` sur les specs converties.

    Vues passées en liste vide — les métriques sont calculées
    côté converter via ``compute_metrics`` directement sur les
    hypothèses extraites des artefacts.  Pattern simple, cohérent :
    on calcule aussi les métriques au moment du benchmark
    (pas via ``EvaluationView``).
    """
    from picarones.app.services.benchmark_service import BenchmarkService
    from picarones.evaluation.projectors.registry import ProjectorRegistry
    from picarones.evaluation.registry.registry import MetricRegistry
    from picarones.evaluation.views.executor import (
        DefaultEvaluationViewExecutor,
    )
    from picarones.pipeline.executor import PipelineExecutor
    from picarones.pipeline.runner import CorpusRunner
    from picarones.pipeline.types import RunContext

    executor = PipelineExecutor(adapter_resolver=adapter_resolver)
    runner = CorpusRunner(
        executor,
        max_in_flight=2,
        timeout_seconds_per_doc=timeout_seconds,
    )

    # ViewExecutor minimal : registres vides.
    view_executor = DefaultEvaluationViewExecutor.from_registries(
        metric_registry=MetricRegistry(),
        projector_registry=ProjectorRegistry(),
        payload_loader=lambda art: None,
    )
    bench = BenchmarkService(
        corpus_runner=runner,
        view_executor=view_executor,
        code_version=code_version,
    )

    # Factory pour les inputs initiaux (toujours IMAGE depuis l'URI).
    def inputs_factory(doc: DocumentRef) -> dict[ArtifactType, Any]:
        from picarones.domain.artifacts import Artifact

        if doc.image_uri is None:
            raise PicaronesError(
                f"Document {doc.id!r} sans image_uri — la pipeline "
                "par défaut consomme une IMAGE en entrée.",
            )
        return {
            ArtifactType.IMAGE: Artifact(
                id=f"{doc.id}:image",
                document_id=doc.id,
                type=ArtifactType.IMAGE,
                uri=doc.image_uri,
            ),
        }

    # GT factory : pas utilisée car ``views=[]``.
    def gt_factory(doc: DocumentRef, art_type: ArtifactType) -> Any:
        return None

    counter_lock = threading.Lock()
    counter_state = {"doc_idx": 0}

    def context_factory(
        doc: DocumentRef, pipeline_name: str,
    ) -> RunContext:
        if progress_callback is not None:
            with counter_lock:
                idx = counter_state["doc_idx"]
                counter_state["doc_idx"] = idx + 1
            engine_name = (
                pipeline_to_engine_name.get(pipeline_name, pipeline_name)
                if pipeline_to_engine_name is not None
                else pipeline_name
            )
            try:
                progress_callback(engine_name, idx, doc.id)
            except Exception as exc:  # noqa: BLE001
                # On ignore silencieusement les erreurs du callback ;
                # un caller qui crashe ne doit pas faire tomber le
                # benchmark.  Logge en debug pour diagnostic.
                logger.debug(
                    "[benchmark_execution] progress_callback raised: %s",
                    exc,
                )
        return RunContext(
            document_id=doc.id,
            code_version=code_version,
            pipeline_name=pipeline_name,
            workspace_uri=workspace_uri,
        )

    # Propagation du cancel_event au CorpusRunner.
    if cancel_event is not None:
        original_run = runner.run

        def _runner_run_with_cancel(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("cancel_event", cancel_event)
            return original_run(*args, **kwargs)

        runner.run = _runner_run_with_cancel  # type: ignore[method-assign]

    return bench.run(
        corpus=corpus_spec,
        pipelines=pipeline_specs,
        views=[],
        ground_truth_factory=gt_factory,
        pipeline_inputs_factory=inputs_factory,
        context_factory=context_factory,
    )


__all__ = ["execute_via_benchmark_service"]
