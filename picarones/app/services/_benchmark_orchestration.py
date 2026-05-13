"""Orchestration interne du benchmark : unified vs with_partial.

Module extrait du god-module ``benchmark_runner.py`` lors de la
Phase 6 (round 6) de l'audit code-quality (2026-05).

Surface publique (rééxportée par ``benchmark_runner.py`` avec
préfixe ``_`` pour préserver l'API privée historique) :

- :func:`run_benchmark_unified` — chemin rapide sans persistance
  intermédiaire (un seul ``BenchmarkService.run`` multi-engine).
- :func:`run_benchmark_with_partial` — chemin reprise per-engine
  avec NDJSON intermédiaire.  Si le run crashe ou est annulé,
  les engines déjà traités sont conservés ; la reprise charge
  les partials et ne re-calcule que les docs manquants.

La distinction entre les deux est gouvernée par l'argument
``partial_dir`` de ``run_benchmark_via_service`` :

- ``None`` → ``run_benchmark_unified`` (workflow demo, CI, smoke).
- ``Path(...)`` → ``run_benchmark_with_partial`` (workflow long,
  prod, benchmark institutionnel).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from picarones.app.services._benchmark_adapter_resolver import (
    build_adapter_resolver,
    engine_to_pipeline_spec,
)
from picarones.app.services._benchmark_conversions import (
    corpus_to_corpus_spec,
)
from picarones.app.services._benchmark_converter import (
    run_result_to_benchmark_result,
)
from picarones.app.services._benchmark_execution import (
    execute_via_benchmark_service,
)
from picarones.app.services._benchmark_helpers import (
    _build_pipeline_info,
    _engine_config_for_fingerprint,
    _safe_engine_version,
)

if TYPE_CHECKING:
    from picarones.evaluation.corpus import Corpus

logger = logging.getLogger(__name__)


def run_benchmark_unified(
    *,
    corpus: "Corpus",
    engines: list[Any],
    char_exclude: Any | None,
    normalization_profile: Any | None,
    profile: str,
    code_version: str,
    progress_callback: Callable[[str, int, str], None] | None,
    timeout_seconds: float,
    cancel_event: Any | None,
) -> Any:
    """Chemin rapide : un seul ``BenchmarkService.run`` multi-engine.

    Pas de persistance intermédiaire — si le run crashe, tout est
    perdu.  Utilisé quand ``partial_dir`` est ``None``.
    """
    with tempfile.TemporaryDirectory(prefix="picarones_bench_") as ws:
        workspace = Path(ws)
        gt_dir = workspace / "gt"
        gt_dir.mkdir()
        run_dir = workspace / "run"
        run_dir.mkdir()

        corpus_spec = corpus_to_corpus_spec(corpus, workspace_dir=gt_dir)
        pipeline_specs = [engine_to_pipeline_spec(e) for e in engines]
        adapter_resolver = build_adapter_resolver(engines)
        pipeline_to_engine_name = {
            spec.name: engine.name
            for spec, engine in zip(pipeline_specs, engines)
        }

        run_result = execute_via_benchmark_service(
            corpus_spec=corpus_spec,
            pipeline_specs=pipeline_specs,
            adapter_resolver=adapter_resolver,
            workspace_uri=str(run_dir),
            code_version=code_version,
            timeout_seconds=timeout_seconds,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            pipeline_to_engine_name=pipeline_to_engine_name,
        )

        return run_result_to_benchmark_result(
            run_result,
            corpus=corpus,
            engines=engines,
            char_exclude=char_exclude,
            normalization_profile=normalization_profile,
            profile=profile,
        )


def run_benchmark_with_partial(
    *,
    corpus: "Corpus",
    engines: list[Any],
    partial_dir: Path,
    char_exclude: Any | None,
    normalization_profile: Any | None,
    profile: str,
    code_version: str,
    progress_callback: Callable[[str, int, str], None] | None,
    timeout_seconds: float,
    cancel_event: Any | None,
) -> Any:
    """Chemin reprise : per-engine avec NDJSON intermédiaire.

    Pour chaque engine, charge le partial existant, filtre les docs
    déjà traités, lance ``BenchmarkService`` sur les restants,
    persiste chaque nouveau ``DocumentResult`` au fil de l'eau.
    """
    from picarones.app.services.partial_store import (
        _delete_partial,
        _load_partial,
        _save_partial_line,
        partial_path_for_engine,
    )
    from picarones.evaluation.benchmark_result import (
        BenchmarkResult,
        EngineReport,
    )
    from picarones.evaluation.corpus import Corpus as LegacyCorpus
    from picarones.evaluation.metric_hooks import run_corpus_aggregators
    # Force l'auto-enregistrement des hooks builtin (décorateurs).
    import picarones.evaluation.metrics.builtin_hooks  # noqa: F401
    from picarones.evaluation.metric_result import aggregate_metrics

    partial_dir.mkdir(parents=True, exist_ok=True)

    # Index des docs par ID — permet de ré-ordonner les
    # DocumentResult rechargés selon l'ordre original du corpus.
    doc_order = {doc.doc_id: idx for idx, doc in enumerate(corpus.documents)}

    engine_reports: list[Any] = []

    for engine in engines:
        # Vérifier la cancellation entre engines.
        if cancel_event is not None and getattr(
            cancel_event, "is_set", lambda: False,
        )():
            logger.info(
                "[partial_dir] benchmark annulé avant l'engine '%s' "
                "— partials conservés pour reprise.", engine.name,
            )
            break

        # Phase 2.3 — fingerprint inclut config moteur + profil
        # normalisation + char_exclude + corpus files (mtime/size) +
        # version code.  Deux runs avec configs différentes →
        # fichiers partiels distincts → pas de réutilisation
        # silencieuse de résultats incompatibles.
        partial_path = partial_path_for_engine(
            corpus=corpus,
            engine=engine,
            partial_dir=partial_dir,
            engine_config=_engine_config_for_fingerprint(engine),
            normalization_profile=normalization_profile,
            char_exclude=char_exclude,
            profile=profile,
            code_version=code_version,
        )
        loaded_results = _load_partial(partial_path)
        loaded_doc_ids = {dr.doc_id for dr in loaded_results}

        if loaded_results:
            logger.info(
                "[partial_dir] reprise '%s' : %d/%d docs déjà traités.",
                engine.name, len(loaded_results), len(corpus.documents),
            )

        remaining_docs = [
            d for d in corpus.documents if d.doc_id not in loaded_doc_ids
        ]

        new_doc_results: list[Any] = []
        if remaining_docs:
            # Sub-corpus avec uniquement les docs restants.  On
            # conserve le ``name`` original pour que les chemins de
            # partial restent cohérents si un re-run arrive.
            sub_corpus = LegacyCorpus(
                name=corpus.name,
                documents=remaining_docs,
                source_path=corpus.source_path,
            )

            with tempfile.TemporaryDirectory(
                prefix="picarones_bench_partial_",
            ) as ws:
                workspace = Path(ws)
                gt_dir = workspace / "gt"
                gt_dir.mkdir()
                run_dir = workspace / "run"
                run_dir.mkdir()

                sub_corpus_spec = corpus_to_corpus_spec(
                    sub_corpus, workspace_dir=gt_dir,
                )
                pipeline_spec = engine_to_pipeline_spec(engine)
                adapter_resolver = build_adapter_resolver([engine])
                pipeline_to_engine_name = {pipeline_spec.name: engine.name}

                run_result = execute_via_benchmark_service(
                    corpus_spec=sub_corpus_spec,
                    pipeline_specs=[pipeline_spec],
                    adapter_resolver=adapter_resolver,
                    workspace_uri=str(run_dir),
                    code_version=code_version,
                    timeout_seconds=timeout_seconds,
                    progress_callback=progress_callback,
                    cancel_event=cancel_event,
                    pipeline_to_engine_name=pipeline_to_engine_name,
                )

                # Convertir ce sous-RunResult en EngineReport avec
                # uniquement les docs restants — puis extraire les
                # ``DocumentResult`` pour append au partial.
                sub_report = run_result_to_benchmark_result(
                    run_result,
                    corpus=sub_corpus,
                    engines=[engine],
                    char_exclude=char_exclude,
                    normalization_profile=normalization_profile,
                    profile=profile,
                )
                new_doc_results = list(
                    sub_report.engine_reports[0].document_results,
                )

                # Append au partial : un cancel mid-engine préserve
                # ce qui a déjà été calculé.
                for dr in new_doc_results:
                    _save_partial_line(partial_path, dr)

        # Fusion : loaded + new, ré-ordonné selon le corpus original.
        all_doc_results = list(loaded_results) + new_doc_results
        all_doc_results.sort(key=lambda dr: doc_order.get(dr.doc_id, 0))

        aggregated = aggregate_metrics([d.metrics for d in all_doc_results])
        pipeline_info = _build_pipeline_info(engine)
        agg_values = run_corpus_aggregators(profile, all_doc_results)

        engine_reports.append(
            EngineReport(
                engine_name=engine.name,
                engine_version=_safe_engine_version(engine),
                engine_config=getattr(engine, "config", {}) or {},
                document_results=all_doc_results,
                aggregated_metrics=aggregated,
                pipeline_info=pipeline_info,
                **agg_values,
            ),
        )

        # Engine traité avec succès → cleanup du partial.  Si on
        # arrive ici sans exception, tous les docs sont dans
        # ``all_doc_results``.
        _delete_partial(partial_path)

    # Phase 3.2 audit code-quality — consume_fallback_log idempotent.
    from picarones.adapters.corpus._fallback_log import consume_fallback_log
    fallbacks = consume_fallback_log()
    metadata: dict[str, Any] = {}
    if fallbacks:
        metadata["importer_fallbacks"] = fallbacks

    return BenchmarkResult(
        corpus_name=corpus.name,
        corpus_source=str(corpus.source_path) if corpus.source_path else None,
        document_count=len(corpus.documents),
        engine_reports=engine_reports,
        metadata=metadata,
    )


__all__ = ["run_benchmark_unified", "run_benchmark_with_partial"]
