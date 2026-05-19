"""Constructeurs stateless du ``RunOrchestrator`` (corpus / pipelines /
vues / service).

Audit prod Phase A — extraction des 4 ``@staticmethod`` (sans
``self``) hors du god-module ``run_orchestrator.py``.  Déplacement
verbatim, comportement strictement préservé : ``run_orchestrator``
réimporte ces noms (façade) et conserve un wrapper mince
``_build_pipelines`` (un test l'appelle via ``orch._build_pipelines``).
"""

from __future__ import annotations

import io
import threading
import zipfile
from pathlib import Path
from typing import Any, Callable

from picarones.app.schemas import RunSpec, resolve_adapter_class
from picarones.app.services.benchmark_service import BenchmarkService
from picarones.app.services.corpus_service import (
    CorpusImportError,
    CorpusService,
)
from picarones.app.services.path_security import WorkspaceManager
from picarones.app.services.registry_service import RegistryService
from picarones.domain.corpus import CorpusSpec
from picarones.evaluation.views import (
    DefaultEvaluationViewExecutor,
    build_alto_view,
    build_search_view,
    build_text_view,
)
from picarones.pipeline import (
    CorpusRunner,
    PipelineExecutor,
    PipelineSpec,
    PipelineStep,
)

from picarones.app.services.run_orchestrator_helpers.loaders import (
    _filesystem_payload_loader,
    _kwargs_signature,
)


def _load_corpus(
    spec: RunSpec, workspace: WorkspaceManager,
) -> tuple[CorpusSpec, Path]:
    """Charge le corpus selon ``corpus_zip`` ou ``corpus_dir``."""
    corpus_service = CorpusService(workspace)
    if spec.corpus_zip is not None:
        zip_path = Path(spec.corpus_zip)
        zip_bytes = zip_path.read_bytes()
        report = corpus_service.import_zip(
            zip_bytes,
            corpus_name=spec.corpus_name or zip_path.stem,
            metadata=spec.corpus_metadata,
        )
        return report.spec, report.extracted_dir

    # corpus_dir : on zippe à la volée le contenu du dir et on
    # délègue à ``CorpusService`` — réutilise toute la détection
    # sans dupliquer la logique de classification image / GT.
    assert spec.corpus_dir is not None  # garanti par RunSpec validator
    src_dir = Path(spec.corpus_dir)
    if not src_dir.is_dir():
        raise CorpusImportError(
            f"corpus_dir n'est pas un répertoire : {src_dir!r}.",
        )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        for file_path in src_dir.rglob("*"):
            if file_path.is_file():
                arc = file_path.relative_to(src_dir).as_posix()
                zf.write(file_path, arcname=arc)
    report = corpus_service.import_zip(
        buf.getvalue(),
        corpus_name=spec.corpus_name or src_dir.name,
        metadata=spec.corpus_metadata,
    )
    return report.spec, report.extracted_dir


def _build_pipelines(
    spec: RunSpec,
) -> tuple[
    list[PipelineSpec],
    Callable[[str], Any],
    dict[str, dict[str, Any]],
]:
    """Construit les ``PipelineSpec`` + un resolver d'adapters.

    Disambiguation des steps :

    - Deux steps avec la même ``(class, kwargs)`` partagent la
      même instance d'adapter (cache).
    - Deux steps avec la même ``id`` mais une ``class`` ou des
      ``kwargs`` différents reçoivent des ``adapter_name``
      distincts (préfixés par le nom de pipeline).

    C'est essentiel pour le cas où plusieurs pipelines utilisent
    la **même classe** avec des **kwargs différents** (ex :
    ``PrecomputedTextAdapter`` instancié N fois avec
    ``source_label`` distincts).
    """
    instance_cache: dict[str, Any] = {}
    registered: dict[str, tuple[type, str]] = {}
    name_to_class: dict[str, type] = {}
    name_to_kwargs: dict[str, dict[str, Any]] = {}

    pipeline_specs: list[PipelineSpec] = []
    for p in spec.pipelines:
        steps: list[PipelineStep] = []
        for s in p.steps:
            cls = resolve_adapter_class(s.adapter_class)
            kwargs_sig = _kwargs_signature(s.adapter_kwargs)
            adapter_name = s.id
            existing = registered.get(adapter_name)
            if existing is not None and existing != (cls, kwargs_sig):
                adapter_name = f"{p.name}__{s.id}"
            registered[adapter_name] = (cls, kwargs_sig)
            name_to_class[adapter_name] = cls
            name_to_kwargs[adapter_name] = s.adapter_kwargs
            # ``inputs_from`` du StepSpec YAML doit être propagé au
            # ``domain.PipelineSpec`` pour que le DAG branchant soit
            # honoré ; sans ce passage, un DAG branchant déclaré dans
            # le YAML serait silencieusement exécuté en linéaire.
            steps.append(PipelineStep(
                id=s.id,
                kind="step",
                adapter_name=adapter_name,
                input_types=s.input_types,
                output_types=s.output_types,
                inputs_from=dict(s.inputs_from),
            ))
        pipeline_specs.append(PipelineSpec(
            name=p.name,
            initial_inputs=p.initial_inputs,
            steps=tuple(steps),
        ))

    def resolver(name: str) -> Any:
        if name not in instance_cache:
            cls = name_to_class[name]
            kwargs = name_to_kwargs[name]
            instance_cache[name] = cls(**kwargs)
        return instance_cache[name]

    # Copie défensive — le manifest doit recevoir un snapshot
    # immuable, pas la map vivante du resolver.
    adapter_kwargs_dump = {
        name: dict(kwargs) for name, kwargs in name_to_kwargs.items()
    }
    return pipeline_specs, resolver, adapter_kwargs_dump


def _build_views(
    view_names: tuple[str, ...],
    *,
    normalization_profile: str | None = None,
    char_exclude: str | None = None,
) -> list[Any]:
    """Map noms canoniques → vues construites.

    Phase B2.5 — ``normalization_profile`` et ``char_exclude``
    sont propagés aux vues qui les supportent (``text_final`` et
    ``searchability``).  ``alto_documentary`` les ignore : ses
    métriques structurelles n'opèrent pas sur du texte.
    """
    text_view_kwargs = {
        "normalization_profile": normalization_profile,
        "char_exclude": char_exclude,
    }
    builders: dict[str, Callable[[], Any]] = {
        "text_final": lambda: build_text_view(**text_view_kwargs),
        "alto_documentary": build_alto_view,
        "searchability": lambda: build_search_view(**text_view_kwargs),
    }
    return [builders[name]() for name in view_names]


def _build_benchmark_service(
    *,
    registries: RegistryService,
    adapter_resolver: Callable[[str], Any],
    code_version: str,
    cancel_event: threading.Event | None = None,
    timeout_seconds_per_doc: float = 300.0,
) -> BenchmarkService:
    """Assemble ``BenchmarkService`` avec un loader filesystem.

    Phase B2.2 — quand ``cancel_event`` est fourni, le
    ``CorpusRunner.run`` est wrappé pour injecter l'event dans
    chaque appel.  Pattern strictement copié de
    ``_benchmark_execution.py:142-149`` (legacy).
    """
    pipeline_executor = PipelineExecutor(
        adapter_resolver=adapter_resolver,
    )
    corpus_runner = CorpusRunner(
        pipeline_executor,
        max_in_flight=2,
        timeout_seconds_per_doc=timeout_seconds_per_doc,
        poll_interval_seconds=0.05,
    )

    if cancel_event is not None:
        original_run = corpus_runner.run

        def _runner_run_with_cancel(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("cancel_event", cancel_event)
            return original_run(*args, **kwargs)

        corpus_runner.run = _runner_run_with_cancel  # type: ignore[method-assign]

    view_executor = DefaultEvaluationViewExecutor.from_registries(
        registries.metrics,
        registries.projectors,
        _filesystem_payload_loader,
    )
    return BenchmarkService(
        corpus_runner=corpus_runner,
        view_executor=view_executor,
        code_version=code_version,
    )


__all__ = [
    "_build_benchmark_service",
    "_build_pipelines",
    "_build_views",
    "_load_corpus",
]
