"""Helpers tests — pattern ``RunOrchestrator`` pour les tests B4.

Phase B3-final (mai 2026) — ce module **n'est plus un re-export**
du shim de production ``legacy_runner_compat``.  Il implémente
directement le pattern 3 étapes (``prepare_preset_args`` →
``execute_preset`` → ``run_result_to_benchmark_result``) pour servir
les 6 fichiers de tests catégorie A migrés en Phase B4.

Pourquoi un helper test dédié plutôt qu'inline dans chaque test ?
-----------------------------------------------------------------
Les tests B4 (~30+ cas) consomment ce helper avec la même signature
que l'ancien ``run_benchmark_via_service``.  Le mettre inline dans
chaque test ajouterait ~10 lignes de boilerplate par cas, noyant
l'intention du test.

Différence vs le shim de production
-----------------------------------
Le shim ``legacy_runner_compat`` exposait ``run_via_orchestrator``
comme API publique pour CLI/Web — il a été supprimé en Phase
B3-final commit 5 au profit du pattern 3 étapes explicite dans
chaque call site.

Ce helper ``run_via_orchestrator`` est un **outil de test**
(préfixe ``_`` du module + dossier ``tests/``).  Son existence ne
constitue pas de la dette technique en production.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from picarones.evaluation.benchmark_result import BenchmarkResult
    from picarones.evaluation.corpus import Corpus


def run_via_orchestrator(
    corpus: "Corpus",
    engines: list[Any],
    *,
    char_exclude: Any | None = None,
    normalization_profile: Any | None = None,
    output_json: str | Path | None = None,
    code_version: str | None = None,
    show_progress: bool = True,  # noqa: ARG001 — absorbé pour compat tests
    progress_callback: Callable[[str, int, str], None] | None = None,
    timeout_seconds: float = 60.0,
    cancel_event: Any | None = None,
    partial_dir: str | Path | None = None,
    entity_extractor: Callable[[str], list[dict]] | str | None = None,
    profile: str = "standard",
) -> "BenchmarkResult":
    """Helper test : invoque ``RunOrchestrator`` et retourne un
    ``BenchmarkResult`` legacy.

    Reproduit le pattern 3 étapes utilisé en production (CLI/Web)
    pour que les tests B4 valident le même chemin que les utilisateurs.

    Signature alignée sur l'ancien ``run_benchmark_via_service`` pour
    minimiser le boilerplate dans les tests.

    NER attach
    ----------
    Si ``entity_extractor`` est un callable direct (pattern legacy),
    le helper invoque ``attach_ner_metrics_to_benchmark`` en post-
    process.  Si c'est un dotted path string, ``execute_preset`` le
    résout lui-même via ``RunSpec.entity_extractor``.
    """
    from picarones.app.services import (
        RunOrchestrator,
        prepare_preset_args,
        run_result_to_benchmark_result,
    )

    # Séparation callable vs dotted path (cf. shim historique).
    entity_extractor_dotted: str | None = None
    entity_extractor_callable: Callable | None = None
    if entity_extractor is not None:
        if isinstance(entity_extractor, str):
            entity_extractor_dotted = entity_extractor
        elif callable(entity_extractor):
            entity_extractor_callable = entity_extractor

    pipeline_to_engine_name = {
        # Construit après pipeline_specs ci-dessous (closure-friendly).
    }
    wrapped_callback = None
    if progress_callback is not None:
        def wrapped_callback(
            pipeline_name: str, doc_idx: int, doc_id: str,
        ) -> None:
            engine_name = pipeline_to_engine_name.get(
                pipeline_name, pipeline_name,
            )
            progress_callback(engine_name, doc_idx, doc_id)

    with tempfile.TemporaryDirectory(prefix="picarones_test_") as ws:
        ws_path = Path(ws)
        run_dir = ws_path / "run"
        args = prepare_preset_args(
            corpus, engines,
            workspace_dir=ws_path / "gt",
            output_dir=run_dir,
            char_exclude=char_exclude,
            normalization_profile=normalization_profile,
            partial_dir=partial_dir,
            entity_extractor=entity_extractor_dotted,
            profile=profile,
            output_json=output_json,
            timeout_seconds_per_doc=timeout_seconds,
            code_version=code_version,
        )
        # Map pipeline_name → engine.name pour le callback wrapper.
        pipeline_to_engine_name.update({
            spec.name: engine.name
            for spec, engine in zip(args.pipeline_specs, engines)
        })

        orch_result = RunOrchestrator(run_dir).execute_preset(
            spec=args.spec,
            corpus_spec=args.corpus_spec,
            extracted_dir=args.extracted_dir,
            pipeline_specs=args.pipeline_specs,
            adapter_resolver=args.adapter_resolver,
            adapter_kwargs=args.adapter_kwargs,
            progress_callback=wrapped_callback,
            cancel_event=cancel_event,
        )

        benchmark_result = run_result_to_benchmark_result(
            orch_result.run_result,
            corpus=corpus, engines=engines,
            char_exclude=char_exclude,
            normalization_profile=normalization_profile,
            profile=profile,
        )

        # NER attach post-process si callable direct fourni.
        if entity_extractor_callable is not None:
            from picarones.app.services._benchmark_ner import (
                attach_ner_metrics_to_benchmark,
            )
            attach_ner_metrics_to_benchmark(
                benchmark_result, corpus, entity_extractor_callable,
            )

        # Sérialisation output_json (legacy comportement).
        if output_json is not None:
            from picarones.app.services._benchmark_persistence import (
                persist_benchmark_result_json,
            )
            persist_benchmark_result_json(
                benchmark_result, Path(output_json),
            )

        return benchmark_result


__all__ = ["run_via_orchestrator"]
