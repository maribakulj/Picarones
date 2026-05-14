"""Helpers de migration B4 — facilite la migration des tests catégorie A
de ``run_benchmark_via_service`` vers ``RunOrchestrator.execute_preset()``.

Ce module fournit ``run_via_orchestrator()``, un drop-in remplacement
de ``run_benchmark_via_service`` qui utilise ``RunOrchestrator`` en
interne mais préserve la signature et le retour ``BenchmarkResult``
legacy.

Cas d'usage typique dans un test catégorie A :

::

    # AVANT (legacy avec DeprecationWarning depuis Phase B3)
    from picarones.app.services.benchmark_runner import run_benchmark_via_service
    bm = run_benchmark_via_service(corpus, [adapter], profile="standard")

    # APRÈS (Phase B4 — via RunOrchestrator)
    from tests._migration_helpers import run_via_orchestrator
    bm = run_via_orchestrator(corpus, [adapter], profile="standard")

Le helper n'est utilisé QUE pour les tests durant la transition.  En
Phase B8 (post-deprecation), il sera supprimé et les tests devront
construire leur ``RunSpec`` explicitement (pattern utilisateur).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from picarones.evaluation.benchmark_result import BenchmarkResult
    from picarones.evaluation.corpus import Corpus


def _dummy_pipeline_yaml(name: str = "preset_pipeline") -> Any:
    """Construit un ``PipelineSpecYaml`` minimaliste pour satisfaire
    le validator ``RunSpec.pipelines`` (min_length=1).

    Le contenu est **ignoré** par ``execute_preset()`` qui utilise les
    ``pipeline_specs`` fournis en kwargs.  Le YAML dummy sert
    uniquement à passer la validation Pydantic.
    """
    from picarones.app.schemas.run_spec import PipelineSpecYaml, StepSpec
    from picarones.domain.artifacts import ArtifactType
    return PipelineSpecYaml(
        name=name,
        initial_inputs=(ArtifactType.IMAGE,),
        steps=(StepSpec(
            id="ocr",
            adapter_class="tests._migration_helpers.IgnoredByPreset",
            adapter_kwargs={},
            input_types=(ArtifactType.IMAGE,),
            output_types=(ArtifactType.RAW_TEXT,),
        ),),
    )


def run_via_orchestrator(
    corpus: "Corpus",
    engines: list[Any],
    *,
    char_exclude: Any | None = None,
    normalization_profile: Any | None = None,
    output_json: str | Path | None = None,
    code_version: str | None = None,
    show_progress: bool = True,  # noqa: ARG001 — absorbé pour compat
    progress_callback: Callable[[str, int, str], None] | None = None,
    timeout_seconds: float = 60.0,
    cancel_event: Any | None = None,
    partial_dir: str | Path | None = None,
    entity_extractor: Callable[[str], list[dict]] | str | None = None,
    profile: str = "standard",
) -> "BenchmarkResult":
    """Drop-in remplacement de ``run_benchmark_via_service`` via
    ``RunOrchestrator.execute_preset()``.

    Préserve la signature legacy pour permettre la migration mécanique
    des call sites de test (Phase B4).  Retourne un ``BenchmarkResult``
    construit via le converter ``run_result_to_benchmark_result``.

    Parameters
    ----------
    corpus, engines:
        Identiques à ``run_benchmark_via_service``.
    char_exclude, normalization_profile, output_json, code_version,
    show_progress, progress_callback, timeout_seconds, cancel_event,
    partial_dir, entity_extractor, profile:
        Identiques à ``run_benchmark_via_service``.

    Notes
    -----
    Quelques différences subtiles vs le legacy :

    - ``entity_extractor`` accepte un callable direct (legacy) OU un
      dotted path string (RunSpec).  Si callable, on l'invoque
      directement en post-process sur le ``BenchmarkResult``.
    - Le workspace temporaire est nettoyé automatiquement via
      ``TemporaryDirectory`` — ne pas s'attendre à des fichiers
      résiduels après l'appel.
    - ``normalization_profile`` accepte un objet ``NormalizationProfile``
      (legacy) OU un nom string (RunSpec).  Conversion automatique.
    """
    from picarones.app.schemas.run_spec import RunSpec
    from picarones.app.services._benchmark_adapter_resolver import (
        build_adapter_resolver,
        engine_to_pipeline_spec,
    )
    from picarones.app.services._benchmark_converter import (
        run_result_to_benchmark_result,
    )
    from picarones.app.services._benchmark_conversions import (
        corpus_to_corpus_spec,
    )
    from picarones.app.services.run_orchestrator import RunOrchestrator

    # Résolution code_version (cohérent avec run_benchmark_via_service:219).
    if code_version is None:
        import importlib
        try:
            code_version = importlib.import_module("picarones").__version__
        except (ImportError, AttributeError):
            code_version = "unknown"

    # ``normalization_profile`` legacy accepte un objet
    # NormalizationProfile.  RunSpec attend une string.  On convertit.
    norm_profile_str = normalization_profile
    if normalization_profile is not None and not isinstance(
        normalization_profile, str,
    ):
        norm_profile_str = getattr(normalization_profile, "name", None)

    # ``entity_extractor`` legacy accepte un callable direct.  RunSpec
    # attend un dotted path.  Si callable, on le traite post-process
    # comme run_benchmark_via_service le fait.
    entity_extractor_dotted: str | None = None
    entity_extractor_callable: Callable | None = None
    if entity_extractor is not None:
        if isinstance(entity_extractor, str):
            entity_extractor_dotted = entity_extractor
        elif callable(entity_extractor):
            entity_extractor_callable = entity_extractor

    with tempfile.TemporaryDirectory(prefix="picarones_b4_") as ws:
        ws_path = Path(ws)
        gt_dir = ws_path / "gt"
        gt_dir.mkdir()
        run_dir = ws_path / "run"
        run_dir.mkdir()

        corpus_spec = corpus_to_corpus_spec(corpus, workspace_dir=gt_dir)
        pipeline_specs = [engine_to_pipeline_spec(e) for e in engines]
        adapter_resolver = build_adapter_resolver(engines)
        pipeline_to_engine_name = {
            spec.name: engine.name
            for spec, engine in zip(pipeline_specs, engines)
        }

        # ``char_exclude`` peut être frozenset (legacy parsed) ou string
        # (RunSpec format).  RunSpec attend une string ; on convertit.
        char_exclude_str: str | None = None
        if char_exclude is not None:
            if isinstance(char_exclude, str):
                char_exclude_str = char_exclude
            else:
                char_exclude_str = "".join(sorted(char_exclude))

        spec = RunSpec(
            corpus_dir=str(ws_path),  # ignoré par execute_preset
            pipelines=(_dummy_pipeline_yaml(),),  # ignoré, juste pour validator
            views=("text_final",),
            output_dir=str(run_dir),
            char_exclude=char_exclude_str,
            normalization_profile=norm_profile_str,
            partial_dir=str(partial_dir) if partial_dir else None,
            entity_extractor=entity_extractor_dotted,
            profile=profile,
            output_json=str(output_json) if output_json else None,
            code_version=code_version,
            timeout_seconds_per_doc=timeout_seconds,
        )

        # Tag des engines avec le nom pour la map pipeline_to_engine
        # (utilisé par le progress_callback wrapper).
        wrapped_callback = None
        if progress_callback is not None:
            def wrapped_callback(
                pipeline_name: str, doc_idx: int, doc_id: str,
            ) -> None:
                engine_name = pipeline_to_engine_name.get(
                    pipeline_name, pipeline_name,
                )
                progress_callback(engine_name, doc_idx, doc_id)

        orch = RunOrchestrator(run_dir)
        orch_result = orch.execute_preset(
            spec,
            corpus_spec=corpus_spec,
            extracted_dir=gt_dir,
            pipeline_specs=pipeline_specs,
            adapter_resolver=adapter_resolver,
            adapter_kwargs={},
            progress_callback=wrapped_callback,
            cancel_event=cancel_event,
        )

        # Converti RunResult → BenchmarkResult via le converter
        # canonique (utilisé aussi par output_json en Phase B2.7).
        benchmark_result = run_result_to_benchmark_result(
            orch_result.run_result,
            corpus=corpus,
            engines=engines,
            char_exclude=char_exclude,  # passe la valeur originale
            normalization_profile=normalization_profile,
            profile=profile,
        )

        # NER attach post-process si entity_extractor callable fourni.
        # Cohérent avec run_benchmark_via_service:261-264.
        if entity_extractor_callable is not None:
            from picarones.app.services._benchmark_ner import (
                attach_ner_metrics_to_benchmark,
            )
            attach_ner_metrics_to_benchmark(
                benchmark_result, corpus, entity_extractor_callable,
            )

        # Sérialisation output_json si demandée (legacy comportement).
        if output_json is not None:
            from picarones.app.services._benchmark_persistence import (
                persist_benchmark_result_json,
            )
            persist_benchmark_result_json(
                benchmark_result, Path(output_json),
            )

        return benchmark_result


__all__ = ["run_via_orchestrator"]
