"""Exécution pivotée par pipeline avec reprise (partial store).

Audit prod Phase B — extraction du plus gros bloc stateful de
``run_orchestrator.py`` (ex-``RunOrchestrator._execute_with_partial``,
~283 l).  Fonction libre : la seule dépendance à ``self`` était
``self._progress_callback``, désormais paramètre explicite.  Aucun
test ne l'appelait en classe-membre — ``run_orchestrator`` la
réimporte (façade) et bascule ses 2 call sites sur le nom
module-global, déplacement strictement verbatim (filet :
``tests/golden/test_run_orchestrator_characterization.py``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from picarones.domain.corpus import CorpusSpec
from picarones.app.services.run_orchestrator_helpers import (
    _default_gt_factory,
    _default_inputs_factory,
    _make_context_factory,
)

logger = logging.getLogger(__name__)


def execute_with_partial(
    *,
    spec: Any,
    bench: Any,
    corpus_spec: Any,
    pipeline_specs: list[Any],
    views: list[Any],
    adapter_kwargs: dict[str, Any],
    deps_lock: dict[str, Any],
    bin_lock: dict[str, Any],
    runtime_dir: Path | None = None,
    progress_callback: Any | None = None,
) -> Any:
    """Phase B2.3 — exécution pivotée par pipeline avec reprise.

    Pour chaque ``pipeline_spec`` :

    1. Calcule un fingerprint SHA-256 du run (pipeline structure +
       normalization + char_exclude + profile + corpus
       mtime/size + code_version).
    2. Cherche un fichier partial existant matchant ce fingerprint.
    3. Charge les ``PipelineResult`` déjà calculés.
    4. Filtre le corpus pour ne soumettre au ``BenchmarkService``
       que les documents manquants.
    5. Append chaque nouveau ``PipelineResult`` au fichier partial
       au fil de l'eau (un crash mid-run préserve ce qui a été
       calculé).
    6. À la fin d'une pipeline traitée intégralement, supprime
       le partial (cleanup).

    Le résultat final est un ``RunResult`` reconstruit à partir de
    tous les ``PipelineResult`` (chargés + nouveaux), réorganisés
    par document selon l'ordre du corpus original.

    Les ``ViewResult`` des documents repris du partial sont
    RECALCULÉS (fix défaut resume — cf. plus bas), pas perdus.
    """
    from picarones.app.results import RunResult
    from picarones.app.services._orchestrator_partial import (
        append_pipeline_result,
        compute_pipeline_fingerprint,
        delete_partial,
        filter_remaining_documents,
        load_partial_pipeline_results,
        partial_path_for_pipeline,
    )
    from picarones.domain.run_manifest import RunManifest
    from picarones.pipeline.run_result import RunDocumentResult

    partial_dir = Path(spec.partial_dir)
    partial_dir.mkdir(parents=True, exist_ok=True)

    # Map : pipeline_name → (partial_path, list[PipelineResult])
    per_pipeline_state: dict[str, tuple[Path, list[Any]]] = {}
    for pipeline_spec in pipeline_specs:
        fingerprint = compute_pipeline_fingerprint(
            pipeline_spec=pipeline_spec,
            corpus_spec=corpus_spec,
            normalization_profile=spec.normalization_profile,
            char_exclude=spec.char_exclude,
            profile=spec.profile,
            code_version=spec.code_version,
        )
        path = partial_path_for_pipeline(
            partial_dir=partial_dir,
            corpus_name=corpus_spec.name,
            pipeline_name=pipeline_spec.name,
            fingerprint=fingerprint,
        )
        loaded = load_partial_pipeline_results(path)
        if loaded:
            logger.info(
                "[run_orchestrator] reprise pipeline %r : %d/%d "
                "documents déjà persistés.",
                pipeline_spec.name,
                len(loaded), len(corpus_spec.documents),
            )
        per_pipeline_state[pipeline_spec.name] = (path, loaded)

    # Lance un sub-run par pipeline avec uniquement les docs
    # manquants.  Sub-RunResult séparés ; on agrège ensuite.
    sub_run_results: list[Any] = []
    for pipeline_spec in pipeline_specs:
        partial_path, loaded_results = per_pipeline_state[pipeline_spec.name]

        remaining_docs, deduplicated_loaded = filter_remaining_documents(
            corpus_spec.documents, loaded_results,
        )
        per_pipeline_state[pipeline_spec.name] = (
            partial_path, deduplicated_loaded,
        )

        if not remaining_docs:
            logger.info(
                "[run_orchestrator] pipeline %r déjà complet — "
                "skip exécution.", pipeline_spec.name,
            )
            # Cleanup du partial : le pipeline est entièrement
            # rechargé, plus besoin de garder le fichier sur disque.
            delete_partial(partial_path)
            continue

        sub_corpus_spec = CorpusSpec(
            name=corpus_spec.name,
            documents=tuple(remaining_docs),
            metadata=dict(corpus_spec.metadata),
        )

        sub_result = bench.run(
            corpus=sub_corpus_spec,
            pipelines=[pipeline_spec],
            views=views,
            ground_truth_factory=_default_gt_factory,
            pipeline_inputs_factory=_default_inputs_factory,
            context_factory=_make_context_factory(
                spec.code_version,
                progress_callback=progress_callback,
                workspace_uri=str(runtime_dir) if runtime_dir else None,
            ),
            adapter_kwargs=adapter_kwargs,
            dependencies_lock=deps_lock,
            system_binaries_lock=bin_lock,
            metadata={
                "orchestrator":
                "picarones.app.services.run_orchestrator",
                "partial_pipeline": pipeline_spec.name,
            },
        )
        sub_run_results.append(sub_result)

        # Persiste chaque nouveau PipelineResult au partial.
        new_count = 0
        for doc_result in sub_result.document_results:
            for pr in doc_result.pipeline_results:
                if pr.pipeline_name == pipeline_spec.name:
                    append_pipeline_result(partial_path, pr)
                    new_count += 1

        # Si tous les docs du corpus original ont été traités
        # (loaded + new) → cleanup du partial.
        loaded_doc_ids = {pr.document_id for pr in deduplicated_loaded}
        new_doc_ids = {
            pr.document_id
            for doc_result in sub_result.document_results
            for pr in doc_result.pipeline_results
            if pr.pipeline_name == pipeline_spec.name
        }
        all_doc_ids = {d.id for d in corpus_spec.documents}
        if (loaded_doc_ids | new_doc_ids) >= all_doc_ids:
            delete_partial(partial_path)
            logger.info(
                "[run_orchestrator] pipeline %r complet (%d docs) "
                "— partial supprimé.",
                pipeline_spec.name, len(all_doc_ids),
            )

    # Reconstruit le RunResult final : pour chaque doc du corpus
    # original, agrège les PipelineResult de tous les pipelines.
    # Map (doc_id, pipeline_name) → PipelineResult
    pr_index: dict[tuple[str, str], Any] = {}
    # Map (doc_id, pipeline_name) → list[ViewResult]
    vr_index: dict[tuple[str, str], list[Any]] = {}

    # Charge les pipeline_results depuis les partials (rechargés)
    # ET recalcule leurs view_results.
    #
    # Fix défaut resume (harnais de caractérisation) : le partial
    # store persiste ``PipelineResult`` mais PAS ``ViewResult``.
    # Sans ce recalcul, les docs repris du partial sortaient avec
    # leurs pipeline_results mais SANS view_results → métriques
    # agrégées (CER…) silencieusement faussées après reprise.
    # Les vues sont une fonction PURE de (pipeline_results + GT +
    # profil) : les recalculer pour les docs repris est correct,
    # ne change pas le format du partial, et garantit des vues
    # fraîches (cohérentes avec le code d'éval courant) plutôt que
    # potentiellement périmées si on les avait persistées.
    doc_by_id = {d.id: d for d in corpus_spec.documents}
    for pipeline_name, (_, loaded_list) in per_pipeline_state.items():
        for pr in loaded_list:
            pr_index[(pr.document_id, pipeline_name)] = pr
            doc = doc_by_id.get(pr.document_id)
            if doc is None:
                continue
            # ``_evaluate_document_in_views`` est l'entrée d'éval
            # canonique (même chemin que le run frais) ; appelée
            # ici en isolation sur le PR rechargé.
            for vr in bench._evaluate_document_in_views(
                document=doc,
                pipeline_results=[pr],
                views=views,
                ground_truth_factory=_default_gt_factory,
            ):
                vr_index.setdefault(
                    (pr.document_id, ""), [],
                ).append(vr)

    # Charge les pipeline_results et view_results depuis les sub-runs.
    for sub_result in sub_run_results:
        for sub_doc in sub_result.document_results:
            for pr in sub_doc.pipeline_results:
                pr_index[(sub_doc.document_id, pr.pipeline_name)] = pr
            for vr in sub_doc.view_results:
                # ``ViewResult.pipeline_name`` n'existe pas ; on
                # regroupe par doc seulement (pas suffisamment
                # granulaire mais OK pour la sortie).
                vr_index.setdefault(
                    (sub_doc.document_id, ""), [],
                ).append(vr)

    # Construit les RunDocumentResult dans l'ordre du corpus.
    final_doc_results: list[Any] = []
    for doc in corpus_spec.documents:
        doc_pipeline_results = tuple(
            pr_index[(doc.id, ps.name)]
            for ps in pipeline_specs
            if (doc.id, ps.name) in pr_index
        )
        doc_view_results = tuple(vr_index.get((doc.id, ""), []))
        final_doc_results.append(RunDocumentResult(
            document_id=doc.id,
            pipeline_results=doc_pipeline_results,
            view_results=doc_view_results,
        ))

    # Synthétise un RunManifest minimal (on prend celui d'un
    # sub-run s'il y en a eu, sinon on synthétise from scratch).
    if sub_run_results:
        # Fusionne les pipeline_specs de tous les sub-runs.
        base_manifest = sub_run_results[0].manifest
        manifest = RunManifest(
            run_id=base_manifest.run_id,
            corpus_name=corpus_spec.name,
            n_documents=len(corpus_spec.documents),
            pipeline_specs=tuple(pipeline_specs),
            adapter_kwargs=adapter_kwargs,
            view_specs=tuple(views),
            code_version=spec.code_version,
            started_at=base_manifest.started_at,
            completed_at=base_manifest.completed_at,
            dependencies_lock=deps_lock,
            system_binaries_lock=bin_lock,
            metadata={
                "orchestrator":
                "picarones.app.services.run_orchestrator",
                "partial_dir": str(partial_dir),
            },
        )
    else:
        # Tous les pipelines ont été chargés depuis partial — pas
        # de sub-run.  On synthétise un manifest from scratch.
        from picarones.app.services.benchmark_service import (
            _default_run_id,
        )
        from picarones.domain.run_manifest import utcnow
        now = utcnow()
        manifest = RunManifest(
            run_id=_default_run_id(corpus_spec.name, now),
            corpus_name=corpus_spec.name,
            n_documents=len(corpus_spec.documents),
            pipeline_specs=tuple(pipeline_specs),
            adapter_kwargs=adapter_kwargs,
            view_specs=tuple(views),
            code_version=spec.code_version,
            started_at=now,
            completed_at=now,
            dependencies_lock=deps_lock,
            system_binaries_lock=bin_lock,
            metadata={
                "orchestrator":
                "picarones.app.services.run_orchestrator",
                "partial_dir": str(partial_dir),
                "fully_resumed": "true",
            },
        )

    return RunResult(
        manifest=manifest,
        document_results=tuple(final_doc_results),
    )


__all__ = ["execute_with_partial"]
