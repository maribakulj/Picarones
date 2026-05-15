"""``run_result_to_benchmark_result`` — converter ``RunResult``
(couche 4, rewrite) → ``BenchmarkResult`` (couche 3, legacy).

Module extrait du god-module ``benchmark_runner.py`` lors de la
Phase 6 (round 6) de l'audit code-quality (2026-05).

Le mapping est en **transposition** :

- ``RunResult`` itère par document puis par pipeline.
- ``BenchmarkResult`` itère par engine puis par document.

Le converter :

1. Récupère le ``PipelineResult`` pour chaque ``(engine, doc)``.
2. Lit les texts finaux (CORRECTED_TEXT prioritaire, RAW_TEXT
   sinon) depuis les artefacts.
3. Calcule CER/WER + hooks document-level (registres typés).
4. Aggrège per-engine via ``aggregate_metrics`` +
   ``run_corpus_aggregators``.
5. Consomme le journal de fallbacks importer (Phase 3.2 audit).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from picarones.app.services._benchmark_helpers import (
    _OCRResultLike,
    _build_pipeline_info,
    _build_pipeline_metadata,
    _extract_first_error,
    _extract_text_outputs,
    _extract_token_confidences,
    _resolve_corpus_lang,
    _safe_engine_version,
)
from picarones.domain.errors import PicaronesError

if TYPE_CHECKING:
    from picarones.evaluation.corpus import Corpus


def run_result_to_benchmark_result(
    run_result: Any,
    *,
    corpus: "Corpus",
    engines: list[Any],
    char_exclude: Any | None = None,
    normalization_profile: Any | None = None,
    profile: str = "standard",
) -> Any:
    """Transpose un ``RunResult`` en ``BenchmarkResult``.

    Parameters
    ----------
    run_result:
        ``RunResult`` produit par ``BenchmarkService.run``.
    corpus:
        Corpus d'origine — fournit ``ground_truth`` et
        ``image_path`` pour chaque document, dans le même ordre
        que ``run_result.document_results``.
    engines:
        Liste d'adapters dans l'ordre où leurs specs ont été
        passées à ``BenchmarkService.run`` (l'ordre détermine
        l'index dans ``RunDocumentResult.pipeline_results``).
    char_exclude:
        Filtre passé à ``compute_metrics``.  ``None`` par défaut.
    normalization_profile:
        Profil de normalisation passé à ``compute_metrics``.
    profile:
        Profil de hooks document-level / agrégateurs corpus-level
        (cf. ``picarones.evaluation.metric_hooks``).

    Returns
    -------
    BenchmarkResult
        Format compatible avec les consommateurs historiques
        (rapport HTML, persistance JSON, narrative engine).
    """
    from picarones.evaluation.benchmark_result import (
        BenchmarkResult,
        DocumentResult,
        EngineReport,
    )
    from picarones.evaluation.metric_hooks import (
        run_corpus_aggregators,
        run_document_hooks,
    )
    # Import nécessaire : les hooks ``builtin`` s'enregistrent dans le
    # registre global au moment de l'import du module (décorateurs).
    # Sans cette ligne, ``run_document_hooks(profile="standard", ...)``
    # retournerait un dict vide.
    import picarones.evaluation.metrics.builtin_hooks  # noqa: F401
    from picarones.evaluation.metric_result import aggregate_metrics
    from picarones.evaluation.metrics.text_metrics import compute_metrics

    documents = list(corpus.documents)
    if len(documents) != len(run_result.document_results):
        raise PicaronesError(
            f"Mismatch documents : corpus={len(documents)} vs "
            f"run_result={len(run_result.document_results)}.",
        )

    corpus_lang = _resolve_corpus_lang(corpus)
    engine_reports: list[Any] = []

    for engine_idx, engine in enumerate(engines):
        doc_results: list[Any] = []
        for doc_idx, document in enumerate(documents):
            run_doc = run_result.document_results[doc_idx]
            if engine_idx >= len(run_doc.pipeline_results):
                # Plus d'engines que de pipeline_results — incohérence.
                continue
            pipeline_result = run_doc.pipeline_results[engine_idx]

            text_final, ocr_intermediate = _extract_text_outputs(
                pipeline_result=pipeline_result,
            )
            engine_error = _extract_first_error(pipeline_result)
            duration = float(pipeline_result.duration_seconds)

            metrics = compute_metrics(
                document.ground_truth,
                text_final,
                normalization_profile=normalization_profile,
                char_exclude=char_exclude,
            )

            pipeline_metadata = _build_pipeline_metadata(
                engine=engine,
                ocr_intermediate=ocr_intermediate,
                ground_truth=document.ground_truth,
                hypothesis=text_final,
            )

            hook_values = run_document_hooks(
                profile=profile,
                ground_truth=document.ground_truth,
                hypothesis=text_final,
                image_path=str(document.image_path or ""),
                corpus_lang=corpus_lang,
                ocr_result=_OCRResultLike(
                    # ``success`` = « le pipeline s'est exécuté sans
                    # erreur », PAS « le pipeline a produit du texte
                    # non-vide ».  Bug B3-final (mai 2026) : la
                    # condition ``and bool(text_final)`` supprimait les
                    # hooks confusion/ligature/taxonomy/structure
                    # exactement quand l'OCR échouait (sortie vide sur
                    # documents patrimoniaux difficiles), alors que
                    # c'est précisément le cas où l'utilisateur veut
                    # ce diagnostic.  Un pipeline OCR+LLM « réparait »
                    # le success en produisant toujours du texte
                    # corrigé non-vide → incohérence (analyse présente
                    # pour LLM, absente pour Tesseract seul).  Une
                    # hypothèse vide sans erreur reste un résultat
                    # valide (mauvais) à analyser ; les hooks gèrent
                    # le cas (matrice = suppressions, ligature = 0).
                    # Cohérent avec ``image_quality`` qui n'a
                    # volontairement pas de ``requires_success``.
                    success=(engine_error is None),
                    token_confidences=_extract_token_confidences(
                        pipeline_result,
                    ),
                ),
            )

            doc_results.append(
                DocumentResult(
                    doc_id=document.doc_id,
                    image_path=str(document.image_path),
                    ground_truth=document.ground_truth,
                    hypothesis=text_final,
                    metrics=metrics,
                    duration_seconds=round(duration, 4),
                    engine_error=engine_error,
                    ocr_intermediate=ocr_intermediate,
                    pipeline_metadata=pipeline_metadata,
                    **hook_values,
                ),
            )

        aggregated = aggregate_metrics([d.metrics for d in doc_results])
        pipeline_info = _build_pipeline_info(engine)
        agg_values = run_corpus_aggregators(profile, doc_results)

        engine_reports.append(
            EngineReport(
                engine_name=engine.name,
                engine_version=_safe_engine_version(engine),
                engine_config=getattr(engine, "config", {}) or {},
                document_results=doc_results,
                aggregated_metrics=aggregated,
                pipeline_info=pipeline_info,
                **agg_values,
            ),
        )

    # Phase 3.2 audit code-quality — consomme le journal des
    # fallbacks d'importer (HTR-United, HuggingFace, etc.).  Le
    # détecteur narratif ``IMPORTER_FALLBACK_TRIGGERED`` lit
    # ``benchmark_data["importer_fallbacks"]``.
    from picarones.adapters.corpus._fallback_log import consume_fallback_log
    fallbacks = consume_fallback_log()
    metadata: dict[str, Any] = {}
    if fallbacks:
        metadata["importer_fallbacks"] = fallbacks

    # Phase B6 — transpose les ViewResult du RunResult en
    # ``view_results`` indexé : ``{view: {engine: {doc: {metric: value}}}}``.
    # Permet au rapport HTML de rendre des sections par vue
    # (TextView/AltoView/SearchView) avec le détail par pipeline.
    view_results_by_view: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for doc_idx, run_doc in enumerate(run_result.document_results):
        if doc_idx >= len(documents):
            break
        doc_id = documents[doc_idx].doc_id
        for vr in run_doc.view_results:
            view_bucket = view_results_by_view.setdefault(vr.view_name, {})
            engine_bucket = view_bucket.setdefault(vr.pipeline_name, {})
            engine_bucket[doc_id] = {
                metric: float(value)
                for metric, value in vr.metric_values.items()
                if isinstance(value, (int, float))
            }

    return BenchmarkResult(
        corpus_name=corpus.name,
        corpus_source=str(corpus.source_path) if corpus.source_path else None,
        document_count=len(documents),
        engine_reports=engine_reports,
        metadata=metadata,
        view_results=view_results_by_view,
    )


__all__ = ["run_result_to_benchmark_result"]
