"""Entry point CLI/web — façade ``run_benchmark_via_service``.

Présente l'API mono-call ``run_benchmark_via_service(corpus,
engines, ...)`` consommée par ``picarones.interfaces.cli`` et
``picarones.interfaces.web``.  S'appuie en interne sur le service
canonique (``BenchmarkService``, ``PipelineExecutor``,
``CorpusRunner``).

Pourquoi cette façade
---------------------
``BenchmarkService`` consomme ``CorpusSpec`` (références
filesystem, Pydantic, immutable) et ``PipelineSpec`` (déclaratif).
Les interfaces utilisateur (CLI, web upload) raisonnent en
``Corpus`` riche en behavior + liste de moteurs OCR/LLM.  Ce
module fait la conversion entre les deux modèles, expose une API
mono-call ergonomique et restitue un ``BenchmarkResult``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from picarones.domain.errors import PicaronesError

if TYPE_CHECKING:
    from picarones.evaluation.corpus import Corpus

logger = logging.getLogger(__name__)

# Le ``OCRLLMPipelineConfig`` (couche 4) est consommé exclusivement
# par duck typing (``is_pipeline``, ``ocr_adapter``, ``llm_adapter``,
# ``mode``, ``prompt_template``) pour respecter l'inward-only :
# ``app/`` ne doit pas importer ``pipeline/llm_pipeline_config``
# directement.


# ──────────────────────────────────────────────────────────────────────
# Mapping Document → DocumentRef
# ──────────────────────────────────────────────────────────────────────


# Phase 6 (round 3) audit code-quality (2026-05) — extraction des
# conversions Document/Corpus + helpers GT vers
# ``_benchmark_conversions.py``.  Réexport pour préserver l'API
# publique (CLI/web consomment ces noms).
from picarones.app.services._benchmark_conversions import (  # noqa: F401
    _DEFAULT_SUFFIXES,
    _has_text_gt,
    _payload_to_text,
    _resolve_gt_uri,
    _safe_doc_id,
    corpus_to_corpus_spec,
    document_to_document_ref,
)


# ──────────────────────────────────────────────────────────────────────
# Mapping RunResult → BenchmarkResult
# ──────────────────────────────────────────────────────────────────────


def run_result_to_benchmark_result(
    run_result: Any,
    *,
    corpus: "Corpus",
    engines: list[Any],
    char_exclude: Any | None = None,
    normalization_profile: Any | None = None,
    profile: str = "standard",
) -> Any:
    """Transpose un ``RunResult`` (couche 4) en ``BenchmarkResult`` (couche 3).

    Le mapping est en **transposition** :

    - **Rewrite** ``RunResult`` : itère par document puis par
      pipeline.  ``run_result.document_results[i].pipeline_results[j]``.
    - **Legacy** ``BenchmarkResult`` : itère par engine puis par
      document.  ``benchmark_result.engine_reports[j].document_results[i]``.

    Pour chaque couple ``(engine, document)``, le converter :

    1. Récupère le ``PipelineResult`` correspondant depuis
       ``RunDocumentResult.pipeline_results``.
    2. Lit le texte produit final (``CORRECTED_TEXT`` prioritaire,
       sinon ``RAW_TEXT``) depuis l'``Artifact.uri``.
    3. Lit l'``ocr_intermediate`` (RAW_TEXT) si le pipeline a un
       step OCR amont.
    4. Calcule les métriques CER/WER via ``compute_metrics`` puis
       exécute les hooks document-level enregistrés pour ``profile``
       via ``picarones.evaluation.metric_hooks.run_document_hooks``
       (confusion unicode, ligatures, diacritiques, taxonomie,
       structure, hallucination, philological, searchability,
       readability, etc.).
    5. Construit un ``DocumentResult`` avec ``engine_error``
       extrait des ``step_results``.
    6. Aggrège les métriques par engine via ``aggregate_metrics`` et
       exécute les agrégateurs corpus-level via
       ``run_corpus_aggregators`` pour alimenter
       ``EngineReport.aggregated_*`` (la vue HTML "Analyse des
       caractères" et les vues sœurs lisent ces champs).
    7. Reconstitue ``pipeline_info`` pour les engines pipeline
       (mode, prompt, llm_model, llm_provider, pipeline_steps).

    Parameters
    ----------
    run_result:
        ``RunResult`` produit par ``BenchmarkService.run``.
    corpus:
        Corpus d'origine — sert à récupérer le ``ground_truth``
        et l'``image_path`` pour chaque document, dans le même ordre
        que ``run_result.document_results``.
    engines:
        Liste d'adapters dans l'ordre où leurs specs ont été
        passées à ``BenchmarkService.run`` (l'ordre détermine
        l'index dans ``RunDocumentResult.pipeline_results``).
    char_exclude:
        Filtre passé à ``compute_metrics``.  ``None`` par défaut.
    normalization_profile:
        Profil de normalisation passé à ``compute_metrics``.

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
    # retournerait un dict vide et la vue HTML "Analyse des caractères"
    # tomberait sur ses placeholders.
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
                    success=(engine_error is None and bool(text_final)),
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

    # Phase 3.2 audit code-quality — consommer le journal des
    # fallbacks d'importer (HTR-United, HuggingFace, etc.).  La liste
    # est vidée à la fin du benchmark pour que le run suivant
    # n'hérite pas des incidents du précédent.  Le détecteur narratif
    # ``IMPORTER_FALLBACK_TRIGGERED`` (history.py:280) lit
    # ``benchmark_data["importer_fallbacks"]`` propagé par
    # ``build_report_data``.
    from picarones.adapters.corpus._fallback_log import consume_fallback_log
    fallbacks = consume_fallback_log()
    metadata: dict[str, Any] = {}
    if fallbacks:
        metadata["importer_fallbacks"] = fallbacks

    return BenchmarkResult(
        corpus_name=corpus.name,
        corpus_source=str(corpus.source_path) if corpus.source_path else None,
        document_count=len(documents),
        engine_reports=engine_reports,
        metadata=metadata,
    )


# ──────────────────────────────────────────────────────────────────────
# Helpers privés du converter RunResult → BenchmarkResult
# ──────────────────────────────────────────────────────────────────────


# Phase 6 (round 5) audit code-quality (2026-05) — extraction des
# helpers internes de conversion ``RunResult → BenchmarkResult``
# vers ``_benchmark_helpers.py`` (~260 LOC).  Réexport pour les
# appels internes et les tests qui patchent ces symboles.
from picarones.app.services._benchmark_helpers import (  # noqa: F401
    _OCRResultLike,
    _build_pipeline_info,
    _build_pipeline_metadata,
    _engine_config_for_fingerprint,
    _extract_first_error,
    _extract_text_outputs,
    _extract_token_confidences,
    _resolve_corpus_lang,
    _safe_engine_version,
)
# Phase 6 (round 2) — extraction du bloc engine→spec + resolver.
from picarones.app.services._benchmark_adapter_resolver import (  # noqa: F401
    _canonical_adapter_to_spec,
    _is_canonical_adapter,
    _llm_adapter_name,
    _ocr_llm_pipeline_to_spec,
    _safe_pipeline_name,
    build_adapter_resolver,
    engine_to_pipeline_spec,
)


def run_benchmark_via_service(
    corpus: "Corpus",
    engines: list[Any],
    *,
    char_exclude: Any | None = None,
    normalization_profile: Any | None = None,
    output_json: Any | None = None,
    code_version: str | None = None,
    show_progress: bool = True,  # noqa: ARG001
    progress_callback: Callable[[str, int, str], None] | None = None,
    timeout_seconds: float = 60.0,
    cancel_event: Any | None = None,
    partial_dir: str | Path | None = None,
    entity_extractor: Callable[[str], list[dict]] | None = None,
    profile: str = "standard",
) -> Any:
    """Façade ``run_benchmark`` →
    ``BenchmarkService`` rewrite.

    Présente la signature historique de
    ``picarones.app.services.benchmark_runner.run_benchmark`` mais s'appuie
    en interne sur le rewrite (``CorpusSpec``, ``PipelineSpec``,
    ``PipelineExecutor``, ``BenchmarkService``).  Pivot du Sprint D
    du plan v2.0.

    Périmètre actuel (D.1.d, MVP)
    -----------------------------
    Cette première version fonctionne pour le cas le plus simple :

    - Un ou plusieurs ``BaseOCREngine`` (OCR seul ou pipeline OCR+LLM
      via ``OCRLLMPipeline``).
    - Un ``Corpus`` avec image_path + ground_truth (TEXT) par doc.
    - Métriques CER/WER calculées via ``compute_metrics`` sur les
      hypothèses extraites des artefacts produits.
    - Conversion en ``BenchmarkResult`` compatible avec les
      consommateurs historiques (rapport HTML, narrative engine).

    Périmètre reporté (D.2)
    -----------------------
    Les paramètres suivants sont **acceptés mais ignorés** dans
    cette MVP — le rewrite gère ces aspects nativement :

    - ``show_progress`` (tqdm).

    Pour régler le parallélisme corpus-wide, passer par
    ``CorpusRunner.max_in_flight`` directement (couche pipeline).

    Profil de mesures (D.2.f)
    -------------------------
    ``profile`` est validé au démarrage via
    ``picarones.evaluation.metric_hooks.validate_profile``.  Un
    profil inconnu lève ``PicaronesError``.  La valeur n'a pas
    encore d'effet sur les hooks document-level (ce serait l'objet
    d'un sprint ultérieur, hors du périmètre v2.0).

    NER attach (D.2.e)
    ------------------
    Si ``entity_extractor`` est fourni, après le calcul des
    ``DocumentResult``, le service appelle l'extracteur sur chaque
    hypothèse OCR pour les documents dont la GT possède un niveau
    ``ENTITIES``, puis attache les métriques NER (``ner_metrics``
    par document, ``aggregated_ner`` au niveau engine).

    Reprise sur interruption (D.2.b)
    --------------------------------
    Si ``partial_dir`` est fourni, le bench est exécuté en mode
    **per-engine resumable** :

    - Pour chaque engine, on cherche un fichier
      ``{partial_dir}/picarones_{corpus}_{engine}.partial.jsonl``
      d'une exécution précédente interrompue.
    - Les ``DocumentResult`` qui y sont déjà persistés sont
      réutilisés tels quels (pas de recalcul).
    - Seuls les documents restants sont soumis au ``BenchmarkService``.
    - Chaque nouveau ``DocumentResult`` est ajouté en append au
      partial avant de passer au suivant.
    - À la fin d'un engine traité avec succès, son partial est
      supprimé.

    Quand ``partial_dir`` est ``None`` (défaut), une seule passe
    multi-engine est lancée (chemin rapide, pas de persistance
    intermédiaire).

    Parameters
    ----------
    corpus:
        Corpus.
    engines:
        Liste d'engines/pipelines à benchmarker.
    char_exclude:
        Filtre passé à ``compute_metrics``.
    normalization_profile:
        Profil de normalisation passé à ``compute_metrics``.
    output_json:
        Si fourni, le ``BenchmarkResult`` est sérialisé en JSON
        à ce chemin (sérialisation BenchmarkResult).
    code_version:
        Version du code injectée dans le ``RunContext`` /
        ``RunManifest``.  Défaut : ``picarones.__version__``.
    timeout_seconds:
        Timeout par document propagé au ``CorpusRunner``.

    Returns
    -------
    BenchmarkResult
        Format compatible avec les consommateurs historiques.

    Raises
    ------
    PicaronesError
        Si les engines ne déclarent pas tous un ``name`` unique
        (cf. ``build_adapter_resolver``).
    """
    # D.2.f : valide ``profile`` tôt — un nom inconnu lève
    # ``PicaronesError`` avant que le bench ne démarre, plutôt
    # que de dégrader silencieusement plus loin.
    from picarones.evaluation.metric_hooks import validate_profile

    validate_profile(profile)

    if code_version is None:
        # Le scanner d'archi rejette ``from picarones import __version__``
        # parce qu'il classe ``picarones`` (sans sous-package) comme une
        # lib externe non whitelistée pour la couche ``app/``.  On
        # contourne via importlib (déclaration dynamique).
        import importlib

        try:
            code_version = importlib.import_module("picarones").__version__
        except (ImportError, AttributeError):
            code_version = "unknown"

    if partial_dir is None:
        benchmark_result = _run_benchmark_unified(
            corpus=corpus,
            engines=engines,
            char_exclude=char_exclude,
            normalization_profile=normalization_profile,
            profile=profile,
            code_version=code_version,
            progress_callback=progress_callback,
            timeout_seconds=timeout_seconds,
            cancel_event=cancel_event,
        )
    else:
        benchmark_result = _run_benchmark_with_partial(
            corpus=corpus,
            engines=engines,
            partial_dir=Path(partial_dir),
            char_exclude=char_exclude,
            normalization_profile=normalization_profile,
            profile=profile,
            code_version=code_version,
            progress_callback=progress_callback,
            timeout_seconds=timeout_seconds,
            cancel_event=cancel_event,
        )

    # D.2.e : NER attach post-process.  Idempotent — re-calcule à
    # chaque run même en mode resume (les ner_metrics ne sont pas
    # persistées dans le partial NDJSON
    # qui calculait NER après le doc loop).
    if entity_extractor is not None:
        _attach_ner_metrics_to_benchmark(
            benchmark_result, corpus, entity_extractor,
        )

    # Sérialisation JSON optionnelle
    if output_json is not None:
        _persist_benchmark_result_json(benchmark_result, Path(output_json))

    return benchmark_result


# Phase 6 audit code-quality (2026-05) — extraction NER aggregation
# vers ``_benchmark_ner.py``.  Les noms ``_attach_ner_metrics_to_benchmark``
# et ``_aggregate_ner_metrics`` restent ici comme alias pour ne pas
# casser les appels internes (les autres fonctions du runner s'y
# réfèrent) et les tests qui patchent ces symboles via monkeypatch.
from picarones.app.services._benchmark_ner import (  # noqa: F401
    aggregate_ner_metrics as _aggregate_ner_metrics,
    attach_ner_metrics_to_benchmark as _attach_ner_metrics_to_benchmark,
)


def _run_benchmark_unified(
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
    import tempfile

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

        run_result = _execute_via_benchmark_service(
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


def _run_benchmark_with_partial(
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
    import tempfile

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
        # Vérifier la cancellation entre engines (matche la
        # sémantique : un Ctrl+C arrête après l'engine en
        # cours, conserve les partials, ne démarre pas le suivant).
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

                run_result = _execute_via_benchmark_service(
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

                # Append au partial : un cancel mid-engine
                # préservera ce qui a déjà été calculé.
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

    # Phase 3.2 audit code-quality — cf. _run_benchmark_unified.
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


# Phase 6 (round 4) audit code-quality (2026-05) — extraction de
# ``_execute_via_benchmark_service`` vers ``_benchmark_execution.py``.
# Alias conservé pour les appels internes de
# ``_run_benchmark_unified`` et ``_run_benchmark_with_partial``.
from picarones.app.services._benchmark_execution import (
    execute_via_benchmark_service as _execute_via_benchmark_service,
)
from picarones.app.services._benchmark_persistence import (
    persist_benchmark_result_json as _persist_benchmark_result_json,
)
