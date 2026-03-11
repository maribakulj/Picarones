"""Orchestrateur du benchmark.

Exécute les moteurs OCR/HTR sur le corpus de manière parallèle :
- ``ProcessPoolExecutor`` pour les moteurs CPU-bound (Tesseract, Pero OCR, Kraken)
- ``ThreadPoolExecutor``  pour les moteurs IO-bound / API (Mistral, Google, Azure, LLMs)

Les résultats partiels sont sauvegardés après chaque document dans un fichier
``{partial_dir}/{corpus}_{engine}.partial.json`` (NDJSON).  Si le benchmark est
interrompu, la prochaine exécution reprend automatiquement depuis ce fichier.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import tempfile
import time
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from picarones.core.corpus import Corpus
from picarones.core.metrics import MetricsResult, compute_metrics
from picarones.core.results import BenchmarkResult, DocumentResult, EngineReport
from picarones.engines.base import BaseOCREngine, EngineResult

logger = logging.getLogger(__name__)

# Classes de moteurs CPU-bound → ProcessPoolExecutor
_CPU_BOUND_ENGINE_CLASSES = frozenset({"TesseractEngine", "PeroOCREngine", "KrakenEngine"})


# ---------------------------------------------------------------------------
# Workers de niveau module (requis pour ProcessPoolExecutor — picklables)
# ---------------------------------------------------------------------------

def _cpu_doc_worker(args: tuple) -> "DocumentResult":
    """Worker pour ProcessPoolExecutor (moteurs CPU-bound).

    Instancie le moteur dans le sous-processus, exécute l'OCR et calcule
    toutes les métriques.  Doit être une fonction de niveau module pour être
    sérialisable par ``pickle``.
    """
    engine_module, engine_class_name, engine_config, doc_id, image_path, ground_truth, char_exclude_chars = args
    import importlib
    mod = importlib.import_module(engine_module)
    engine_cls = getattr(mod, engine_class_name)
    engine = engine_cls(config=engine_config)
    ocr_result = engine.run(image_path)
    char_exclude = frozenset(char_exclude_chars) if char_exclude_chars else None
    return _compute_document_result(
        doc_id=doc_id,
        image_path=image_path,
        ground_truth=ground_truth,
        ocr_result=ocr_result,
        char_exclude=char_exclude,
    )


def _io_doc_worker(
    engine: BaseOCREngine,
    doc: object,
    char_exclude: Optional[frozenset],
) -> "DocumentResult":
    """Worker pour ThreadPoolExecutor (moteurs IO-bound / API).

    Exécute l'OCR et calcule les métriques dans un thread.  L'instance du
    moteur est partagée entre les threads — les adaptateurs HTTP sont
    généralement sans état mutable entre les appels.
    """
    ocr_result = engine.run(doc.image_path)  # type: ignore[attr-defined]
    return _compute_document_result(
        doc_id=doc.doc_id,  # type: ignore[attr-defined]
        image_path=str(doc.image_path),  # type: ignore[attr-defined]
        ground_truth=doc.ground_truth,  # type: ignore[attr-defined]
        ocr_result=ocr_result,
        char_exclude=char_exclude,
    )


# ---------------------------------------------------------------------------
# Calcul documentaire centralisé
# ---------------------------------------------------------------------------

def _compute_document_result(
    doc_id: str,
    image_path: str,
    ground_truth: str,
    ocr_result: EngineResult,
    char_exclude: Optional[frozenset],
) -> DocumentResult:
    """Calcule toutes les métriques pour un document et retourne un DocumentResult.

    Utilisable à la fois dans le processus principal (IO-bound) et dans les
    sous-processus créés par ProcessPoolExecutor (CPU-bound).
    Les imports lourds sont différés pour accélérer le démarrage des sous-processus.
    Les analyses secondaires qui échouent sont loguées en WARNING et non propagées :
    le benchmark continue avec les métriques de base disponibles.
    """
    import logging as _log
    _logger = _log.getLogger(__name__)

    if ocr_result.success:
        metrics = compute_metrics(ground_truth, ocr_result.text, char_exclude=char_exclude)
    else:
        metrics = MetricsResult(
            cer=1.0, cer_nfc=1.0, cer_caseless=1.0,
            wer=1.0, wer_normalized=1.0, mer=1.0, wil=1.0,
            reference_length=len(ground_truth),
            hypothesis_length=0,
            error=ocr_result.error,
        )

    ocr_intermediate = ocr_result.metadata.get("ocr_intermediate")
    pipeline_meta: dict = {}

    if ocr_result.metadata.get("is_pipeline"):
        pipeline_meta = {
            "pipeline_mode": ocr_result.metadata.get("pipeline_mode"),
            "prompt_file": ocr_result.metadata.get("prompt_file"),
            "llm_model": ocr_result.metadata.get("llm_model"),
            "llm_provider": ocr_result.metadata.get("llm_provider"),
        }
        if ocr_intermediate is not None and ocr_result.success:
            try:
                from picarones.pipelines.over_normalization import detect_over_normalization
                over_norm = detect_over_normalization(
                    ground_truth=ground_truth,
                    ocr_text=ocr_intermediate,
                    llm_text=ocr_result.text,
                )
                pipeline_meta["over_normalization"] = over_norm.as_dict()
            except Exception as e:
                _logger.warning("[over_normalization] fonctionnalité dégradée : %s", e)

    confusion_data = None
    char_scores_data = None
    taxonomy_data = None
    structure_data = None
    image_quality_data = None

    if ocr_result.success:
        try:
            from picarones.core.confusion import build_confusion_matrix
            cm = build_confusion_matrix(ground_truth, ocr_result.text)
            confusion_data = cm.as_dict()
        except Exception as e:
            _logger.warning("[confusion] fonctionnalité dégradée : %s", e)

        try:
            from picarones.core.char_scores import compute_ligature_score, compute_diacritic_score
            lig = compute_ligature_score(ground_truth, ocr_result.text)
            diac = compute_diacritic_score(ground_truth, ocr_result.text)
            char_scores_data = {"ligature": lig.as_dict(), "diacritic": diac.as_dict()}
        except Exception as e:
            _logger.warning("[char_scores] fonctionnalité dégradée : %s", e)

        try:
            from picarones.core.taxonomy import classify_errors
            tax = classify_errors(ground_truth, ocr_result.text)
            taxonomy_data = tax.as_dict()
        except Exception as e:
            _logger.warning("[taxonomy] fonctionnalité dégradée : %s", e)

        try:
            from picarones.core.structure import analyze_structure
            struct = analyze_structure(ground_truth, ocr_result.text)
            structure_data = struct.as_dict()
        except Exception as e:
            _logger.warning("[structure] fonctionnalité dégradée : %s", e)

    try:
        from picarones.core.image_quality import analyze_image_quality
        iq = analyze_image_quality(image_path)
        if iq.error is None:
            image_quality_data = iq.as_dict()
    except Exception as e:
        _logger.warning("[image_quality] fonctionnalité dégradée : %s", e)

    return DocumentResult(
        doc_id=doc_id,
        image_path=image_path,
        ground_truth=ground_truth,
        hypothesis=ocr_result.text,
        metrics=metrics,
        duration_seconds=ocr_result.duration_seconds,
        engine_error=ocr_result.error,
        ocr_intermediate=ocr_intermediate,
        pipeline_metadata=pipeline_meta,
        confusion_matrix=confusion_data,
        char_scores=char_scores_data,
        taxonomy=taxonomy_data,
        structure=structure_data,
        image_quality=image_quality_data,
    )


def _make_timeout_doc_result(doc: object, timeout_seconds: float) -> DocumentResult:
    """DocumentResult synthétique pour un document ayant dépassé le timeout."""
    err = f"timeout ({timeout_seconds:.0f}s)"
    metrics = MetricsResult(
        cer=1.0, cer_nfc=1.0, cer_caseless=1.0,
        wer=1.0, wer_normalized=1.0, mer=1.0, wil=1.0,
        reference_length=len(doc.ground_truth),  # type: ignore[attr-defined]
        hypothesis_length=0,
        error=err,
    )
    return DocumentResult(
        doc_id=doc.doc_id,  # type: ignore[attr-defined]
        image_path=str(doc.image_path),  # type: ignore[attr-defined]
        ground_truth=doc.ground_truth,  # type: ignore[attr-defined]
        hypothesis="",
        metrics=metrics,
        duration_seconds=timeout_seconds,
        engine_error=err,
    )


def _make_error_doc_result(doc: object, error_msg: str) -> DocumentResult:
    """DocumentResult synthétique pour un document en erreur inattendue."""
    metrics = MetricsResult(
        cer=1.0, cer_nfc=1.0, cer_caseless=1.0,
        wer=1.0, wer_normalized=1.0, mer=1.0, wil=1.0,
        reference_length=len(doc.ground_truth),  # type: ignore[attr-defined]
        hypothesis_length=0,
        error=error_msg,
    )
    return DocumentResult(
        doc_id=doc.doc_id,  # type: ignore[attr-defined]
        image_path=str(doc.image_path),  # type: ignore[attr-defined]
        ground_truth=doc.ground_truth,  # type: ignore[attr-defined]
        hypothesis="",
        metrics=metrics,
        duration_seconds=0.0,
        engine_error=error_msg,
    )


# ---------------------------------------------------------------------------
# Résultats partiels (sauvegarde / reprise)
# ---------------------------------------------------------------------------

def _sanitize_filename(s: str) -> str:
    return re.sub(r"[^\w\-]", "_", s)[:64]


def _partial_path(
    corpus_name: str,
    engine_name: str,
    partial_dir: Optional[str | Path],
) -> Path:
    base = Path(partial_dir) if partial_dir else Path(tempfile.gettempdir())
    name = (
        f"picarones_{_sanitize_filename(corpus_name)}"
        f"_{_sanitize_filename(engine_name)}.partial.json"
    )
    return base / name


def _load_partial(
    corpus_name: str,
    engine_name: str,
    partial_dir: Optional[str | Path],
) -> tuple[Path, list[DocumentResult]]:
    """Charge les résultats partiels d'une exécution précédente interrompue.

    Returns
    -------
    (path, results) — chemin du fichier partiel et liste des DocumentResult déjà calculés.
    """
    path = _partial_path(corpus_name, engine_name, partial_dir)
    results: list[DocumentResult] = []
    if not path.exists():
        return path, results

    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                m = d.get("metrics", {})
                metrics = MetricsResult(
                    cer=m.get("cer", 1.0),
                    cer_nfc=m.get("cer_nfc", 1.0),
                    cer_caseless=m.get("cer_caseless", 1.0),
                    wer=m.get("wer", 1.0),
                    wer_normalized=m.get("wer_normalized", 1.0),
                    mer=m.get("mer", 1.0),
                    wil=m.get("wil", 1.0),
                    reference_length=m.get("reference_length", 0),
                    hypothesis_length=m.get("hypothesis_length", 0),
                    error=m.get("error"),
                )
                results.append(DocumentResult(
                    doc_id=d["doc_id"],
                    image_path=d.get("image_path", ""),
                    ground_truth=d.get("ground_truth", ""),
                    hypothesis=d.get("hypothesis", ""),
                    metrics=metrics,
                    duration_seconds=d.get("duration_seconds", 0.0),
                    engine_error=d.get("engine_error"),
                    ocr_intermediate=d.get("ocr_intermediate"),
                    pipeline_metadata=d.get("pipeline_metadata", {}),
                    confusion_matrix=d.get("confusion_matrix"),
                    char_scores=d.get("char_scores"),
                    taxonomy=d.get("taxonomy"),
                    structure=d.get("structure"),
                    image_quality=d.get("image_quality"),
                ))
    except Exception as e:
        logger.warning("Impossible de charger les résultats partiels '%s' : %s", path, e)
        results = []

    return path, results


def _save_partial_line(partial_path: Path, doc_result: DocumentResult) -> None:
    """Ajoute une entrée NDJSON au fichier de résultats partiels."""
    try:
        with partial_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(doc_result.as_dict(), ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("Impossible d'écrire dans le fichier partiel '%s' : %s", partial_path, e)


def _delete_partial(partial_path: Path) -> None:
    """Supprime le fichier de résultats partiels à la fin d'un moteur."""
    try:
        if partial_path.exists():
            partial_path.unlink()
    except Exception as e:
        logger.warning("Impossible de supprimer le fichier partiel '%s' : %s", partial_path, e)


# ---------------------------------------------------------------------------
# Benchmark principal
# ---------------------------------------------------------------------------

def run_benchmark(
    corpus: Corpus,
    engines: list[BaseOCREngine],
    output_json: Optional[str | Path] = None,
    show_progress: bool = True,
    progress_callback: Optional[callable] = None,
    char_exclude: Optional[frozenset] = None,
    max_workers: int = 4,
    timeout_seconds: float = 60.0,
    partial_dir: Optional[str | Path] = None,
) -> BenchmarkResult:
    """Exécute le benchmark d'un ou plusieurs moteurs/pipelines sur un corpus.

    Les pipelines OCR+LLM (``OCRLLMPipeline``) sont traités exactement comme
    les moteurs OCR classiques — ils implémentent la même interface
    ``BaseOCREngine`` et produisent les mêmes métriques CER/WER.

    Parallélisation
    ---------------
    * Moteurs CPU-bound (Tesseract, Pero OCR, Kraken) : ``ProcessPoolExecutor``
    * Moteurs IO-bound / API (Mistral, Google, Azure, LLMs) : ``ThreadPoolExecutor``

    Reprise sur interruption
    ------------------------
    Les résultats partiels sont sauvegardés document par document dans
    ``{partial_dir}/{corpus}_{engine}.partial.json``.  Si le benchmark est
    interrompu, la prochaine exécution repart automatiquement de là où elle
    s'est arrêtée.

    Parameters
    ----------
    corpus:
        Corpus à évaluer.
    engines:
        Liste d'adaptateurs moteurs ou de pipelines OCR+LLM.
    output_json:
        Chemin optionnel pour écrire le résultat JSON.
    show_progress:
        Affiche une barre de progression tqdm.
    progress_callback:
        Fonction ``(engine_name, doc_idx, doc_id) → None`` appelée après chaque
        document traité.  Une exception dans le callback est loguée en WARNING
        et n'interrompt pas le benchmark.
    char_exclude:
        Ensemble de caractères à exclure du calcul CER/WER.
    max_workers:
        Taille maximale des pools de threads/processus (défaut : 4).
        Peut être défini via le champ ``max_workers`` du YAML de configuration.
    timeout_seconds:
        Timeout par document en secondes (défaut : 60).  Un document dépassant
        ce délai est marqué comme erreur ``timeout`` et le benchmark continue.
    partial_dir:
        Répertoire pour les fichiers de reprise (défaut : répertoire temporaire
        système).

    Returns
    -------
    BenchmarkResult
    """
    engine_reports: list[EngineReport] = []

    for engine in engines:
        logger.info("Démarrage : %s", engine.name)

        # Reprise depuis résultats partiels d'une éventuelle exécution précédente
        partial_path, loaded_results = _load_partial(corpus.name, engine.name, partial_dir)
        loaded_doc_ids = {dr.doc_id for dr in loaded_results}
        if loaded_results:
            logger.info(
                "Reprise depuis résultats partiels : %d/%d documents déjà traités.",
                len(loaded_results), len(corpus),
            )

        docs_to_process = [doc for doc in corpus.documents if doc.doc_id not in loaded_doc_ids]
        document_results: list[DocumentResult] = list(loaded_results)

        # Sélection du type d'exécution selon la classe du moteur
        is_cpu_bound = engine.__class__.__name__ in _CPU_BOUND_ENGINE_CLASSES
        ExecutorClass = (
            concurrent.futures.ProcessPoolExecutor
            if is_cpu_bound
            else concurrent.futures.ThreadPoolExecutor
        )
        logger.info(
            "[%s] classe=%s, exécuteur=%s, docs à traiter=%d (reprise=%d).",
            engine.name,
            engine.__class__.__name__,
            "ProcessPoolExecutor" if is_cpu_bound else "ThreadPoolExecutor",
            len(docs_to_process),
            len(loaded_results),
        )

        pbar = tqdm(
            total=len(corpus.documents),
            initial=len(loaded_results),
            desc=f"[{engine.name}]",
            unit="doc",
            disable=not show_progress,
        )
        processed_count = len(loaded_results)

        executor = ExecutorClass(max_workers=max_workers)
        try:
            # Soumission de tous les documents au pool
            future_to_doc: dict = {}
            submitted_at: dict = {}

            for doc in docs_to_process:
                if is_cpu_bound:
                    engine_module = engine.__class__.__module__
                    engine_class_name = engine.__class__.__name__
                    char_exclude_tuple = tuple(char_exclude) if char_exclude else ()
                    future = executor.submit(
                        _cpu_doc_worker,
                        (engine_module, engine_class_name, engine.config,
                         doc.doc_id, str(doc.image_path), doc.ground_truth,
                         char_exclude_tuple),
                    )
                else:
                    future = executor.submit(_io_doc_worker, engine, doc, char_exclude)
                future_to_doc[future] = doc
                submitted_at[future] = time.monotonic()

            remaining = set(future_to_doc)

            while remaining:
                done, remaining = concurrent.futures.wait(
                    remaining,
                    timeout=0.5,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                for future in done:
                    doc = future_to_doc[future]
                    try:
                        doc_result = future.result()
                    except Exception as e:
                        logger.warning(
                            "[%s] doc %s : erreur inattendue : %s",
                            engine.name, doc.doc_id, e,
                        )
                        doc_result = _make_error_doc_result(doc, str(e))

                    document_results.append(doc_result)
                    _save_partial_line(partial_path, doc_result)
                    pbar.update(1)

                    if progress_callback is not None:
                        try:
                            progress_callback(engine.name, processed_count, doc.doc_id)
                        except Exception as e:
                            logger.warning("[progress_callback] fonctionnalité dégradée : %s", e)
                    processed_count += 1

                # Vérification des timeouts par document
                now = time.monotonic()
                timed_out = [
                    f for f in remaining
                    if now - submitted_at[f] > timeout_seconds
                ]
                for future in timed_out:
                    remaining.discard(future)
                    doc = future_to_doc[future]
                    future.cancel()
                    logger.warning(
                        "[%s] doc %s : timeout (%.0fs), document marqué en erreur.",
                        engine.name, doc.doc_id, timeout_seconds,
                    )
                    doc_result = _make_timeout_doc_result(doc, timeout_seconds)
                    document_results.append(doc_result)
                    _save_partial_line(partial_path, doc_result)
                    pbar.update(1)

                    if progress_callback is not None:
                        try:
                            progress_callback(engine.name, processed_count, doc.doc_id)
                        except Exception as e:
                            logger.warning(
                                "[progress_callback] fonctionnalité dégradée : %s", e
                            )
                    processed_count += 1

        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            pbar.close()

        # Réordonner selon l'ordre du corpus pour reproductibilité
        doc_order = {doc.doc_id: i for i, doc in enumerate(corpus.documents)}
        document_results.sort(key=lambda dr: doc_order.get(dr.doc_id, len(doc_order)))

        logger.info(
            "[%s] collecte terminée — %d/%d documents (dont %d chargés depuis reprise).",
            engine.name,
            len(document_results),
            len(corpus.documents),
            len(loaded_results),
        )
        if not document_results:
            logger.warning(
                "[%s] aucun DocumentResult collecté — le rapport affichera 0/0 documents. "
                "Vérifier que le moteur/pipeline a bien produit des résultats.",
                engine.name,
            )

        # Supprimer le fichier partiel — moteur terminé avec succès
        _delete_partial(partial_path)

        engine_version = engine._safe_version()
        pipeline_info = _build_pipeline_info(engine, document_results)

        agg_confusion = _aggregate_confusion(document_results)
        agg_char_scores = _aggregate_char_scores(document_results)
        agg_taxonomy = _aggregate_taxonomy(document_results)
        agg_structure = _aggregate_structure(document_results)
        agg_image_quality = _aggregate_image_quality(document_results)

        report = EngineReport(
            engine_name=engine.name,
            engine_version=engine_version,
            engine_config=engine.config,
            document_results=document_results,
            pipeline_info=pipeline_info,
            aggregated_confusion=agg_confusion,
            aggregated_char_scores=agg_char_scores,
            aggregated_taxonomy=agg_taxonomy,
            aggregated_structure=agg_structure,
            aggregated_image_quality=agg_image_quality,
        )
        engine_reports.append(report)
        logger.info(
            "%s terminé — CER moyen : %.2f%%",
            engine.name,
            (report.mean_cer or 0) * 100,
        )

    benchmark = BenchmarkResult(
        corpus_name=corpus.name,
        corpus_source=corpus.source_path,
        document_count=len(corpus),
        engine_reports=engine_reports,
    )

    if output_json:
        path = benchmark.to_json(output_json)
        logger.info("Résultats écrits dans : %s", path)

    return benchmark


def _build_pipeline_info(engine: BaseOCREngine, doc_results: list[DocumentResult]) -> dict:
    """Construit le dictionnaire pipeline_info pour un EngineReport."""
    first_with_meta = next(
        (dr for dr in doc_results if dr.pipeline_metadata), None
    )
    if first_with_meta is None:
        return {}

    meta = first_with_meta.pipeline_metadata
    info: dict = {
        "pipeline_mode": meta.get("pipeline_mode"),
        "prompt_file": meta.get("prompt_file"),
        "llm_model": meta.get("llm_model"),
        "llm_provider": meta.get("llm_provider"),
    }

    try:
        from picarones.pipelines.base import OCRLLMPipeline
        if isinstance(engine, OCRLLMPipeline):
            info["pipeline_steps"] = engine._build_steps_info()
            info["prompt_template"] = engine._prompt_template
    except ImportError:
        pass

    over_norm_results = [
        dr.pipeline_metadata.get("over_normalization")
        for dr in doc_results
        if dr.pipeline_metadata.get("over_normalization") is not None
    ]
    if over_norm_results:
        total_correct = sum(r["total_correct_ocr_words"] for r in over_norm_results)
        total_over = sum(r["over_normalized_count"] for r in over_norm_results)
        info["over_normalization"] = {
            "score": round(total_over / total_correct, 4) if total_correct > 0 else 0.0,
            "total_correct_ocr_words": total_correct,
            "over_normalized_count": total_over,
            "document_count": len(over_norm_results),
        }

    return info


# ---------------------------------------------------------------------------
# Helpers d'agrégation Sprint 5
# ---------------------------------------------------------------------------

def _aggregate_confusion(doc_results: list) -> Optional[dict]:
    """Agrège les matrices de confusion unicode sur tous les documents."""
    try:
        from picarones.core.confusion import aggregate_confusion_matrices, ConfusionMatrix
        matrices = [
            ConfusionMatrix(**dr.confusion_matrix)
            for dr in doc_results
            if dr.confusion_matrix is not None
        ]
        if not matrices:
            return None
        agg = aggregate_confusion_matrices(matrices)
        return agg.as_compact_dict(min_count=2)
    except Exception as e:
        logger.warning("[aggregate_confusion] fonctionnalité dégradée : %s", e)
        return None


def _aggregate_char_scores(doc_results: list) -> Optional[dict]:
    """Agrège les scores ligatures/diacritiques."""
    try:
        from picarones.core.char_scores import (
            aggregate_ligature_scores, aggregate_diacritic_scores,
            LigatureScore, DiacriticScore,
        )
        lig_scores = [
            LigatureScore(**dr.char_scores["ligature"])
            for dr in doc_results
            if dr.char_scores is not None
        ]
        diac_scores = [
            DiacriticScore(**dr.char_scores["diacritic"])
            for dr in doc_results
            if dr.char_scores is not None
        ]
        if not lig_scores:
            return None
        return {
            "ligature": aggregate_ligature_scores(lig_scores),
            "diacritic": aggregate_diacritic_scores(diac_scores),
        }
    except Exception as e:
        logger.warning("[aggregate_char_scores] fonctionnalité dégradée : %s", e)
        return None


def _aggregate_taxonomy(doc_results: list) -> Optional[dict]:
    """Agrège les classifications taxonomiques."""
    try:
        from picarones.core.taxonomy import aggregate_taxonomy, TaxonomyResult
        results = [
            TaxonomyResult.from_dict(dr.taxonomy)
            for dr in doc_results
            if dr.taxonomy is not None
        ]
        if not results:
            return None
        return aggregate_taxonomy(results)
    except Exception as e:
        logger.warning("[aggregate_taxonomy] fonctionnalité dégradée : %s", e)
        return None


def _aggregate_structure(doc_results: list) -> Optional[dict]:
    """Agrège les métriques structurelles."""
    try:
        from picarones.core.structure import aggregate_structure, StructureResult
        results = [
            StructureResult.from_dict(dr.structure)
            for dr in doc_results
            if dr.structure is not None
        ]
        if not results:
            return None
        return aggregate_structure(results)
    except Exception as e:
        logger.warning("[aggregate_structure] fonctionnalité dégradée : %s", e)
        return None


def _aggregate_image_quality(doc_results: list) -> Optional[dict]:
    """Agrège les métriques de qualité image."""
    try:
        from picarones.core.image_quality import aggregate_image_quality, ImageQualityResult
        results = [
            ImageQualityResult.from_dict(dr.image_quality)
            for dr in doc_results
            if dr.image_quality is not None
        ]
        if not results:
            return None
        return aggregate_image_quality(results)
    except Exception as e:
        logger.warning("[aggregate_image_quality] fonctionnalité dégradée : %s", e)
        return None
