"""Orchestrateur du benchmark : exécute les moteurs/pipelines sur le corpus et agrège les résultats."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from picarones.core.corpus import Corpus
from picarones.core.metrics import MetricsResult, compute_metrics
from picarones.core.results import BenchmarkResult, DocumentResult, EngineReport
from picarones.engines.base import BaseOCREngine

logger = logging.getLogger(__name__)


def run_benchmark(
    corpus: Corpus,
    engines: list[BaseOCREngine],
    output_json: Optional[str | Path] = None,
    show_progress: bool = True,
) -> BenchmarkResult:
    """Exécute le benchmark d'un ou plusieurs moteurs/pipelines sur un corpus.

    Les pipelines OCR+LLM (``OCRLLMPipeline``) sont traités exactement comme
    les moteurs OCR classiques — ils implémentent la même interface
    ``BaseOCREngine`` et produisent les mêmes métriques CER/WER.

    En supplément, pour les pipelines :
    - La sortie OCR intermédiaire est conservée dans ``DocumentResult.ocr_intermediate``
    - La sur-normalisation LLM (classe 10) est calculée et stockée dans
      ``DocumentResult.pipeline_metadata["over_normalization"]``
    - Les stats agrégées de sur-normalisation figurent dans ``EngineReport.pipeline_info``

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

    Returns
    -------
    BenchmarkResult
    """
    engine_reports: list[EngineReport] = []

    for engine in engines:
        logger.info("Démarrage concurrent : %s", engine.name)
        document_results: list[DocumentResult] = []

        iterator = tqdm(
            corpus.documents,
            desc=f"[{engine.name}]",
            unit="doc",
            disable=not show_progress,
        )

        for doc in iterator:
            ocr_result = engine.run(doc.image_path)

            if ocr_result.success:
                metrics = compute_metrics(doc.ground_truth, ocr_result.text)
            else:
                metrics = MetricsResult(
                    cer=1.0, cer_nfc=1.0, cer_caseless=1.0,
                    wer=1.0, wer_normalized=1.0, mer=1.0, wil=1.0,
                    reference_length=len(doc.ground_truth),
                    hypothesis_length=0,
                    error=ocr_result.error,
                )

            # Extraction des champs pipeline depuis les métadonnées EngineResult
            ocr_intermediate = ocr_result.metadata.get("ocr_intermediate")
            pipeline_meta: dict = {}

            if ocr_result.metadata.get("is_pipeline"):
                pipeline_meta = {
                    "pipeline_mode": ocr_result.metadata.get("pipeline_mode"),
                    "prompt_file": ocr_result.metadata.get("prompt_file"),
                    "llm_model": ocr_result.metadata.get("llm_model"),
                    "llm_provider": ocr_result.metadata.get("llm_provider"),
                }
                # Calcul de la sur-normalisation (classe 10) si OCR intermédiaire disponible
                if ocr_intermediate is not None and ocr_result.success:
                    from picarones.pipelines.over_normalization import detect_over_normalization
                    over_norm = detect_over_normalization(
                        ground_truth=doc.ground_truth,
                        ocr_text=ocr_intermediate,
                        llm_text=ocr_result.text,
                    )
                    pipeline_meta["over_normalization"] = over_norm.as_dict()

            document_results.append(
                DocumentResult(
                    doc_id=doc.doc_id,
                    image_path=str(doc.image_path),
                    ground_truth=doc.ground_truth,
                    hypothesis=ocr_result.text,
                    metrics=metrics,
                    duration_seconds=ocr_result.duration_seconds,
                    engine_error=ocr_result.error,
                    ocr_intermediate=ocr_intermediate,
                    pipeline_metadata=pipeline_meta,
                )
            )

        engine_version = engine._safe_version()
        pipeline_info = _build_pipeline_info(engine, document_results)

        report = EngineReport(
            engine_name=engine.name,
            engine_version=engine_version,
            engine_config=engine.config,
            document_results=document_results,
            pipeline_info=pipeline_info,
        )
        engine_reports.append(report)
        logger.info(
            "Concurrent %s terminé — CER moyen : %.2f%%",
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

    # Récupérer les steps depuis le moteur si c'est un OCRLLMPipeline
    try:
        from picarones.pipelines.base import OCRLLMPipeline
        if isinstance(engine, OCRLLMPipeline):
            info["pipeline_steps"] = engine._build_steps_info()
            info["prompt_template"] = engine._prompt_template
    except ImportError:
        pass

    # Agréger les stats de sur-normalisation sur tous les documents
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
