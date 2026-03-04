"""Orchestrateur du benchmark : exécute les moteurs sur le corpus et agrège les résultats."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from picarones.core.corpus import Corpus
from picarones.core.metrics import compute_metrics
from picarones.core.results import BenchmarkResult, DocumentResult, EngineReport
from picarones.engines.base import BaseOCREngine

logger = logging.getLogger(__name__)


def run_benchmark(
    corpus: Corpus,
    engines: list[BaseOCREngine],
    output_json: Optional[str | Path] = None,
    show_progress: bool = True,
) -> BenchmarkResult:
    """Exécute le benchmark d'un ou plusieurs moteurs sur un corpus.

    Pour chaque moteur, chaque document est traité séquentiellement.
    Les sorties sont évaluées par rapport à la vérité terrain via
    les métriques CER et WER.

    Parameters
    ----------
    corpus:
        Corpus à évaluer (objet ``Corpus`` avec ses ``Document``).
    engines:
        Liste d'adaptateurs moteurs à comparer.
    output_json:
        Chemin optionnel pour écrire le résultat JSON. Si ``None``, pas
        d'écriture disque.
    show_progress:
        Affiche une barre de progression tqdm (défaut : True).

    Returns
    -------
    BenchmarkResult
        Objet contenant tous les résultats, agrégations et classement.
    """
    engine_reports: list[EngineReport] = []

    for engine in engines:
        logger.info("Démarrage moteur : %s", engine.name)
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
                # Moteur en erreur → métriques dégradées avec erreur tracée
                from picarones.core.metrics import MetricsResult

                metrics = MetricsResult(
                    cer=1.0, cer_nfc=1.0, cer_caseless=1.0,
                    wer=1.0, wer_normalized=1.0, mer=1.0, wil=1.0,
                    reference_length=len(doc.ground_truth),
                    hypothesis_length=0,
                    error=ocr_result.error,
                )

            document_results.append(
                DocumentResult(
                    doc_id=doc.doc_id,
                    image_path=str(doc.image_path),
                    ground_truth=doc.ground_truth,
                    hypothesis=ocr_result.text,
                    metrics=metrics,
                    duration_seconds=ocr_result.duration_seconds,
                    engine_error=ocr_result.error,
                )
            )

        engine_version = engine._safe_version()
        report = EngineReport(
            engine_name=engine.name,
            engine_version=engine_version,
            engine_config=engine.config,
            document_results=document_results,
        )
        engine_reports.append(report)
        logger.info(
            "Moteur %s terminé — CER moyen : %.2f%%",
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
