"""Orchestrateur principal du benchmark.

Contient :func:`run_benchmark` et son helper :func:`_build_pipeline_info`.

Le runner exécute chaque moteur de la liste sur le corpus complet :

- Pour les moteurs CPU-bound (``execution_mode == "cpu"`` :
  Tesseract, Pero OCR, Kraken), utilise un ``ProcessPoolExecutor``
  et délègue aux workers picklables de :mod:`workers`.
- Pour les moteurs IO-bound (Mistral, Google Vision, Azure, LLMs),
  utilise un ``ThreadPoolExecutor``.

Les résultats partiels (NDJSON par moteur) sont gérés par
:mod:`partial` ; le calcul d'un :class:`DocumentResult` individuel
par :mod:`document` ; l'agrégation finale par les hooks délégués à
:mod:`builtin_hooks` (chantier 2 post-Sprint 97).
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from picarones.evaluation.corpus import Corpus
from picarones.evaluation.benchmark_result import BenchmarkResult, DocumentResult, EngineReport
from picarones.evaluation.engines.base import BaseOCREngine
from picarones.measurements.runner.document import (
    _make_error_doc_result,
    _make_timeout_doc_result,
)
from picarones.measurements.runner.ner_attach import (
    _aggregate_ner,
    _attach_ner_metrics,
)
from picarones.measurements.runner.partial import (
    _delete_partial,
    _load_partial,
    _save_partial_line,
)
from picarones.measurements.runner.workers import (
    _cpu_doc_worker,
    _io_doc_worker,
)

logger = logging.getLogger(__name__)


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
    cancel_event: Optional[threading.Event] = None,
    entity_extractor: Optional[callable] = None,
    profile: str = "standard",
    normalization_profile: Optional[str] = None,
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
    cancel_event:
        ``threading.Event`` optionnel.  Si défini et signalé (``set()``),
        le benchmark s'interrompt proprement dès que possible et retourne
        les résultats partiels collectés jusque-là.
    profile:
        Profil de calcul des métriques (chantier 2 post-Sprint 97).
        Valeurs : ``"minimal"`` (CER/WER seuls), ``"standard"`` (défaut,
        comportement historique avec les 12 hooks), ``"philological"``,
        ``"diagnostics"``, ``"economics"``, ``"pipeline"``, ``"full"``.
        Le profil ``"standard"`` est strictement rétrocompatible avec
        le runner pré-chantier-2.
    normalization_profile:
        Identifiant d'un profil de normalisation diplomatique
        (cf. ``measurements.normalization.NORMALIZATION_PROFILES``).
        Sprint A14-S1 — A.I.0 P0 : auparavant l'API web exposait ce
        paramètre mais il était silencieusement perdu avant
        d'atteindre ``compute_metrics``, ce qui rendait
        scientifiquement faux tout benchmark lancé via la web app.
        Désormais propagé end-to-end : web → run_benchmark → workers
        → compute_metrics.  ``None`` = profil par défaut (medieval_french).

    Returns
    -------
    BenchmarkResult
    """
    # Validation du profil dès l'entrée pour échouer rapidement sur
    # une faute de frappe utilisateur, avant de soumettre des futures
    # aux pools.  Eager-load des hooks natifs pour peupler le registre
    # dans le main process (les sous-processus du pool feront leur
    # propre import dans ``_compute_document_result``).
    import picarones.measurements.builtin_hooks  # noqa: F401
    from picarones.evaluation.metric_hooks import (
        run_corpus_aggregators, validate_profile,
    )
    validate_profile(profile)

    # Sprint A14-S1 — résolution one-shot du profil de normalisation.
    # On le fait ici (main process) pour échouer rapidement sur un ID
    # invalide avant de soumettre des futures aux pools, et pour
    # éviter de re-résoudre N fois côté workers.
    norm_profile_obj = None
    if normalization_profile is not None:
        from picarones.measurements.normalization import get_builtin_profile
        norm_profile_obj = get_builtin_profile(normalization_profile)

    def _is_cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()
    engine_reports: list[EngineReport] = []
    # Sprint 36 — collecte des hypothèses brutes par moteur avant
    # ``compact()`` pour pouvoir calculer la divergence taxonomique et
    # la complémentarité (oracle) en fin de benchmark.
    per_engine_outputs: dict[str, dict[str, str]] = {}
    ground_truths_by_doc: dict[str, str] = {}
    # Sprint 45 — A.III stratification : capture du ``script_type`` par
    # document avant ``compact()`` (qui efface ``image_quality``).
    doc_strata: dict[str, str] = {}

    # Sprint 87 — langue du corpus pour le delta Flesch (A.II.2).
    # Lecture depuis corpus.metadata, fallback "fr".
    corpus_lang: str = (corpus.metadata or {}).get("language", "fr")
    if corpus_lang not in ("fr", "en"):
        # Sprint 52 ne supporte que fr/en — fallback "fr" en warning.
        logger.warning(
            "[readability] langue '%s' non supportée, fallback 'fr'.",
            corpus_lang,
        )
        corpus_lang = "fr"

    for engine in engines:
        if _is_cancelled():
            logger.info("Benchmark annulé avant le moteur '%s'.", engine.name)
            break
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
        if loaded_doc_ids:
            logger.info(
                "[%s] %d doc(s) ignorés (résultats partiels existants) — "
                "supprimer le fichier partiel '%s' pour forcer le recalcul.",
                engine.name, len(loaded_doc_ids), partial_path,
            )
        document_results: list[DocumentResult] = list(loaded_results)

        # Sélection du type d'exécution selon execution_mode du moteur
        is_cpu_bound = getattr(engine, "execution_mode", "io") == "cpu"
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
                if _is_cancelled():
                    logger.info("[%s] annulation — arrêt de la soumission.", engine.name)
                    break
                if is_cpu_bound:
                    engine_module = engine.__class__.__module__
                    engine_class_name = engine.__class__.__name__
                    char_exclude_tuple = tuple(char_exclude) if char_exclude else ()
                    future = executor.submit(
                        _cpu_doc_worker,
                        (engine_module, engine_class_name, engine.config,
                         doc.doc_id, str(doc.image_path), doc.ground_truth,
                         char_exclude_tuple, corpus_lang, profile,
                         norm_profile_obj),
                    )
                else:
                    future = executor.submit(
                        _io_doc_worker, engine, doc, char_exclude,
                        corpus_lang, profile, norm_profile_obj,
                    )
                future_to_doc[future] = doc
                submitted_at[future] = time.monotonic()

            remaining = set(future_to_doc)

            while remaining:
                if _is_cancelled():
                    logger.info("[%s] annulation — annulation des futures restantes.", engine.name)
                    for f in remaining:
                        f.cancel()
                    break

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

        if _is_cancelled():
            logger.info(
                "[%s] annulé — %d documents traités sur %d.",
                engine.name, len(document_results) - len(loaded_results),
                len(docs_to_process),
            )
            # Conserver le fichier partiel pour reprise ultérieure
            break

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

        # Chantier 2 (post-Sprint 97) — agrégation déléguée au registre.
        # Les 12 appels manuels aux fonctions ``_aggregate_*`` sont
        # remplacés par un seul appel qui itère sur les agrégateurs
        # actifs du profil. Le profil ``"standard"`` (défaut) reproduit
        # exactement le comportement pré-chantier-2.
        aggregated = run_corpus_aggregators(profile, document_results)

        report = EngineReport(
            engine_name=engine.name,
            engine_version=engine_version,
            engine_config=engine.config,
            document_results=document_results,
            pipeline_info=pipeline_info,
            aggregated_confusion=aggregated.get("aggregated_confusion"),
            aggregated_char_scores=aggregated.get("aggregated_char_scores"),
            aggregated_taxonomy=aggregated.get("aggregated_taxonomy"),
            aggregated_structure=aggregated.get("aggregated_structure"),
            aggregated_image_quality=aggregated.get("aggregated_image_quality"),
            aggregated_line_metrics=aggregated.get("aggregated_line_metrics"),
            aggregated_hallucination=aggregated.get("aggregated_hallucination"),
            aggregated_calibration=aggregated.get("aggregated_calibration"),
            aggregated_philological=aggregated.get("aggregated_philological"),
            aggregated_searchability=aggregated.get("aggregated_searchability"),
            aggregated_numerical_sequences=aggregated.get("aggregated_numerical_sequences"),
            aggregated_readability=aggregated.get("aggregated_readability"),
        )
        engine_reports.append(report)
        logger.info(
            "%s terminé — CER moyen : %.2f%%",
            engine.name,
            (report.mean_cer or 0) * 100,
        )

        # Sprint 36 — capture des hypothèses brutes pour le calcul
        # inter-moteurs (effectué après la boucle, avant la sérialisation).
        # On clone les chaînes pour ne pas dépendre de la durée de vie des
        # DocumentResult après ``compact()``.
        per_engine_outputs[engine.name] = {
            dr.doc_id: dr.hypothesis for dr in document_results
            if dr.engine_error is None
        }
        for dr in document_results:
            if dr.doc_id not in ground_truths_by_doc and dr.ground_truth:
                ground_truths_by_doc[dr.doc_id] = dr.ground_truth
            # Sprint 45 — capture script_type avant compact()
            if dr.doc_id not in doc_strata and dr.image_quality:
                st = dr.image_quality.get("script_type")
                if st:
                    doc_strata[dr.doc_id] = str(st)

        # Sprint 40 — calcul des métriques NER si :
        #   1. l'utilisateur a fourni un EntityExtractor au runner ;
        #   2. ET le document a un niveau de GT ENTITIES (Sprint 32).
        # Fait dans le main process (pas dans les sous-processus du pool)
        # pour éviter de pickler l'extracteur (spaCy + modèle).
        if entity_extractor is not None:
            _attach_ner_metrics(corpus, document_results, entity_extractor)
            agg_ner = _aggregate_ner(document_results)
            report.aggregated_ner = agg_ner

        # Sprint A14-S1 — A.I.0 P0 : la compaction inconditionnelle qui
        # vivait ici amputait silencieusement le JSON exporté (et donc
        # le rapport HTML qui le consomme) en supprimant 13 dicts
        # d'analyse per-document et en tronquant les textes à 200 chars.
        # ``DocumentResult.compact()`` est désormais opt-in (paramètres
        # ``text_limit`` et ``drop_analyses``) ; le runner ne compacte
        # plus par défaut afin que ``output_json`` contienne réellement
        # toutes les analyses détaillées promises par le README.
        # Un caller qui veut un JSON léger peut appeler
        # ``dr.compact(text_limit=200, drop_analyses=True)`` lui-même
        # après ``run_benchmark`` et avant la sérialisation finale.

    # Sprint 36 — analyse inter-moteurs (divergence taxonomique +
    # complémentarité / oracle).  N'est calculée qu'à partir de 2
    # moteurs ; en deçà l'analyse n'a pas de sens.
    inter_engine_payload: Optional[dict] = None
    if len(engine_reports) >= 2:
        try:
            from picarones.measurements.inter_engine import compute_inter_engine_analysis

            taxonomy_distros = {
                report.engine_name: (
                    report.aggregated_taxonomy.get("class_distribution", {})
                    if report.aggregated_taxonomy
                    else {}
                )
                for report in engine_reports
            }
            # Élimine les moteurs sans distribution taxonomique pour ne pas
            # polluer la matrice.
            taxonomy_distros = {
                name: dist for name, dist in taxonomy_distros.items() if dist
            }
            inter_engine_payload = compute_inter_engine_analysis(
                per_engine_outputs=per_engine_outputs,
                ground_truths=ground_truths_by_doc,
                taxonomy_distributions=taxonomy_distros or None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[runner] analyse inter-moteurs dégradée : %s — section omise du rapport",
                exc,
            )

    benchmark = BenchmarkResult(
        corpus_name=corpus.name,
        corpus_source=corpus.source_path,
        document_count=len(corpus),
        engine_reports=engine_reports,
        inter_engine_analysis=inter_engine_payload,
        doc_strata=dict(doc_strata) if doc_strata else None,
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


__all__ = ["_build_pipeline_info", "run_benchmark"]
