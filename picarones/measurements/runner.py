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
import threading
import time
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from picarones.core.corpus import Corpus
from picarones.measurements.metrics import MetricsResult, compute_metrics
from picarones.core.results import BenchmarkResult, DocumentResult, EngineReport
from picarones.engines.base import BaseOCREngine, EngineResult

logger = logging.getLogger(__name__)

# Lock pour la sérialisation des écritures de résultats partiels
_partial_write_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Workers de niveau module (requis pour ProcessPoolExecutor — picklables)
# ---------------------------------------------------------------------------

def _cpu_doc_worker(args: tuple) -> "DocumentResult":
    """Worker pour ProcessPoolExecutor (moteurs CPU-bound).

    Instancie le moteur dans le sous-processus, exécute l'OCR et calcule
    toutes les métriques.  Doit être une fonction de niveau module pour être
    sérialisable par ``pickle``.

    Le tuple ``args`` peut contenir, par compatibilité ascendante :
    - 7 éléments : legacy (Sprint 13)
    - 8 éléments : + ``corpus_lang`` (Sprint 87)
    - 9 éléments : + ``profile`` (chantier 2 post-Sprint 97)
    """
    if len(args) == 9:
        (engine_module, engine_class_name, engine_config, doc_id,
         image_path, ground_truth, char_exclude_chars, corpus_lang,
         profile) = args
    elif len(args) == 8:
        (engine_module, engine_class_name, engine_config, doc_id,
         image_path, ground_truth, char_exclude_chars, corpus_lang) = args
        profile = "standard"
    else:
        (engine_module, engine_class_name, engine_config, doc_id,
         image_path, ground_truth, char_exclude_chars) = args
        corpus_lang = "fr"
        profile = "standard"
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
        corpus_lang=corpus_lang,
        profile=profile,
    )


def _io_doc_worker(
    engine: BaseOCREngine,
    doc: object,
    char_exclude: Optional[frozenset],
    corpus_lang: str = "fr",
    profile: str = "standard",
) -> "DocumentResult":
    """Worker pour ThreadPoolExecutor (moteurs IO-bound / API).

    Exécute l'OCR et calcule les métriques dans un thread.  L'instance du
    moteur est partagée entre les threads — les adaptateurs HTTP sont
    généralement sans état mutable entre les appels.

    Si le document possède un texte OCR pré-calculé (corpus triplet) et que
    le moteur est un pipeline OCR+LLM, utilise ``run_with_ocr_text()`` pour
    court-circuiter l'étape OCR et tester directement la post-correction LLM.
    """
    doc_ocr_text = getattr(doc, "ocr_text", None)
    if doc_ocr_text is not None:
        # Corpus triplet — vérifier si le moteur supporte run_with_ocr_text
        run_with = getattr(engine, "run_with_ocr_text", None)
        if run_with is not None:
            ocr_result = run_with(doc.image_path, doc_ocr_text)  # type: ignore[attr-defined]
        else:
            # Moteur OCR classique — ignorer le texte OCR pré-calculé
            ocr_result = engine.run(doc.image_path)  # type: ignore[attr-defined]
    else:
        ocr_result = engine.run(doc.image_path)  # type: ignore[attr-defined]

    return _compute_document_result(
        doc_id=doc.doc_id,  # type: ignore[attr-defined]
        image_path=str(doc.image_path),  # type: ignore[attr-defined]
        ground_truth=doc.ground_truth,  # type: ignore[attr-defined]
        ocr_result=ocr_result,
        char_exclude=char_exclude,
        corpus_lang=corpus_lang,
        profile=profile,
    )


# ---------------------------------------------------------------------------
# Calcul documentaire centralisé
# ---------------------------------------------------------------------------


# Chantier 2 (post-Sprint 97) — la logique du helper calibration vit
# désormais dans :mod:`picarones.measurements.builtin_hooks`. Ce nom reste exposé
# ici pour la rétrocompat des tests Sprint 42 qui font
# ``from picarones.measurements.runner import _calibration_from_engine_result``.
def _calibration_from_engine_result(
    ground_truth: str,
    token_confidences: list,
) -> Optional[dict]:
    """Délégation vers :func:`picarones.measurements.builtin_hooks.calibration_from_engine_result`.

    Conservé pour la rétrocompat des tests existants ; toute évolution
    du calcul doit se faire dans ``builtin_hooks``.
    """
    from picarones.measurements.builtin_hooks import calibration_from_engine_result
    return calibration_from_engine_result(ground_truth, token_confidences)




def _compute_document_result(
    doc_id: str,
    image_path: str,
    ground_truth: str,
    ocr_result: EngineResult,
    char_exclude: Optional[frozenset],
    corpus_lang: str = "fr",
    profile: str = "standard",
) -> DocumentResult:
    """Calcule toutes les métriques pour un document et retourne un DocumentResult.

    Utilisable à la fois dans le processus principal (IO-bound) et dans les
    sous-processus créés par ProcessPoolExecutor (CPU-bound).
    Les imports lourds sont différés pour accélérer le démarrage des sous-processus.

    Chantier 2 (post-Sprint 97) — refonte
    ------------------------------------
    Les 11 ``try/except`` codés en dur (Sprints 5+10+39+42+61+86+87) sont
    désormais centralisés dans ``picarones.measurements.builtin_hooks`` et
    sélectionnés via ``run_document_hooks(profile)``.  Le profil
    ``"standard"`` (défaut) reproduit strictement le comportement
    pré-chantier-2.  Les profils ``"minimal"``, ``"philological"``,
    ``"diagnostics"``, ``"economics"``, ``"pipeline"``, ``"full"``
    permettent à l'utilisateur de moduler le coût de calcul.
    """
    import logging as _logging
    _logger = _logging.getLogger(__name__)

    # Eager-load des hooks natifs pour peupler le registre dans les
    # sous-processus du pool (le top-level ``import`` du runner ne le fait
    # pas pour ne pas pénaliser le démarrage des moteurs minimaux).
    import picarones.measurements.builtin_hooks  # noqa: F401
    from picarones.core.metric_hooks import run_document_hooks

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

    # Hooks document-level — chaque hook produit un attribut nommé du
    # ``DocumentResult``.  Les hooks invalides pour ce contexte (échec
    # OCR pour les hooks ``requires_success``, absence de
    # ``token_confidences`` pour ``calibration``) sont sautés
    # silencieusement.  Les exceptions levées par un hook sont
    # capturées et loggées en warning par ``run_document_hooks``.
    extras = run_document_hooks(
        profile,
        ground_truth=ground_truth,
        hypothesis=ocr_result.text,
        image_path=image_path,
        corpus_lang=corpus_lang,
        ocr_result=ocr_result,
    )

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
        confusion_matrix=extras.get("confusion_matrix"),
        char_scores=extras.get("char_scores"),
        taxonomy=extras.get("taxonomy"),
        structure=extras.get("structure"),
        image_quality=extras.get("image_quality"),
        line_metrics=extras.get("line_metrics"),
        hallucination_metrics=extras.get("hallucination_metrics"),
        calibration_metrics=extras.get("calibration_metrics"),
        philological_metrics=extras.get("philological_metrics"),
        searchability_metrics=extras.get("searchability_metrics"),
        numerical_sequence_metrics=extras.get("numerical_sequence_metrics"),
        readability_metrics=extras.get("readability_metrics"),
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
                    line_metrics=d.get("line_metrics"),
                    hallucination_metrics=d.get("hallucination_metrics"),
                ))
    except Exception as e:
        logger.warning("Impossible de charger les résultats partiels '%s' : %s", path, e)
        results = []

    return path, results


def _save_partial_line(partial_path: Path, doc_result: DocumentResult) -> None:
    """Ajoute une entrée NDJSON au fichier de résultats partiels (thread-safe)."""
    try:
        line = json.dumps(doc_result.as_dict(), ensure_ascii=False) + "\n"
        with _partial_write_lock:
            with partial_path.open("a", encoding="utf-8") as fh:
                fh.write(line)
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
    cancel_event: Optional[threading.Event] = None,
    entity_extractor: Optional[callable] = None,
    profile: str = "standard",
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
    from picarones.core.metric_hooks import (
        run_corpus_aggregators, validate_profile,
    )
    validate_profile(profile)

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
                         char_exclude_tuple, corpus_lang, profile),
                    )
                else:
                    future = executor.submit(
                        _io_doc_worker, engine, doc, char_exclude,
                        corpus_lang, profile,
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

        # Libérer la mémoire des analyses per-document après agrégation
        for dr in document_results:
            dr.compact()

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


# ---------------------------------------------------------------------------
# Helpers d'agrégation — délégations rétrocompat
# ---------------------------------------------------------------------------
# Chantier 2 (post-Sprint 97) : les implémentations vivent désormais dans
# :mod:`picarones.measurements.builtin_hooks` (single source of truth, exposé via
# le registre :mod:`picarones.core.metric_hooks`).  Les noms ci-dessous
# restent disponibles depuis ``picarones.measurements.runner`` pour la rétrocompat
# des tests Sprint 13 / 42 qui les importent directement.

def _aggregate_confusion(doc_results: list) -> Optional[dict]:
    """Délégation vers :func:`builtin_hooks._aggregate_confusion`."""
    from picarones.measurements.builtin_hooks import _aggregate_confusion as _impl
    return _impl(doc_results)


def _aggregate_char_scores(doc_results: list) -> Optional[dict]:
    """Délégation vers :func:`builtin_hooks._aggregate_char_scores`."""
    from picarones.measurements.builtin_hooks import _aggregate_char_scores as _impl
    return _impl(doc_results)


def _aggregate_taxonomy(doc_results: list) -> Optional[dict]:
    """Délégation vers :func:`builtin_hooks._aggregate_taxonomy`."""
    from picarones.measurements.builtin_hooks import _aggregate_taxonomy as _impl
    return _impl(doc_results)


def _aggregate_structure(doc_results: list) -> Optional[dict]:
    """Délégation vers :func:`builtin_hooks._aggregate_structure`."""
    from picarones.measurements.builtin_hooks import _aggregate_structure as _impl
    return _impl(doc_results)


def _aggregate_image_quality(doc_results: list) -> Optional[dict]:
    """Délégation vers :func:`builtin_hooks._aggregate_image_quality`."""
    from picarones.measurements.builtin_hooks import _aggregate_image_quality as _impl
    return _impl(doc_results)


def _aggregate_line_metrics(doc_results: list) -> Optional[dict]:
    """Délégation vers :func:`builtin_hooks._aggregate_line_metrics`."""
    from picarones.measurements.builtin_hooks import _aggregate_line_metrics as _impl
    return _impl(doc_results)


def _aggregate_hallucination(doc_results: list) -> Optional[dict]:
    """Délégation vers :func:`builtin_hooks._aggregate_hallucination`."""
    from picarones.measurements.builtin_hooks import _aggregate_hallucination as _impl
    return _impl(doc_results)


# ──────────────────────────────────────────────────────────────────────────
# Sprint 40 — extraction NER au post-process et agrégation
# ──────────────────────────────────────────────────────────────────────────


def _attach_ner_metrics(
    corpus: Corpus,
    doc_results: list,
    entity_extractor: callable,
) -> None:
    """Calcule et attache ``DocumentResult.ner_metrics`` pour chaque doc
    dont la GT possède un niveau ``ENTITIES`` (Sprint 32).

    L'extracteur est appelé sur l'hypothèse OCR ``dr.hypothesis``.
    Les erreurs sont dégradées en warnings (pas de propagation) afin
    de ne pas casser le benchmark si un document spécifique fait
    crasher le NER.
    """
    try:
        from picarones.core.corpus import GTLevel
        from picarones.measurements.ner import compute_ner_metrics
    except ImportError as exc:
        logger.warning("[ner.attach] imports indisponibles : %s", exc)
        return

    docs_by_id = {d.doc_id: d for d in corpus.documents}
    n_done = 0
    for dr in doc_results:
        if dr.engine_error is not None or not dr.hypothesis:
            continue
        doc = docs_by_id.get(dr.doc_id)
        if doc is None or not doc.has_gt(GTLevel.ENTITIES):
            continue
        try:
            gt_payload = doc.get_gt(GTLevel.ENTITIES)
            gt_entities = list(gt_payload.entities) if gt_payload else []
            hyp_entities = entity_extractor(dr.hypothesis) or []
            dr.ner_metrics = compute_ner_metrics(gt_entities, hyp_entities)
            n_done += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[ner.attach] %s : extraction/comparaison NER dégradée : %s",
                dr.doc_id, exc,
            )

    if n_done > 0:
        logger.info("[ner] %d documents évalués pour NER.", n_done)


def _aggregate_calibration(doc_results: list) -> Optional[dict]:
    """Délégation vers :func:`builtin_hooks._aggregate_calibration`.

    Conservé pour la rétrocompat du test ``test_sprint42_calibration_runner``
    qui importe directement depuis ``picarones.measurements.runner``. La logique
    réelle vit dans :mod:`picarones.measurements.builtin_hooks` (chantier 2
    post-Sprint 97).
    """
    from picarones.measurements.builtin_hooks import _aggregate_calibration as _impl
    return _impl(doc_results)


def _aggregate_ner(doc_results: list) -> Optional[dict]:
    """Agrège les métriques NER au niveau du moteur.

    Recalcule precision/recall/F1 *micro* à partir des sommes globales
    de TP/FP/FN, plus le détail par catégorie, plus les compteurs
    totaux d'hallucinations et d'entités manquées.
    """
    relevant = [dr for dr in doc_results if dr.ner_metrics is not None]
    if not relevant:
        return None

    total_tp = 0
    total_fp = 0
    total_fn = 0
    cat_tp: dict[str, int] = {}
    cat_fp: dict[str, int] = {}
    cat_fn: dict[str, int] = {}
    total_hallucinated = 0
    total_missed = 0
    iou_threshold = 0.5

    for dr in relevant:
        m = dr.ner_metrics
        total_tp += int(m.get("true_positives", 0))
        total_fp += int(m.get("false_positives", 0))
        total_fn += int(m.get("false_negatives", 0))
        total_hallucinated += len(m.get("hallucinated_entities", []) or [])
        total_missed += len(m.get("missed_entities", []) or [])
        iou_threshold = float(m.get("iou_threshold", iou_threshold))
        for cat, stats in (m.get("per_category") or {}).items():
            cat_tp[cat] = cat_tp.get(cat, 0)
            cat_fp[cat] = cat_fp.get(cat, 0)
            cat_fn[cat] = cat_fn.get(cat, 0)
            # Reconstitue les sommes par catégorie via support et P/R
            support = int(stats.get("support", 0))
            recall = float(stats.get("recall", 0.0))
            precision = float(stats.get("precision", 0.0))
            tp_cat = round(support * recall) if support > 0 else 0
            fn_cat = max(0, support - tp_cat)
            fp_cat = (
                round(tp_cat * (1 - precision) / precision)
                if precision > 0 else 0
            )
            cat_tp[cat] += tp_cat
            cat_fp[cat] += fp_cat
            cat_fn[cat] += fn_cat

    def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {"precision": p, "recall": r, "f1": f1, "support": tp + fn}

    return {
        "global": _prf(total_tp, total_fp, total_fn),
        "per_category": {
            cat: _prf(cat_tp[cat], cat_fp[cat], cat_fn[cat])
            for cat in sorted(set(cat_tp) | set(cat_fp) | set(cat_fn))
        },
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "hallucinated_total": total_hallucinated,
        "missed_total": total_missed,
        "doc_count": len(relevant),
        "iou_threshold": iou_threshold,
    }
