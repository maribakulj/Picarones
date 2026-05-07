"""Construction d'un :class:`DocumentResult` à partir d'un OCR.

Centralise le calcul de toutes les métriques attachées à un document
unique : métriques principales (CER/WER/MER/WIL via jiwer), hooks
optionnels (calibration, taxonomy, philological, etc. — exécutés via
``run_document_hooks(profile)``), et meta pipeline OCR+LLM.

Aussi : helpers pour construire les ``DocumentResult`` synthétiques
en cas de timeout ou d'erreur d'engine (``_make_timeout_doc_result``,
``_make_error_doc_result``).
"""

from __future__ import annotations

from typing import Optional

from picarones.evaluation.benchmark_result import DocumentResult
from picarones.adapters.legacy_engines.base import EngineResult
from picarones.evaluation.metric_result import MetricsResult
from picarones.measurements.metrics import compute_metrics


def _calibration_from_engine_result(
    ground_truth: str,
    token_confidences: list,
) -> Optional[dict]:
    """Délégation vers
    :func:`picarones.measurements.builtin_hooks.calibration_from_engine_result`.

    Conservé pour la rétrocompat des tests Sprint 42 qui font
    ``from picarones.measurements.runner import _calibration_from_engine_result``.
    Toute évolution du calcul doit se faire dans ``builtin_hooks``.
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
    normalization_profile: Optional[object] = None,
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
    from picarones.evaluation.metric_hooks import run_document_hooks

    if ocr_result.success:
        # Sprint A14-S1 — A.I.0 P0 : propagation du profil de
        # normalisation depuis le runner.  ``normalization_profile``
        # est un ``NormalizationProfile`` résolu en main process par
        # ``run_benchmark`` (cf. orchestration.py).
        metrics = compute_metrics(
            ground_truth, ocr_result.text,
            normalization_profile=normalization_profile,  # type: ignore[arg-type]
            char_exclude=char_exclude,
        )
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
    """DocumentResult synthétique pour une erreur lors d'un appel engine."""
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


__all__ = [
    "_calibration_from_engine_result",
    "_compute_document_result",
    "_make_error_doc_result",
    "_make_timeout_doc_result",
]
