"""Helpers internes pour la conversion ``RunResult → BenchmarkResult``.

Module extrait du god-module ``benchmark_runner.py`` lors de la
Phase 6 (round 5) de l'audit code-quality (2026-05).

Surface publique : aucune.  Tous les symboles sont préfixés ``_``
et destinés à être consommés exclusivement par
``benchmark_runner.run_result_to_benchmark_result`` (et indirectement
par ``run_benchmark_via_service``).  Réexportés via alias dans
``benchmark_runner.py`` pour les tests qui patchent ces symboles.

Contenu :

- :class:`_OCRResultLike` — shim Pydantic minimal consommé par
  ``run_document_hooks`` (attributs ``success`` + ``token_confidences``).
- :func:`_extract_text_outputs` — ``(text_final, ocr_intermediate)``.
- :func:`_extract_first_error` — premier step en erreur.
- :func:`_build_pipeline_metadata` — métadonnées pipeline+over_normalization.
- :func:`_build_pipeline_info` — ``EngineReport.pipeline_info``.
- :func:`_engine_config_for_fingerprint` — fingerprint partial_store.
- :func:`_safe_engine_version` — ``engine.version()`` tolérant.
- :func:`_resolve_corpus_lang` — langue corpus pour ``readability``.
- :func:`_extract_token_confidences` — confidences pour ``calibration``.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from picarones.domain.artifacts import ArtifactType

if TYPE_CHECKING:
    from picarones.evaluation.corpus import Corpus

logger = logging.getLogger(__name__)


def _extract_text_outputs(pipeline_result: Any) -> tuple[str, str | None]:
    """Extrait ``(text_final, ocr_intermediate)`` du PipelineResult.

    - ``text_final`` : ``CORRECTED_TEXT`` prioritaire (post-correction
      LLM), sinon ``RAW_TEXT`` (OCR seul ou VLM zero-shot).
    - ``ocr_intermediate`` : ``RAW_TEXT`` quand un ``CORRECTED_TEXT``
      coexiste — ce qui correspond au texte OCR avant correction LLM.
      ``None`` si pas de pipeline composé.
    """
    corrected_text: str | None = None
    raw_text: str | None = None
    for art in pipeline_result.artifacts:
        if art.uri is None:
            continue
        if art.type == ArtifactType.CORRECTED_TEXT and corrected_text is None:
            try:
                corrected_text = Path(art.uri).read_text(encoding="utf-8")
            except OSError:
                corrected_text = ""
        elif art.type == ArtifactType.RAW_TEXT and raw_text is None:
            try:
                raw_text = Path(art.uri).read_text(encoding="utf-8")
            except OSError:
                raw_text = ""

    if corrected_text is not None:
        return corrected_text, raw_text
    if raw_text is not None:
        return raw_text, None
    return "", None


def _extract_first_error(pipeline_result: Any) -> str | None:
    """Retourne le ``error`` du premier step en échec, ou ``None``."""
    for step in pipeline_result.step_results:
        err = getattr(step, "error", None)
        if err:
            return str(err)
    return None


def _build_pipeline_metadata(
    *,
    engine: Any,
    ocr_intermediate: str | None,
    ground_truth: str = "",
    hypothesis: str = "",
) -> dict:
    """Reconstitue les ``pipeline_metadata`` pour un DocumentResult.

    Pour les pipelines composées OCR+LLM, calcule
    ``over_normalization`` (détection des cas où le LLM a sur-normalisé
    le texte par rapport à la GT) si ``ocr_intermediate`` est
    disponible.
    """
    if not getattr(engine, "is_pipeline", False):
        return {}
    metadata: dict = {
        "pipeline_mode": getattr(engine, "mode", None),
        "is_pipeline": True,
    }
    mode = metadata["pipeline_mode"]
    if mode is not None and hasattr(mode, "value"):
        metadata["pipeline_mode"] = mode.value
    llm_adapter = getattr(engine, "llm_adapter", None)
    if llm_adapter is not None:
        metadata["llm_model"] = llm_adapter.model
        metadata["llm_provider"] = llm_adapter.name
    if ocr_intermediate is not None:
        metadata["ocr_intermediate"] = ocr_intermediate
        try:
            from picarones.evaluation.metrics.over_normalization import (
                detect_over_normalization,
            )
            over_norm = detect_over_normalization(
                ground_truth=ground_truth,
                ocr_text=ocr_intermediate,
                llm_text=hypothesis,
            )
            metadata["over_normalization"] = over_norm.as_dict()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[over_normalization] fonctionnalité dégradée : %s",
                exc,
            )
    return metadata


def _build_pipeline_info(engine: Any) -> dict:
    """Reconstitue ``EngineReport.pipeline_info`` pour un engine pipeline."""
    if not getattr(engine, "is_pipeline", False):
        return {}
    info: dict = {
        "pipeline_steps": getattr(engine, "pipeline_steps_info", []),
        "prompt_template": getattr(engine, "prompt_template", ""),
    }
    llm_adapter = getattr(engine, "llm_adapter", None)
    if llm_adapter is not None:
        info["llm_model"] = llm_adapter.model
        info["llm_provider"] = llm_adapter.name
    mode = getattr(engine, "mode", None)
    if mode is not None:
        info["mode"] = mode.value if hasattr(mode, "value") else mode
    prompt_path = getattr(engine, "prompt_path", None)
    if prompt_path is not None:
        info["prompt_file"] = prompt_path
    return info


def _engine_config_for_fingerprint(engine: Any) -> dict:
    """Extrait une config sérialisable d'un engine pour le fingerprint.

    Utilisé par :func:`partial_store.compute_run_fingerprint` pour
    distinguer deux runs avec le même couple ``(corpus, engine.name)``
    mais des paramètres internes différents (psm/lang Tesseract,
    modèle LLM, prompt_template, mode pipeline, …).  Un changement
    non capturé par ce dict = potentiel faux résultat en reprise.
    """
    cfg: dict = {"engine_name": getattr(engine, "name", "")}

    if getattr(engine, "is_pipeline", False):
        mode = getattr(engine, "mode", None)
        cfg["mode"] = mode.value if hasattr(mode, "value") else mode
        prompt = getattr(engine, "prompt_template", None)
        if prompt is not None:
            # SHA-1 utilisé comme identifiant de cache uniquement
            # (fingerprint partial store) — ``usedforsecurity=False``
            # neutralise le faux positif bandit B324.
            cfg["prompt_sha1"] = hashlib.sha1(
                str(prompt).encode("utf-8"),
                usedforsecurity=False,
            ).hexdigest()[:12]
        llm = getattr(engine, "llm_adapter", None)
        if llm is not None:
            cfg["llm_model"] = getattr(llm, "model", "")
            cfg["llm_provider"] = getattr(llm, "name", "")
        ocr = getattr(engine, "ocr_adapter", None)
        if ocr is not None:
            cfg["ocr_name"] = getattr(ocr, "name", "")
    else:
        for attr in ("lang", "psm", "model", "model_id", "feature_type"):
            value = getattr(engine, attr, None)
            if value is not None:
                cfg[attr] = value
    return cfg


def _safe_engine_version(engine: Any) -> str:
    """Retourne ``engine.version()`` ou ``"unknown"`` en cas d'erreur."""
    version_attr = getattr(engine, "version", None)
    if version_attr is None:
        return "unknown"
    try:
        v = version_attr() if callable(version_attr) else version_attr
        return str(v) if v else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


@dataclass
class _OCRResultLike:
    """Shim minimal consommé par ``run_document_hooks``.

    Les hooks utilisent deux attributs : ``success`` (filtre
    ``requires_success``) et ``token_confidences`` (filtre
    ``requires_token_confidences`` + entrée du hook calibration).
    Le runner canonique manipule des ``PipelineResult`` et non des
    ``OCRResult`` legacy — ce shim fournit juste les deux attributs
    nécessaires sans tirer le modèle legacy.
    """

    success: bool
    token_confidences: list | None = None


def _resolve_corpus_lang(corpus: "Corpus") -> str:
    """Récupère la langue du corpus pour le hook ``readability``.

    Cherche dans ``corpus.metadata`` (clés ``lang`` ou ``language``)
    puis tombe sur ``"fr"`` (cohérent avec le défaut de
    ``compute_readability_metrics``).
    """
    metadata = getattr(corpus, "metadata", None) or {}
    for key in ("lang", "language"):
        value = metadata.get(key) if isinstance(metadata, dict) else None
        if value:
            return str(value)
    return "fr"


def _extract_token_confidences(pipeline_result: Any) -> list | None:
    """Récupère les confidences au token si un step OCR en a publié.

    Les adapters canoniques n'exposent pas encore systématiquement
    ces données ; le hook calibration retombera silencieusement via
    ``requires_token_confidences`` quand ``None``.
    """
    for step in getattr(pipeline_result, "step_results", ()) or ():
        confidences = getattr(step, "token_confidences", None)
        if confidences:
            return list(confidences)
    return None


__all__ = [
    "_OCRResultLike",
    "_build_pipeline_info",
    "_build_pipeline_metadata",
    "_engine_config_for_fingerprint",
    "_extract_first_error",
    "_extract_text_outputs",
    "_extract_token_confidences",
    "_resolve_corpus_lang",
    "_safe_engine_version",
]
