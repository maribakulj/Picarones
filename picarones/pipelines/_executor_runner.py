"""Helper d'exécution mono-document via ``PipelineExecutor`` (Sprint B).

Sprint B du plan v2.0 — pont entre l'API mono-document
``OCRLLMPipeline.run(image_path) -> EngineResult`` (legacy) et le
``PipelineExecutor`` du rewrite.

Ce helper isole toute la plomberie nécessaire pour exécuter une
``PipelineSpec`` sur un seul document avec :

- création d'un ``tempdir`` éphémère comme ``workspace_uri`` ;
- adapter resolver minimal qui mappe les noms de la spec aux
  instances OCR/LLM portées par le ``OCRLLMPipeline`` ;
- conversion du ``PipelineResult`` en ``EngineResult`` legacy ;
- préservation des warnings comportementaux du legacy
  (texte OCR vide, texte LLM vide, erreur pipeline globale).

Trace de retrait
----------------
Ce module est temporaire (Sprint B-D du plan v2.0).  Il sera
supprimé en Sprint C quand les 3 callers (``web/benchmark_utils``,
``measurements/runner/orchestration``, ``fixtures``) consommeront
des ``PipelineSpec`` directement plutôt que des ``OCRLLMPipeline``.
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from picarones.adapters.legacy_engines._step_executor import (
    LegacyOCREngineExecutor,
)
from picarones.adapters.legacy_engines.base import EngineResult
from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.domain.documents import DocumentRef
from picarones.domain.pipeline_spec import (
    INITIAL_STEP_ID,
    PipelineSpec,
    PipelineStep,
)
from picarones.pipeline import (
    PipelineExecutor,
    RunContext,
    make_ocr_llm_pipeline_spec,
)

if TYPE_CHECKING:
    from picarones.pipelines.base import OCRLLMPipeline


logger = logging.getLogger("picarones.pipelines.base")


def run_pipeline_via_executor(
    pipeline: "OCRLLMPipeline",
    image_path: Path,
    *,
    ocr_text: Optional[str] = None,
) -> EngineResult:
    """Exécute une chaîne OCR+LLM via ``PipelineExecutor``.

    Cas 1 — ``ocr_text=None`` (run() classique) :
        Modes ``text_only`` / ``text_and_image`` / ``zero_shot``.
        La spec a un step OCR (sauf zero-shot) + un step LLM.

    Cas 2 — ``ocr_text`` fourni (run_with_ocr_text, corpus triplet) :
        Le texte OCR est pré-calculé.  La spec n'a qu'un step LLM
        qui consomme ``RAW_TEXT`` directement depuis les inputs
        initiaux (pas d'OCR engine appelé).

    Parameters
    ----------
    pipeline:
        L'instance ``OCRLLMPipeline`` qui porte ``ocr_engine``,
        ``llm_adapter``, ``mode`` et ``_prompt_template``.
    image_path:
        Chemin de l'image à transcrire.
    ocr_text:
        Si fourni, mode "post-correction" — le LLM reçoit ce texte
        directement, sans appel OCR.

    Returns
    -------
    EngineResult
        Format legacy compatible avec ``BaseOCREngine.run()``.  Les
        métadonnées portent ``pipeline_mode``, ``pipeline_steps``,
        ``llm_model``, ``llm_provider``, ``ocr_intermediate``,
        ``is_pipeline=True`` etc.
    """
    start = time.perf_counter()

    # Le LLM peut être un BaseLLMAdapter ou un BaseVLMAdapter — les
    # deux exposent .name et .model.  On compose un identifiant
    # ``provider:model`` stable pour le adapter resolver.
    llm_name = f"{pipeline.llm_adapter.name}:{pipeline.llm_adapter.model}"

    with tempfile.TemporaryDirectory(prefix="picarones_pipe_") as ws:
        workspace = Path(ws)

        # ── Construit la spec adaptée au cas (avec ou sans OCR)
        if ocr_text is None:
            spec, ocr_step_executor = _build_spec_for_run(
                pipeline=pipeline,
                llm_name=llm_name,
            )
            initial_inputs = {
                ArtifactType.IMAGE: _make_image_artifact(image_path, "doc"),
            }
        else:
            spec, ocr_step_executor = _build_spec_for_run_with_ocr_text(
                pipeline=pipeline,
                llm_name=llm_name,
            )
            # Écrire le texte OCR pré-fourni dans le workspace pour
            # qu'il soit accessible via Artifact.uri.
            text_path = workspace / "ocr_input.txt"
            text_path.write_text(ocr_text, encoding="utf-8")
            initial_inputs = {
                ArtifactType.IMAGE: _make_image_artifact(image_path, "doc"),
                ArtifactType.RAW_TEXT: Artifact(
                    id="doc:initial:raw_text",
                    document_id="doc",
                    type=ArtifactType.RAW_TEXT,
                    uri=str(text_path),
                ),
            }

        # ── Adapter resolver — mappe les noms de la spec aux instances
        def resolver(name: str) -> Any:
            if ocr_step_executor is not None and (
                pipeline.ocr_engine is not None
                and name == pipeline.ocr_engine.name
            ):
                return ocr_step_executor
            if name == llm_name:
                return pipeline.llm_adapter
            raise KeyError(f"adapter inconnu pour la spec : {name!r}")

        document = DocumentRef(id="doc", image_uri=str(image_path))
        context = RunContext(
            document_id="doc",
            code_version=_safe_code_version(),
            pipeline_name=spec.name,
            workspace_uri=str(workspace),
        )

        executor = PipelineExecutor(adapter_resolver=resolver)
        try:
            result = executor.run(spec, document, initial_inputs, context)
            error: Optional[str] = None
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%s] erreur pipeline pour '%s' : %s",
                pipeline.name, image_path.name, exc,
            )
            return _engine_result_failure(
                pipeline=pipeline,
                image_path=image_path,
                error=str(exc),
                duration=time.perf_counter() - start,
                ocr_text=ocr_text,
            )

        # ── Récupère le texte final depuis le bag d'artifacts
        text, ocr_intermediate = _extract_outputs(
            result=result,
            mode=pipeline.mode.value,
            ocr_text=ocr_text,
        )

        # ── Préserve les warnings comportementaux du legacy
        if ocr_text is None and pipeline.mode.value != "zero_shot":
            if ocr_intermediate is not None and not ocr_intermediate.strip():
                logger.warning(
                    "[%s] texte OCR vide pour '%s' — le LLM recevra "
                    "{ocr_output} vide.",
                    pipeline.name, image_path.name,
                )
        if not text or not text.strip():
            logger.warning(
                "[%s] le LLM ('%s') a retourné un texte vide pour '%s'. "
                "CER sera calculé à 1.0 (100%%). "
                "Vérifier : (1) le prompt contient-il {ocr_output} ? "
                "(2) le modèle supporte-t-il ce mode d'appel ? "
                "(3) la réponse n'est-elle pas tronquée (max_tokens) ?",
                pipeline.name, pipeline.llm_adapter.model, image_path.name,
            )

        # ── Si le pipeline a échoué (un step en error), on traduit
        # l'erreur du premier step en échec en EngineResult.error.
        if not result.succeeded:
            failed_step = next(
                (s for s in result.step_results if s.error is not None),
                None,
            )
            error = failed_step.error if failed_step is not None else "pipeline failed"

        duration = time.perf_counter() - start

        metadata = _build_metadata(
            pipeline=pipeline,
            ocr_intermediate=ocr_intermediate,
            ocr_source="corpus" if ocr_text is not None else None,
        )

        return EngineResult(
            engine_name=pipeline.name,
            image_path=str(image_path),
            text=text if text else "",
            duration_seconds=round(duration, 4),
            error=error,
            metadata=metadata,
        )


# ──────────────────────────────────────────────────────────────────────
# Helpers privés
# ──────────────────────────────────────────────────────────────────────


def _build_spec_for_run(
    pipeline: "OCRLLMPipeline",
    llm_name: str,
) -> tuple[PipelineSpec, Optional[LegacyOCREngineExecutor]]:
    """Spec pour ``run()`` — mode text_only / text_and_image / zero_shot."""
    mode = pipeline.mode.value
    llm_params = {"prompt_template": pipeline._prompt_template}

    if mode == "zero_shot":
        spec = make_ocr_llm_pipeline_spec(
            mode="zero_shot",
            llm_adapter_name=llm_name,
            llm_params=llm_params,
        )
        return spec, None

    if pipeline.ocr_engine is None:
        raise ValueError(
            f"ocr_engine est requis pour le mode {mode!r} — "
            "utiliser run_with_ocr_text() pour la post-correction sans engine."
        )
    ocr_step = LegacyOCREngineExecutor(pipeline.ocr_engine)
    spec = make_ocr_llm_pipeline_spec(
        mode=mode,
        ocr_adapter_name=pipeline.ocr_engine.name,
        llm_adapter_name=llm_name,
        llm_params=llm_params,
    )
    return spec, ocr_step


def _build_spec_for_run_with_ocr_text(
    pipeline: "OCRLLMPipeline",
    llm_name: str,
) -> tuple[PipelineSpec, None]:
    """Spec pour ``run_with_ocr_text()`` — 1 seul step LLM, RAW_TEXT
    et IMAGE viennent des inputs initiaux."""
    mode = pipeline.mode.value
    llm_params = {"prompt_template": pipeline._prompt_template}

    llm_input_types: list[ArtifactType] = [ArtifactType.RAW_TEXT]
    llm_inputs_from: dict[ArtifactType, str] = {
        ArtifactType.RAW_TEXT: INITIAL_STEP_ID,
    }
    if mode == "text_and_image":
        llm_input_types.append(ArtifactType.IMAGE)
        llm_inputs_from[ArtifactType.IMAGE] = INITIAL_STEP_ID

    spec = PipelineSpec(
        name=f"post_correction_{mode}_{_safe_name_for_id(llm_name)}",
        description=(
            f"Post-correction LLM mono-step (mode {mode}, "
            f"texte OCR pré-fourni)"
        ),
        initial_inputs=(ArtifactType.IMAGE, ArtifactType.RAW_TEXT),
        steps=(
            PipelineStep(
                id="llm",
                kind="post_correction",
                adapter_name=llm_name,
                params=llm_params,
                input_types=tuple(llm_input_types),
                output_types=(ArtifactType.CORRECTED_TEXT,),
                inputs_from=llm_inputs_from,
            ),
        ),
    )
    return spec, None


def _make_image_artifact(image_path: Path, doc_id: str) -> Artifact:
    return Artifact(
        id=f"{doc_id}:initial:image",
        document_id=doc_id,
        type=ArtifactType.IMAGE,
        uri=str(image_path),
    )


def _extract_outputs(
    *,
    result: Any,
    mode: str,
    ocr_text: Optional[str],
) -> tuple[str, Optional[str]]:
    """Extrait ``(text_final, ocr_intermediate)`` du PipelineResult.

    En zero_shot : le VLM produit ``RAW_TEXT`` final.  Pas
    d'``ocr_intermediate``.

    En text_only / text_and_image : le LLM produit ``CORRECTED_TEXT``.
    L'``ocr_intermediate`` est l'``RAW_TEXT`` produit par l'OCR ou
    fourni via ``ocr_text`` (mode triplet).
    """
    text_final = ""
    ocr_intermediate: Optional[str] = ocr_text

    if mode == "zero_shot":
        # Le step VLM produit RAW_TEXT en sortie finale.
        for art in result.artifacts:
            if art.type == ArtifactType.RAW_TEXT and art.uri:
                text_final = Path(art.uri).read_text(encoding="utf-8")
                break
        return text_final, None

    # text_only / text_and_image : prendre CORRECTED_TEXT
    for art in result.artifacts:
        if art.type == ArtifactType.CORRECTED_TEXT and art.uri:
            text_final = Path(art.uri).read_text(encoding="utf-8")
            break

    # ocr_intermediate : si pas fourni, lire le RAW_TEXT produit
    if ocr_intermediate is None:
        for art in result.artifacts:
            if art.type == ArtifactType.RAW_TEXT and art.uri:
                ocr_intermediate = Path(art.uri).read_text(encoding="utf-8")
                break

    return text_final, ocr_intermediate


def _build_metadata(
    *,
    pipeline: "OCRLLMPipeline",
    ocr_intermediate: Optional[str],
    ocr_source: Optional[str],
) -> dict:
    metadata: dict = {
        "engine_version": pipeline._safe_version(),
        "pipeline_mode": pipeline.mode.value,
        "prompt_file": pipeline.prompt_path,
        "prompt_template": pipeline._prompt_template,
        "llm_model": pipeline.llm_adapter.model,
        "llm_provider": pipeline.llm_adapter.name,
        "pipeline_steps": pipeline._build_steps_info(),
        "is_pipeline": True,
    }
    if ocr_intermediate is not None:
        metadata["ocr_intermediate"] = ocr_intermediate
    if ocr_source is not None:
        metadata["ocr_source"] = ocr_source
    return metadata


def _engine_result_failure(
    *,
    pipeline: "OCRLLMPipeline",
    image_path: Path,
    error: str,
    duration: float,
    ocr_text: Optional[str],
) -> EngineResult:
    """Construit un ``EngineResult`` en échec quand l'executor lève."""
    metadata = _build_metadata(
        pipeline=pipeline,
        ocr_intermediate=ocr_text,
        ocr_source="corpus" if ocr_text is not None else None,
    )
    return EngineResult(
        engine_name=pipeline.name,
        image_path=str(image_path),
        text="",
        duration_seconds=round(duration, 4),
        error=error,
        metadata=metadata,
    )


def _safe_code_version() -> str:
    try:
        from picarones import __version__
        return __version__
    except ImportError:
        return "unknown"


def _safe_name_for_id(s: str) -> str:
    return (
        s.replace(":", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
        .lower()
    )


__all__ = ["run_pipeline_via_executor"]
