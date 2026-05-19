"""Factories stateless du ``RunOrchestrator`` (GT / inputs / contexte).

Audit prod P1.1 — éclatement du module plat ``run_orchestrator_
helpers`` en sous-package cohésif.  Aucune dépendance vers
``RunOrchestrator`` ni vers l'état d'instance.
"""

from __future__ import annotations

import threading
from typing import Callable

from picarones.app.services.corpus_service import CorpusImportError
from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.domain.documents import DocumentRef
from picarones.pipeline import RunContext


def _default_gt_factory(
    doc: DocumentRef, art_type: ArtifactType,
) -> Artifact | None:
    """Factory GT par défaut.

    Convention : un candidat ``CORRECTED_TEXT`` est comparé contre
    la GT ``RAW_TEXT`` (les deux sont du texte plat — la distinction
    de type ne porte que sur le côté candidat).  Cas typique : un
    pipeline OCR + post-correction LLM produit un ``CORRECTED_TEXT``
    qu'on compare au ``.gt.txt`` original.
    """
    effective_type = (
        ArtifactType.RAW_TEXT
        if art_type == ArtifactType.CORRECTED_TEXT
        else art_type
    )
    gt_ref = doc.gt_for(effective_type)
    if gt_ref is None:
        return None
    return Artifact(
        id=f"{doc.id}:gt:{effective_type.value}",
        document_id=doc.id,
        type=effective_type,
        uri=gt_ref.uri,
    )


def _default_inputs_factory(doc: DocumentRef) -> dict[ArtifactType, Artifact]:
    """``{IMAGE: artifact_image}``.  Lève si ``doc.image_uri`` absent."""
    if doc.image_uri is None:
        raise CorpusImportError(
            f"Document {doc.id!r} sans ``image_uri`` — la pipeline "
            "par défaut consomme une IMAGE en entrée.",
        )
    return {ArtifactType.IMAGE: Artifact(
        id=f"{doc.id}:image",
        document_id=doc.id,
        type=ArtifactType.IMAGE,
        uri=doc.image_uri,
    )}


def _make_context_factory(
    code_version: str,
    *,
    progress_callback: Callable[[str, int, str], None] | None = None,
    workspace_uri: str | None = None,
) -> Callable[[DocumentRef, str], RunContext]:
    """Phase B2.1 — factory de ``RunContext`` avec callback de progression.

    Pattern strictement copié de
    ``_benchmark_execution.py:109-139`` (legacy) pour garantir
    l'équivalence numérique du compteur ``doc_idx`` à
    ``run_benchmark_via_service``.

    Le ``counter_lock`` partagé empêche les race conditions quand le
    ``CorpusRunner`` traite plusieurs documents en parallèle
    (``max_in_flight > 1``).  Le compteur est **global au run**, pas
    par pipeline — un benchmark à 2 pipelines × 5 docs émet 10
    notifications ``doc_idx ∈ {0..9}``.

    Phase B4 — ``workspace_uri`` propagé au ``RunContext`` pour que les
    adapters qui en ont besoin (PrecomputedTextAdapter,
    TesseractAdapter, etc.) puissent écrire leurs artefacts
    intermédiaires.  Cohérent avec ``_benchmark_execution.py:134-139``.
    """
    counter_lock = threading.Lock()
    counter_state = {"doc_idx": 0}

    def _factory(doc: DocumentRef, pipeline_name: str) -> RunContext:
        if progress_callback is not None:
            with counter_lock:
                idx = counter_state["doc_idx"]
                counter_state["doc_idx"] = idx + 1
            try:
                progress_callback(pipeline_name, idx, doc.id)
            except Exception:
                # On ignore silencieusement les erreurs du callback :
                # un caller qui crashe ne doit pas faire tomber le
                # benchmark.  Cohérent avec
                # ``_benchmark_execution.py:126-133``.
                import logging
                logging.getLogger(__name__).debug(
                    "[run_orchestrator] progress_callback levé — "
                    "ignoré pour ne pas tomber le bench.",
                    exc_info=True,
                )
        return RunContext(
            document_id=doc.id,
            code_version=code_version,
            pipeline_name=pipeline_name,
            workspace_uri=workspace_uri,
        )
    return _factory


__all__ = [
    "_default_gt_factory",
    "_default_inputs_factory",
    "_make_context_factory",
]
