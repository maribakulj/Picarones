"""Pipelines OCR+LLM legacy — Sprint C du plan v2.0 (mai 2026).

Sous-package transitoire qui contient ``OCRLLMPipeline`` (legacy)
et son helper ``_executor_runner``.  Pendant la phase de retrait
du legacy, ces modules vivent ici plutôt que dans
``picarones.pipelines/`` (top-level) pour respecter l'invariant
architectural ``test_layer_imports_are_legal`` — la couche
``adapters/`` autorise les imports legacy par design.

Périmètre
---------
- ``base.OCRLLMPipeline`` — wrapper composé OCR+LLM (3 modes).
  Délègue à ``picarones.pipeline.PipelineExecutor`` depuis
  Sprint B du plan v2.0.
- ``_executor_runner.run_pipeline_via_executor`` — pont
  mono-document utilisé par ``OCRLLMPipeline.run()``.

Trace de retrait
----------------
Ce sous-package sera supprimé entièrement quand
``OCRLLMPipeline`` n'aura plus aucun consommateur externe (les
callers actuels — ``web/benchmark_utils.py``, tests Sprint 3 et
15 — passeront alors à la construction directe d'une
``PipelineSpec`` via ``picarones.pipeline.make_ocr_llm_pipeline_spec``).
"""

from __future__ import annotations

from picarones.adapters.legacy_pipelines.base import (
    OCRLLMPipeline,
    PipelineMode,
)

__all__ = ["OCRLLMPipeline", "PipelineMode"]
