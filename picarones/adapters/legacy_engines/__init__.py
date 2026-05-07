"""Engines OCR legacy — Sprint 33+ pré-rewrite.

Phase 7.A — package relocalisé depuis ``picarones.engines`` vers
``picarones.adapters.legacy_engines``.  Le chemin legacy reste
disponible via des shims avec ``DeprecationWarning`` ; suppression
prévue en 2.0.

Coexistence avec ``picarones.adapters.ocr``
-------------------------------------------
``evaluation.engines`` porte les 5 OCR engines historiques qui
héritent de ``BaseOCREngine`` (basé sur ``BaseModule``,
``run() → EngineResult``).  Ils sont consommés par le runner
legacy (``measurements/runner/``) et le ``PipelineRunner`` legacy.

``picarones.adapters.ocr`` (Sprint A14-S26) est la cible
canonique : un design ``StepExecutor`` Protocol, ``Artifact``
typés, sans héritage de ``BaseModule``.  Les 5 OCR adapters
canoniques (``TesseractAdapter``, etc.) y vivent.

La convergence des deux est documentée dans
``docs/migration/pipeline-convergence-plan.md`` (sub-phases
7.A-7.D, stratégie 4.B).  Tant que ``BaseModule`` n'est pas
retiré, les engines legacy gardent leur place.
"""

from __future__ import annotations

from picarones.adapters.legacy_engines.base import BaseOCREngine, EngineResult
from picarones.adapters.legacy_engines.factory import engine_from_name
from picarones.adapters.legacy_engines.tesseract import TesseractEngine
from picarones.adapters.legacy_engines.mistral_ocr import MistralOCREngine
from picarones.adapters.legacy_engines.google_vision import GoogleVisionEngine
from picarones.adapters.legacy_engines.azure_doc_intel import AzureDocIntelEngine

__all__ = [
    "BaseOCREngine",
    "EngineResult",
    "engine_from_name",
    "TesseractEngine",
    "MistralOCREngine",
    "GoogleVisionEngine",
    "AzureDocIntelEngine",
]

try:
    from picarones.adapters.legacy_engines.pero_ocr import PeroOCREngine  # noqa: F401

    __all__.append("PeroOCREngine")
except ImportError:
    pass
