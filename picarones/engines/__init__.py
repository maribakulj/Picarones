"""``picarones.engines`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.adapters.legacy_engines`.  Phase 7.A du
retrait du legacy.
"""

from __future__ import annotations

import warnings

from picarones.adapters.legacy_engines.base import BaseOCREngine, EngineResult
from picarones.adapters.legacy_engines.factory import engine_from_name
from picarones.adapters.legacy_engines.tesseract import TesseractEngine
from picarones.adapters.legacy_engines.mistral_ocr import MistralOCREngine
from picarones.adapters.legacy_engines.google_vision import GoogleVisionEngine
from picarones.adapters.legacy_engines.azure_doc_intel import AzureDocIntelEngine

warnings.warn(
    "picarones.engines is deprecated and will be removed in 2.0.  "
    "Import from picarones.adapters.legacy_engines instead.",
    DeprecationWarning,
    stacklevel=2,
)

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
