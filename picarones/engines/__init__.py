"""Adaptateurs moteurs OCR."""

from picarones.engines.base import BaseOCREngine, EngineResult
from picarones.engines.tesseract import TesseractEngine
from picarones.engines.mistral_ocr import MistralOCREngine
from picarones.engines.google_vision import GoogleVisionEngine
from picarones.engines.azure_doc_intel import AzureDocIntelEngine

__all__ = [
    "BaseOCREngine",
    "EngineResult",
    "TesseractEngine",
    "MistralOCREngine",
    "GoogleVisionEngine",
    "AzureDocIntelEngine",
]

try:
    from picarones.engines.pero_ocr import PeroOCREngine

    __all__.append("PeroOCREngine")
except ImportError:
    pass
