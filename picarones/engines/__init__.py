"""Adaptateurs moteurs OCR."""

from picarones.engines.base import BaseOCREngine, EngineResult
from picarones.engines.tesseract import TesseractEngine

__all__ = ["BaseOCREngine", "EngineResult", "TesseractEngine"]

try:
    from picarones.engines.pero_ocr import PeroOCREngine

    __all__.append("PeroOCREngine")
except ImportError:
    pass
