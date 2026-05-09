"""Adapters OCR — couche 5 (libs externes autorisées).

Contrat ``BaseOCRAdapter`` exprimé en termes du ``ArtifactType``
et de l'interface ``execute(inputs, params, context)`` consommée
par ``PipelineExecutor``.

Implémentations livrées
-----------------------
- ``TesseractAdapter`` — Tesseract 5 (OSS, CPU-bound).
- ``PeroOCRAdapter`` — Pero OCR (manuscrits, GPU recommandé).
- ``MistralOCRAdapter`` — Mistral OCR API (cloud).
- ``GoogleVisionAdapter`` — Google Vision API (cloud).
- ``AzureDocIntelAdapter`` — Azure Document Intelligence (cloud).
- ``PrecomputedTextAdapter`` — lit un texte OCR pré-calculé depuis
  le filesystem.  Cas BnF : comparer N transcriptions déjà produites
  par d'autres outils sans relancer d'OCR.
"""

from __future__ import annotations

from picarones.adapters.ocr.azure_doc_intel import AzureDocIntelAdapter
from picarones.adapters.ocr.base import BaseOCRAdapter, OCRAdapterError
from picarones.adapters.ocr.factory import ocr_adapter_from_name
from picarones.adapters.ocr.google_vision import GoogleVisionAdapter
from picarones.adapters.ocr.mistral_ocr import MistralOCRAdapter
from picarones.adapters.ocr.pero_ocr import PeroOCRAdapter
from picarones.adapters.ocr.precomputed import PrecomputedTextAdapter
from picarones.adapters.ocr.tesseract import TesseractAdapter

__all__ = [
    "BaseOCRAdapter",
    "OCRAdapterError",
    "AzureDocIntelAdapter",
    "GoogleVisionAdapter",
    "MistralOCRAdapter",
    "PeroOCRAdapter",
    "PrecomputedTextAdapter",
    "TesseractAdapter",
    "ocr_adapter_from_name",
]
