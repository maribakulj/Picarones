"""Adapters OCR du nouveau monde — Sprint A14-S26.

Contrat ``BaseOCRAdapter`` natif au rewrite : pas hérité du legacy
``picarones.engines.base.BaseOCREngine``, exprimé directement en
termes du nouveau ``ArtifactType`` et de l'interface
``execute(inputs, params, context)`` du ``PipelineExecutor``.

Implémentations livrées
-----------------------
- ``PrecomputedTextAdapter`` — lit un texte OCR pré-calculé depuis
  le filesystem.  Cas BnF : comparer N transcriptions déjà produites
  par d'autres outils sans relancer d'OCR.

Adapters concrets pour Tesseract / Pero OCR / Mistral OCR / Google
Vision / Azure DI : à écrire au cas par cas dans des sprints
dédiés, **natifs** au nouveau contrat (pas de shim sur le legacy
``picarones.engines``).
"""

from __future__ import annotations

from picarones.adapters.ocr.base import BaseOCRAdapter, OCRAdapterError
from picarones.adapters.ocr.precomputed import PrecomputedTextAdapter
from picarones.adapters.ocr.tesseract import TesseractAdapter

__all__ = [
    "BaseOCRAdapter",
    "OCRAdapterError",
    "PrecomputedTextAdapter",
    "TesseractAdapter",
]
