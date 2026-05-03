"""Adaptateurs OCR — Sprint S11.

Cible : déplacement (sans modification logique) de
``picarones.engines.{tesseract,pero_ocr,mistral_ocr,google_vision,
azure_doc_intel}``.  Chaque adapter implémente le protocole
``StepExecutor`` du package ``pipeline``.

Règle : un adapter OCR produit un artefact ``RAW_TEXT`` (et
optionnellement ``ALTO_XML`` / ``token_confidences``).  Il ne
calcule **rien** sur ce texte — pas de CER, pas de normalisation,
pas d'analyse linguistique.  Tout ça est dans ``evaluation/``.
"""

from __future__ import annotations

__all__: list[str] = []
