"""Couche 5 — Adapters.

Implémentations concrètes des contrats du domain.  C'est ici que
vivent les dépendances externes lourdes (pytesseract, pero_ocr,
mistralai, openai, anthropic, google-cloud-vision, datasets, etc.).

Sous-packages :

- ``ocr/`` — Tesseract, Pero OCR, Kraken, Mistral OCR, Google
  Vision, Azure Doc Intel.  Cible Sprint S11.
- ``llm/`` — OpenAI, Anthropic, Mistral, Ollama.  Cible S11.
- ``vlm/`` — Qwen-VL, Gemini, Claude vision, etc.  À remplir
  post-livraison (dans la limite de ce qui justifie une vraie
  comparaison avec OCR+LLM).
- ``corpus/`` — local folder, IIIF, Gallica, HTR-United,
  HuggingFace Datasets, eScriptorium.  Cible S11.
- ``storage/`` — filesystem, SQLite (jobs, history).  Cible S20.

Règles d'import : un adapter peut importer le domain et ses libs
externes.  Il ne doit **jamais** importer ``app/`` ou
``interfaces/``.  Il n'a aucune logique d'évaluation (un OCR
adapter ne calcule pas le CER — il produit un artefact texte que
``evaluation/`` consommera).
"""

from __future__ import annotations

__all__: list[str] = []
