"""Adaptateurs VLM (Vision-Language Models).

Volontairement vide à la livraison du rewrite ciblé.  Les VLM
arrivent post-livraison une fois que le pattern d'adapter LLM est
stabilisé et que les vues d'évaluation
(``HallucinationView``, ``ReconstructionView``) sont en place pour
les comparer honnêtement avec les pipelines OCR+LLM (cf.
``BACKLOG_POST_LIVRAISON.md`` §2.2).

Cibles à terme : Qwen-VL, Gemini Vision, GPT-4o Vision, Claude
Sonnet/Opus Vision, Pixtral.

Note : un VLM peut produire ``RAW_TEXT`` ou ``CANONICAL_DOCUMENT``
selon le mode (zero-shot transcription vs. document understanding).
Le pipeline le branche selon le besoin de l'expérience.
"""

from __future__ import annotations

__all__: list[str] = []
