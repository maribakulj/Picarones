"""Adaptateurs LLM — Sprint S11.

Cible : déplacement de ``picarones.llm.{openai,anthropic,mistral,
ollama}_adapter``.  Wrappers minces autour des SDK provider, qui
exposent un ``complete(prompt, ...)`` uniforme.

Un adapter LLM ne sait **rien** d'OCR ou de patrimoine.  Il fait
``prompt → completion``.  La logique de pipeline (prompt
construction, post-traitement, gestion d'erreur) vit dans
``pipeline/`` ou dans le module utilisateur qui compose la
pipeline.
"""

from __future__ import annotations

__all__: list[str] = []
