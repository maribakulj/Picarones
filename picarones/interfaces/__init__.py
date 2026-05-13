"""Couche 8 — Interfaces (CLI, web).

Couches de transport.  Code mince qui parse des arguments / des
requêtes HTTP, appelle un service applicatif, retourne une réponse.

**Aucune logique métier ici.**  Si tu te vois écrire un calcul, un
parsing de format, une orchestration → c'est qu'il vit ailleurs
(``app/services/`` typiquement).

Sous-packages :

- ``cli/`` — Click commands.  Cible Sprint S22.
- ``web/`` — FastAPI + routers + middlewares + templates SPA.
  Cible Sprint S21.

Règle d'import : peut importer ``app/`` uniquement (et les libs
externes spécifiques au transport : ``fastapi``, ``click``,
``starlette``, ``uvicorn``).  Pas d'accès direct aux adaptateurs.
"""

from __future__ import annotations

__all__: list[str] = []
