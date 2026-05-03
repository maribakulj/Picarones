"""FastAPI app — Sprint S21.

Cible : nouvelle implémentation **mince** des routers.  Chaque
endpoint = 5-15 lignes max : valide DTO Pydantic, appelle un
service de ``app/``, retourne une réponse.

Sécurité durcie par défaut au S21 (cohérence avec les 6 P0 du S1) :

- CSRF activé par défaut (plus de ``PICARONES_CSRF_REQUIRED``
  opt-in).
- CSP sans ``'unsafe-inline'`` (refactor JS pour supprimer les
  ``onclick=`` inline).
- Cookie ``Secure`` détecté automatiquement (header
  ``X-Forwarded-Proto`` + liste de proxies de confiance).
- Rate limit sur IP **réelle** (proxy chain validée), plus
  d'``X-Forwarded-For`` aveugle.
- Workspaces isolés par session via ``WorkspaceManager``.

Le code de ``picarones.web`` reste en place jusqu'au S22 — au S21
on construit la nouvelle SPA en parallèle, au S22 on bascule
``picarones`` script entry et on supprime l'ancien ``web/``.
"""

from __future__ import annotations

__all__: list[str] = []
