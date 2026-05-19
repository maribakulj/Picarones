"""Content-Security-Policy + en-têtes durcis (extrait de ``security.py``).

Audit prod P1.2 — dégonflage du god-module ``security``.  Réimporté
par ``security`` (API préservée).
"""

from __future__ import annotations

import os


def is_huggingface_space() -> bool:
    """Vrai si l'instance tourne dans un HuggingFace Space.

    HuggingFace injecte ``SPACE_ID`` (au format ``user/space``) dans
    l'environnement du container — c'est le marqueur canonique
    documenté par HuggingFace, présent quel que soit le SDK (Docker,
    Streamlit, Gradio…). On l'utilise pour adapter automatiquement la
    CSP : un Space est servi via une ``<iframe>`` côté
    ``huggingface.co`` / ``*.hf.space``, donc ``frame-ancestors 'none'``
    et ``X-Frame-Options: DENY`` rendent la SPA invisible (page blanche
    bien que le serveur réponde).
    """
    return bool(os.environ.get("SPACE_ID", "").strip())


#: Origines autorisées à embarquer la SPA dans une iframe quand on tourne
#: dans un HuggingFace Space. ``huggingface.co`` est l'origine du Hub qui
#: rend la page parente, ``*.hf.space`` est le domaine où HF expose les
#: containers Space (utilisé par certains rendus directs et liens
#: partageables).
_HF_FRAME_ANCESTORS = "'self' https://huggingface.co https://*.hf.space"


def _frame_ancestors_directive() -> str:
    """Retourne la directive ``frame-ancestors`` adaptée au déploiement.

    - Local / institutionnel : ``'none'`` (pas d'embed possible).
    - HuggingFace Space : autorise ``huggingface.co`` et ``*.hf.space``
      pour que la SPA s'affiche dans l'iframe du Space sans tomber en
      page blanche.
    """
    return f"frame-ancestors {_HF_FRAME_ANCESTORS}" if is_huggingface_space() else "frame-ancestors 'none'"


#: Politique CSP par défaut (sans la directive ``frame-ancestors``, qui est
#: composée dynamiquement par :func:`get_csp_policy` selon le déploiement).
#:
#: Sprint 25 a extrait tout le JavaScript de la SPA (~1131 lignes) dans
#: ``picarones/web/static/web-app.js`` — c'est la victoire concrète. Reste
#: dans le HTML environ 30 ``onclick="..."`` inline qui forcent à conserver
#: ``'unsafe-inline'`` dans ``script-src``. Leur migration vers
#: ``addEventListener`` est planifiée (sous-sprint dédié à ne pas mélanger
#: avec l'extraction des templates pour limiter les risques de régression).
#: ``style-src`` reste sur ``'unsafe-inline'`` pour les ``style="..."``
#: sémantiques dans les partials (états vert/rouge/jaune).
_CSP_BASE = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; "
    "font-src 'self' data:; "
    "connect-src 'self'; "
    "base-uri 'self'; "
    "form-action 'self'"
)

#: Politique CSP complète exposée pour rétrocompatibilité (mode local
#: strict). En production HuggingFace, :func:`get_csp_policy` la
#: recompose dynamiquement avec ``frame-ancestors`` permissif.
DEFAULT_CSP = _CSP_BASE + "; frame-ancestors 'none'"


def get_csp_policy() -> str:
    """Retourne la CSP à appliquer (override possible via env).

    Si ``PICARONES_CSP`` est défini, il prend précédence absolue —
    l'admin sait ce qu'il fait. Sinon, on compose ``_CSP_BASE`` plus la
    directive ``frame-ancestors`` adaptée à l'environnement détecté
    (HF Space ou local).
    """
    override = os.environ.get("PICARONES_CSP")
    if override:
        return override
    return f"{_CSP_BASE}; {_frame_ancestors_directive()}"


async def csp_middleware(request, call_next):
    """Middleware FastAPI : ajoute Content-Security-Policy + en-têtes durcis.

    Sur HuggingFace Space, ``X-Frame-Options: DENY`` est sciemment omis :
    ce header (priorité absolue dans les anciens navigateurs, fallback
    moderne quand le navigateur ne supporte pas ``frame-ancestors``)
    bloque l'iframe parente du Hub HF même si la CSP est permissive.
    Le contrôle d'embed est alors entièrement délégué à
    ``frame-ancestors``.
    """
    response = await call_next(request)
    response.headers.setdefault("Content-Security-Policy", get_csp_policy())
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    if not is_huggingface_space():
        response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    return response


__all__ = [
    "DEFAULT_CSP",
    "csp_middleware",
    "get_csp_policy",
    "is_huggingface_space",
]
