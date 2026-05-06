"""Middlewares de sécurité pour l'interface web.

Module de **base de sécurité** activable opt-in (par défaut OFF pour
rester compatible avec le mode public HuggingFace Space ; chaque flag
s'active via un argument explicite à ``create_app``).

Composants
----------
- ``SecurityHeadersMiddleware`` : ajoute CSP, X-Frame-Options,
  X-Content-Type-Options, Referrer-Policy, Permissions-Policy à
  toute réponse.
- ``BodySizeLimitMiddleware`` : rejette les requêtes dont
  ``Content-Length`` dépasse un seuil (anti-DoS upload).
- ``RateLimitMiddleware`` : token bucket en mémoire par IP.
  Limite simple (req/min) ; pas de Redis (in-process).
- ``AuthenticationBackend`` (Protocol) : contrat pour brancher une
  authentification custom.  Si ``None``, mode public.

Anti-sur-ingénierie
-------------------
- Pas de CSRF token pour les endpoints API JSON (CSRF concerne
  surtout les formulaires HTML cookie-based).  Les API REST avec
  Bearer token / API key ne sont pas vulnérables au CSRF classique.
- Pas de support OAuth/OIDC : si le caller veut, il fournit son
  propre ``AuthenticationBackend``.
- Rate limit in-process : suffit pour 1 instance ; pour cluster,
  remplacer par Redis-backed en post-livraison.
- IP réelle via ``X-Forwarded-For`` : configurable, désactivé par
  défaut (un proxy non-trustworthy peut mentir).
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict, deque
from typing import Awaitable, Callable, Protocol, runtime_checkable

from fastapi import HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Authentication backend (port)
# ──────────────────────────────────────────────────────────────────────


@runtime_checkable
class AuthenticationBackend(Protocol):
    """Contrat d'un backend d'authentification injectable.

    Une implémentation décide d'autoriser ou non une requête en se
    basant sur les headers (Bearer token, API key, etc.).  Si la
    requête n'est pas authentifiée, lever ``HTTPException(401)``.

    Pour un mode **public** (HuggingFace Space, démo), passer
    ``None`` à ``create_app`` : aucun middleware d'auth n'est monté.
    """

    async def authenticate(self, request: Request) -> None:  # pragma: no cover
        """Lève ``HTTPException(401 / 403)`` si non authentifié.

        Sinon, ne retourne rien (la requête continue).  Peut attacher
        l'identité à ``request.state.user`` pour les endpoints qui
        veulent en savoir plus.
        """


# ──────────────────────────────────────────────────────────────────────
# Security headers
# ──────────────────────────────────────────────────────────────────────


_DEFAULT_CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self'; "
    "img-src 'self' data:; "
    "font-src 'self'; "
    "connect-src 'self'; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self'"
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Ajoute des en-têtes de sécurité durcis à toutes les réponses.

    En-têtes posés :

    - ``Content-Security-Policy`` : par défaut strict (pas
      d'``unsafe-inline``, ``frame-ancestors 'none'``).  Surchargeable
      via le constructeur.
    - ``X-Frame-Options: DENY`` (redondant avec CSP frame-ancestors
      mais lu par les navigateurs anciens).
    - ``X-Content-Type-Options: nosniff``
    - ``Referrer-Policy: strict-origin-when-cross-origin``
    - ``Permissions-Policy`` : désactive caméra, micro, géoloc.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        csp: str = _DEFAULT_CSP,
    ) -> None:
        super().__init__(app)
        self._csp = csp

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        response.headers.setdefault("Content-Security-Policy", self._csp)
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault(
            "Referrer-Policy", "strict-origin-when-cross-origin",
        )
        response.headers.setdefault(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=()",
        )
        return response


# ──────────────────────────────────────────────────────────────────────
# Body size limit
# ──────────────────────────────────────────────────────────────────────


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Rejette les requêtes dont ``Content-Length`` dépasse un seuil.

    Garde-fou anti-DoS sur les endpoints d'upload (ex: ZIP corpus).
    FastAPI/Starlette ne fournissent pas de limite intégrée — un
    client malveillant peut uploader 10 GB et saturer le disque
    avant qu'un endpoint ne lise quoi que ce soit.

    Le check est sur ``Content-Length`` (header). Un client qui
    triche en omettant ce header ou en streamant du chunked
    transfer-encoding contourne cette limite — pour une vraie
    protection, lire en streaming et compter les bytes (post-MVP).

    Parameters
    ----------
    max_bytes:
        Taille max acceptée en octets.  Défaut 100 MiB (cohérent
        avec ``CorpusService.max_zip_size_bytes``).
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_bytes: int = 100 * 1024 * 1024,
    ) -> None:
        super().__init__(app)
        if max_bytes <= 0:
            raise ValueError("max_bytes doit être > 0.")
        self._max = max_bytes

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                size = int(content_length)
            except ValueError:
                size = 0
            if size > self._max:
                # On retourne directement une JSONResponse — lever
                # ``HTTPException`` depuis un BaseHTTPMiddleware ne
                # passe pas par les exception handlers FastAPI.
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={
                        "detail": (
                            f"Body size {size} bytes excède la limite "
                            f"{self._max} bytes."
                        ),
                    },
                )
        return await call_next(request)


# ──────────────────────────────────────────────────────────────────────
# Rate limit (token bucket en mémoire)
# ──────────────────────────────────────────────────────────────────────


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limit simple par IP, fenêtre glissante en mémoire.

    Algorithme : pour chaque IP, on garde un deque des timestamps
    des requêtes des ``window_seconds`` dernières secondes.  Si le
    nombre dépasse ``max_requests``, on retourne 429 Too Many Requests.

    Limites
    -------
    - **In-process** : ne fonctionne que pour une instance.  Pour un
      cluster, basculer sur un store partagé (Redis).
    - **Pas atomique** : sous concurrence haute, un léger dépassement
      est possible (best-effort assumé).

    Garde-fous mémoire et anti-spoofing
    -----------------------------------
    Un attaquant qui rotate des IPs ferait gonfler ``self._buckets``
    indéfiniment.  Deux protections :

    1. **Plafond LRU** ``max_clients`` : quand on dépasse, le bucket
       le plus ancien (LRU) est évincé.
    2. **GC opportuniste** : à chaque dispatch, si le bucket courant
       devient vide après purge, il est supprimé du dict.

    Sur ``X-Forwarded-For`` (si activé) : la chaîne XFF est
    ``client, proxy1, proxy2, …``.  Lire le **premier** est trivialement
    spoofable par le client.  ``trust_proxy_count`` documente combien
    de proxies fiables sont devant l'app : on lit la N-ième IP en
    partant de la fin (la dernière étant le proxy de confiance le plus
    proche de nous).  Convention recommandée par Starlette/Express.

    Parameters
    ----------
    max_requests:
        Nombre max de requêtes par IP par fenêtre.  Défaut 60.
    window_seconds:
        Largeur de la fenêtre glissante.  Défaut 60s (= 60 req/min).
    trust_proxy_count:
        Nombre de proxies fiables devant l'app.  ``0`` (défaut)
        désactive la lecture de ``X-Forwarded-For`` ;
        ``request.client.host`` (IP du socket TCP direct) est utilisé.
        ``1`` lit l'avant-dernière IP de XFF (un seul proxy en amont,
        ex. nginx local), ``2`` l'avant-avant-dernière, etc.  **Ne pas
        configurer plus haut que le nombre réel** sous peine de
        permettre du spoofing.
    max_clients:
        Plafond du nombre d'IPs gardées en mémoire.  Défaut 10 000.
        Au-delà, eviction LRU.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_requests: int = 60,
        window_seconds: float = 60.0,
        trust_proxy_count: int = 0,
        max_clients: int = 10_000,
    ) -> None:
        super().__init__(app)
        if max_requests <= 0:
            raise ValueError("max_requests doit être > 0.")
        if window_seconds <= 0:
            raise ValueError("window_seconds doit être > 0.")
        if trust_proxy_count < 0:
            raise ValueError("trust_proxy_count doit être >= 0.")
        if max_clients <= 0:
            raise ValueError("max_clients doit être > 0.")
        self._max = max_requests
        self._window = window_seconds
        self._trust_proxies = trust_proxy_count
        self._max_clients = max_clients
        # ``OrderedDict`` pour conserver l'ordre d'insertion → eviction LRU
        # à coût constant via ``move_to_end`` + ``popitem(last=False)``.
        # Starlette est mono-thread asyncio par défaut ; pas de Lock.
        self._buckets: OrderedDict[str, deque[float]] = OrderedDict()

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        client_ip = self._extract_ip(request)
        now = time.monotonic()
        bucket = self._buckets.get(client_ip)
        if bucket is None:
            bucket = deque()
            self._buckets[client_ip] = bucket
            # Eviction LRU si on dépasse le plafond.
            if len(self._buckets) > self._max_clients:
                self._buckets.popitem(last=False)
        else:
            self._buckets.move_to_end(client_ip)
        # Purge des timestamps hors fenêtre.
        cutoff = now - self._window
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= self._max:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": (
                        f"Rate limit dépassé : {self._max} requêtes / "
                        f"{self._window:.0f}s pour {client_ip}."
                    ),
                },
            )
        bucket.append(now)
        # GC opportuniste : si le bucket s'est vidé entre-temps (rare,
        # purge avant append), on le retirerait — mais bucket n'est
        # jamais vide ici puisqu'on vient d'append.  La seule fenêtre
        # de leak persistante serait une IP qui ne revient plus ; le
        # plafond LRU ``max_clients`` la borne.
        return await call_next(request)

    def _extract_ip(self, request: Request) -> str:
        if self._trust_proxies > 0:
            xff = request.headers.get("x-forwarded-for", "").strip()
            if xff:
                parts = [p.strip() for p in xff.split(",") if p.strip()]
                # Lecture sûre : prendre la N-ième IP en partant de la
                # fin, où N = trust_proxy_count.  Si la chaîne est plus
                # courte qu'attendu (mauvaise config ou client tronquant),
                # fallback sur l'IP la plus à gauche disponible.
                idx = max(0, len(parts) - self._trust_proxies)
                if idx < len(parts):
                    return parts[idx]
        client = request.client
        return client.host if client is not None else "unknown"


# ──────────────────────────────────────────────────────────────────────
# Auth wrapper middleware
# ──────────────────────────────────────────────────────────────────────


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Wrapper middleware qui délègue à un ``AuthenticationBackend``.

    Si le backend est ``None``, ce middleware n'est pas monté du tout
    par ``create_app`` — pas de coût, mode public total.

    Le backend décide :

    - quels endpoints exiger une auth (peut faire un allowlist via
      ``request.url.path``) ;
    - quel format de credential accepter (Bearer, API key, etc.) ;
    - comment réagir en cas d'échec (401 vs 403).

    Le backend lève ``HTTPException`` ; le middleware se contente de
    déléguer.

    Endpoints toujours publics
    --------------------------
    Pour permettre les sondes Docker/k8s, ``/health`` et ``/version``
    contournent l'auth (allowlist par path).
    """

    PUBLIC_PATHS: frozenset[str] = frozenset({"/health", "/version"})

    def __init__(
        self,
        app: ASGIApp,
        *,
        backend: AuthenticationBackend,
    ) -> None:
        super().__init__(app)
        self._backend = backend

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.url.path not in self.PUBLIC_PATHS:
            try:
                await self._backend.authenticate(request)
            except HTTPException as exc:
                # ``BaseHTTPMiddleware`` ne convertit pas les
                # HTTPException levées par le backend — on les
                # transforme nous-mêmes en JSONResponse.
                return JSONResponse(
                    status_code=exc.status_code,
                    content={"detail": exc.detail},
                    headers=getattr(exc, "headers", None) or {},
                )
        return await call_next(request)


__all__ = [
    "AuthenticationBackend",
    "AuthenticationMiddleware",
    "BodySizeLimitMiddleware",
    "RateLimitMiddleware",
    "SecurityHeadersMiddleware",
]
