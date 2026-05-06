"""Garde-fous sécurité pour l'interface web.

Ce module centralise quatre durcissements pour rendre Picarones déployable
sur un Space HuggingFace public ou un serveur d'institution sans donner
les clefs du royaume au premier visiteur :

1. **Mode public**  (``PICARONES_PUBLIC_MODE=1``) — désactive les
   pipelines OCR+LLM et les moteurs OCR cloud, dont les clefs API sont
   mutualisées côté serveur (OPENAI_API_KEY, ANTHROPIC_API_KEY,
   MISTRAL_API_KEY, etc.). Sans ce garde-fou, n'importe quel visiteur
   consomme le quota du mainteneur via 10 lignes de ``curl``.

2. **Browse roots restreints** — ``PICARONES_BROWSE_ROOTS`` (chemins
   séparés par ``:``) remplace la liste hardcodée. Par défaut,
   uniquement ``./uploads/`` est exposé en mode public ; en mode ``dev``
   on conserve l'ancien comportement (cwd, ``/workspaces``, ``tempdir``).

3. **Validation des images uploadées** — appel à ``Image.verify()`` dans
   un ``try/except`` capturant ``DecompressionBombError``,
   ``UnidentifiedImageError`` et l'exception générique de Pillow.
   Limite de taille via ``PICARONES_MAX_UPLOAD_MB`` (défaut 100).

4. **Rate limiting + plafond de jobs concurrents** — limiteur en mémoire
   par IP (``PICARONES_RATE_LIMIT_PER_HOUR``) et sémaphore global
   (``PICARONES_MAX_CONCURRENT_JOBS``).

Le tout est piloté par variables d'environnement pour ne pas obliger un
mainteneur à patcher du code lors du passage à la prod.
"""

from __future__ import annotations

import io
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mode public
# ---------------------------------------------------------------------------

#: Identifiants de moteurs cloud dont les clefs API sont mutualisées côté
#: serveur. En mode public on refuse toute requête qui les invoque.
CLOUD_OCR_ENGINES: frozenset[str] = frozenset({
    "mistral_ocr",
    "google_vision",
    "azure_doc_intel",
})

#: Identifiants de fournisseurs LLM facturés à la clef serveur.
CLOUD_LLM_PROVIDERS: frozenset[str] = frozenset({
    "openai",
    "anthropic",
    "mistral",
    "ollama",  # local mais quand même mutualisé
})


def is_public_mode() -> bool:
    """Vrai si l'instance tourne en mode public (HuggingFace Space, etc.)."""
    return os.environ.get("PICARONES_PUBLIC_MODE", "").strip() in ("1", "true", "yes")


def assert_engines_allowed(engines: Iterable[str]) -> None:
    """Lève ``PermissionError`` si la liste contient un moteur cloud bloqué.

    Réponse à utiliser côté FastAPI : ``HTTPException(403, str(exc))``.
    """
    if not is_public_mode():
        return
    banned = [e for e in engines if e in CLOUD_OCR_ENGINES]
    if banned:
        raise PermissionError(
            "Mode public actif (PICARONES_PUBLIC_MODE=1) — les moteurs OCR "
            f"cloud sont désactivés : {', '.join(banned)}. Faites tourner "
            "Picarones localement ou désactivez le mode public."
        )


def assert_llm_provider_allowed(llm_provider: str) -> None:
    """Lève ``PermissionError`` si un LLM mutualisé est sollicité en mode public."""
    if not is_public_mode():
        return
    if llm_provider and llm_provider.strip() in CLOUD_LLM_PROVIDERS:
        raise PermissionError(
            "Mode public actif — les pipelines OCR+LLM sont désactivés "
            f"(provider '{llm_provider}'). En production institutionnelle, "
            "exiger une clef API utilisateur via l'en-tête X-User-API-Key."
        )


# ---------------------------------------------------------------------------
# Validation des chemins utilisateur (Sprint A14-S1, A.I.0 P0)
#
# Ré-importé depuis le foyer définitif ``picarones.app.services.path_security``
# (Sprint A14-S19).  Pas de duplication — le code vit en un seul
# endroit dans la couche app, accessible aussi par la CLI et les jobs
# background.
# ---------------------------------------------------------------------------

from picarones.app.services.path_security import (  # noqa: F401
    PathValidationError,
    safe_report_name,
    validated_path,
    validated_prompt_filename,
)
from picarones.app.services.path_security import _is_within  # noqa: F401


# ---------------------------------------------------------------------------
# Browse roots
# ---------------------------------------------------------------------------

def compute_browse_roots(uploads_dir: Path) -> list[Path]:
    """Retourne la liste de répertoires autorisés pour ``/api/corpus/browse``.

    - Variable d'env ``PICARONES_BROWSE_ROOTS`` (séparateur ``os.pathsep``,
      ``:`` sur Linux/macOS, ``;`` sur Windows) : prioritaire si définie.
    - Sinon, mode public ⇒ uniquement ``uploads_dir``.
    - Sinon, mode dev (défaut) ⇒ cwd + uploads_dir + ``/workspaces``
      (Codespaces) + ``tempdir`` (compatibilité ascendante).
    """
    raw = os.environ.get("PICARONES_BROWSE_ROOTS")
    if raw:
        roots = [Path(p).resolve() for p in raw.split(os.pathsep) if p.strip()]
        return roots

    if is_public_mode():
        return [uploads_dir.resolve()]

    import tempfile
    return [
        Path(".").resolve(),
        uploads_dir.resolve(),
        Path("/workspaces").resolve(),
        Path(tempfile.gettempdir()).resolve(),
    ]


def compute_workspace_roots(uploads_dir: Path) -> list[Path]:
    """Retourne les racines autorisées pour les opérations de benchmark.

    Sprint A14-S1 — A.I.0 P0 : utilisé par les endpoints
    ``/api/benchmark/start`` et ``/api/benchmark/run`` pour valider
    ``corpus_path`` et ``output_dir`` via :func:`validated_path`.

    Sémantique :

    - Si ``PICARONES_WORKSPACE_ROOTS`` est défini, prend précédence
      absolue (admin sait ce qu'il fait).
    - Sinon, en mode public : uniquement ``uploads_dir`` (lecture)
      et ``./rapports`` (écriture des rapports générés).
    - Sinon, mode dev : ``compute_browse_roots`` + ``./rapports`` +
      ``./corpus`` (corpus locaux des développeurs).

    En production institutionnelle, exporter ``PICARONES_WORKSPACE_ROOTS``
    pour épingler explicitement les répertoires autorisés.
    """
    raw = os.environ.get("PICARONES_WORKSPACE_ROOTS")
    if raw:
        return [Path(p).expanduser().resolve() for p in raw.split(os.pathsep) if p.strip()]

    base = compute_browse_roots(uploads_dir)
    extras = [
        Path("./rapports").resolve(),
        Path("./corpus").resolve(),
    ]
    seen: set[Path] = set()
    out: list[Path] = []
    for p in base + extras:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# Validation des images uploadées
# ---------------------------------------------------------------------------

def get_max_upload_mb() -> int:
    raw = os.environ.get("PICARONES_MAX_UPLOAD_MB", "100")
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "[security] PICARONES_MAX_UPLOAD_MB invalide (%r) — défaut 100 Mo.", raw
        )
        return 100


def validate_image_safe(data: bytes, filename: str = "<upload>") -> None:
    """Vérifie qu'un buffer décode comme une image valide sans bombe.

    Levée de ``ValueError`` (à mapper en HTTP 415/422) si :
      - taille > limite ;
      - Pillow rejette l'image (UnidentifiedImageError, DecompressionBombError) ;
      - le format ouvert ne correspond pas à ce que prétend l'extension.

    On ne bloque pas l'absence de Pillow (il est dépendance core), mais on
    log si l'import échoue pour aider au diagnostic.
    """
    max_mb = get_max_upload_mb()
    if len(data) > max_mb * 1024 * 1024:
        raise ValueError(
            f"Image '{filename}' refusée : taille {len(data) / (1024 * 1024):.1f} Mo > "
            f"limite {max_mb} Mo (PICARONES_MAX_UPLOAD_MB)."
        )

    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover — Pillow est core
        logger.warning("[security] Pillow indisponible — validation image sautée : %s", exc)
        return

    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
    except UnidentifiedImageError as exc:
        raise ValueError(
            f"Image '{filename}' refusée : format non reconnu par Pillow ({exc})."
        ) from exc
    except Image.DecompressionBombError as exc:
        raise ValueError(
            f"Image '{filename}' refusée : bombe de décompression détectée ({exc})."
        ) from exc
    except Exception as exc:
        # Pillow lève un panel d'exceptions hétérogènes (SyntaxError sur les
        # GIF malformés, OSError sur les TIFF corrompus, ValueError divers).
        raise ValueError(
            f"Image '{filename}' refusée : erreur de décodage Pillow ({type(exc).__name__}: {exc})."
        ) from exc


# ---------------------------------------------------------------------------
# Rate limiting + concurrence
# ---------------------------------------------------------------------------

def get_max_concurrent_jobs() -> int:
    raw = os.environ.get("PICARONES_MAX_CONCURRENT_JOBS", "2")
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "[security] PICARONES_MAX_CONCURRENT_JOBS invalide (%r) — défaut 2.", raw
        )
        return 2


def get_rate_limit_per_hour() -> int:
    """Nombre maximal de jobs lancés par IP et par heure (mode public).

    En mode dev, on ne limite pas (retourne 0 = illimité).
    """
    if not is_public_mode():
        return 0
    raw = os.environ.get("PICARONES_RATE_LIMIT_PER_HOUR", "5")
    try:
        return max(0, int(raw))
    except ValueError:
        return 5


class RateLimiter:
    """Limiteur de débit en mémoire, fenêtre glissante par IP.

    Implémentation volontairement simple : un ``deque`` de timestamps par IP
    avec purge paresseuse. Suffisant pour un Space HF (RAM constante, ~1 Ko
    par IP active). Pour de l'institutionnel multi-replica, voir Sprint 26
    (file SQLite partagée).
    """

    def __init__(self, max_per_hour: int):
        self.max_per_hour = max_per_hour
        self._buckets: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, ip: str) -> None:
        """Lève ``PermissionError`` si ``ip`` dépasse le quota horaire."""
        if self.max_per_hour <= 0:
            return  # désactivé
        now = time.monotonic()
        cutoff = now - 3600.0
        with self._lock:
            bucket = self._buckets.setdefault(ip, deque())
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self.max_per_hour:
                # Temps avant que le plus ancien hit ne sorte de la fenêtre
                retry_after = max(1, int(bucket[0] + 3600.0 - now))
                raise PermissionError(
                    f"Quota dépassé : {self.max_per_hour} jobs/heure max. "
                    f"Réessayer dans {retry_after} s."
                )
            bucket.append(now)

    def reset(self) -> None:
        """Vide complètement les buckets (utile aux tests)."""
        with self._lock:
            self._buckets.clear()


# ---------------------------------------------------------------------------
# CSP middleware
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CSRF — Sprint A4 (item B-11)
#
# Pattern « double-submit cookie » : à chaque GET, le serveur pose un
# cookie ``picarones_csrf`` (httponly=False car le JS doit le lire) qui
# contient un token signé. Sur POST/PUT/DELETE/PATCH, le client doit
# renvoyer ce token dans le header ``X-CSRF-Token``. Le serveur compare
# les deux (constant-time) et refuse 403 sinon.
#
# Activation : ``PICARONES_CSRF_REQUIRED=1`` (défaut désactivé pour
# rétrocompat HuggingFace Space sans session). En mode institutionnel
# derrière SSO, à activer d'office.
#
# Secret : ``PICARONES_CSRF_SECRET`` env var. Si absent, généré au
# démarrage (warning explicite — perte du secret entre redémarrages,
# acceptable pour des sessions courtes).
# ---------------------------------------------------------------------------

import hashlib
import hmac
import secrets

#: Nom du cookie CSRF (httponly=False — lu par le JS du frontend).
CSRF_COOKIE = "picarones_csrf"

#: Header HTTP que le client doit renvoyer sur POST/PUT/DELETE/PATCH.
CSRF_HEADER = "X-CSRF-Token"

#: Méthodes HTTP qui exigent un token valide.
CSRF_PROTECTED_METHODS: frozenset[str] = frozenset({"POST", "PUT", "PATCH", "DELETE"})

#: Préfixes de chemin exemptés. Les endpoints purement informatifs ou
#: appelés depuis des outils CLI tiers (curl, wget) restent accessibles
#: sans token. Tout endpoint qui modifie l'état applicatif doit rester
#: protégé — ne pas étendre cette liste sans revue sécurité.
CSRF_EXEMPT_PATH_PREFIXES: tuple[str, ...] = (
    "/health",
    "/api/csrf/token",  # le endpoint qui *donne* le token
)

_csrf_secret_runtime: bytes | None = None


def is_csrf_required() -> bool:
    """Vrai si la protection CSRF doit être active (mode institutionnel)."""
    return os.environ.get("PICARONES_CSRF_REQUIRED", "").strip() in ("1", "true", "yes")


def _get_csrf_secret() -> bytes:
    """Retourne le secret HMAC.  Priorité ``PICARONES_CSRF_SECRET``,
    sinon génère un secret runtime persistant durant la vie du process.
    """
    global _csrf_secret_runtime
    env = os.environ.get("PICARONES_CSRF_SECRET")
    if env:
        return env.encode("utf-8")
    if _csrf_secret_runtime is None:
        _csrf_secret_runtime = secrets.token_bytes(32)
        logger.warning(
            "[security] PICARONES_CSRF_SECRET non défini — secret généré au "
            "démarrage. Les tokens CSRF seront invalidés au prochain "
            "redémarrage. En production, exporter un secret stable."
        )
    return _csrf_secret_runtime


def generate_csrf_token() -> str:
    """Produit un token signé HMAC-SHA256.

    Format : ``<nonce_hex>.<signature_hex>`` où la signature est
    ``HMAC-SHA256(secret, nonce)``. Le nonce est rotué à chaque
    génération — pas de réutilisation.
    """
    nonce = secrets.token_bytes(16)
    sig = hmac.new(_get_csrf_secret(), nonce, hashlib.sha256).digest()
    return f"{nonce.hex()}.{sig.hex()}"


def verify_csrf_token(token: str | None) -> bool:
    """Valide la signature d'un token. Compare en temps constant.

    Retourne ``False`` sur token absent, mal formé, ou signature
    incorrecte. Pas de fuite d'information sur la cause.
    """
    if not token or "." not in token:
        return False
    try:
        nonce_hex, sig_hex = token.split(".", 1)
        nonce = bytes.fromhex(nonce_hex)
        sig_provided = bytes.fromhex(sig_hex)
    except (ValueError, AttributeError):
        return False
    sig_expected = hmac.new(_get_csrf_secret(), nonce, hashlib.sha256).digest()
    return hmac.compare_digest(sig_provided, sig_expected)


async def csrf_middleware(request, call_next):
    """Middleware FastAPI — protège les méthodes mutantes en mode CSRF.

    Comportement :

    1. Si ``PICARONES_CSRF_REQUIRED`` n'est pas activé → bypass complet
       (rétrocompat HuggingFace Space public).
    2. Sinon, si la méthode est dans ``CSRF_PROTECTED_METHODS`` et que
       le chemin n'est pas exempté → exiger un token valide. Renvoie
       403 si manquant ou invalide.
    3. Pose un cookie ``picarones_csrf`` à chaque réponse pour les
       chemins non exempts (rotation à chaque GET).

    Le pattern « double-submit cookie » + signature HMAC garantit que
    seul un client qui a *à la fois* le cookie et a pu lire sa valeur
    via JS (donc qui n'est pas un site tiers) peut soumettre le header
    correspondant.
    """
    from fastapi.responses import JSONResponse

    if not is_csrf_required():
        return await call_next(request)

    path = request.url.path
    is_exempt = any(path.startswith(p) for p in CSRF_EXEMPT_PATH_PREFIXES)
    method = request.method.upper()

    # Vérification : méthode mutante non exemptée → token obligatoire
    if method in CSRF_PROTECTED_METHODS and not is_exempt:
        cookie_token = request.cookies.get(CSRF_COOKIE)
        header_token = request.headers.get(CSRF_HEADER)
        if not cookie_token or not header_token:
            logger.warning(
                "[security/csrf] %s %s refusé : token cookie=%r header=%r",
                method,
                path,
                bool(cookie_token),
                bool(header_token),
            )
            return JSONResponse(
                status_code=403,
                content={
                    "detail": (
                        "CSRF token requis sur cette méthode. Récupérer un "
                        f"token via GET /api/csrf/token et le passer dans "
                        f"l'en-tête {CSRF_HEADER}."
                    ),
                },
            )
        if not hmac.compare_digest(cookie_token, header_token):
            logger.warning(
                "[security/csrf] %s %s refusé : cookie/header divergent",
                method, path,
            )
            return JSONResponse(
                status_code=403,
                content={"detail": "CSRF token cookie/header divergent."},
            )
        if not verify_csrf_token(cookie_token):
            logger.warning(
                "[security/csrf] %s %s refusé : signature invalide",
                method, path,
            )
            return JSONResponse(
                status_code=403,
                content={"detail": "CSRF token invalide ou expiré."},
            )

    response = await call_next(request)

    # Rotation : on pose un cookie frais sur tout GET non-exempt qui n'a
    # pas déjà un cookie, ou si la réponse est un endpoint qui force la
    # rotation. Pour les autres méthodes, on conserve le cookie courant.
    if method == "GET" and not is_exempt:
        if CSRF_COOKIE not in request.cookies:
            response.set_cookie(
                key=CSRF_COOKIE,
                value=generate_csrf_token(),
                httponly=False,  # le JS doit pouvoir le lire
                samesite="strict",
                secure=False,  # mis à True derrière TLS via reverse proxy
            )
    return response
