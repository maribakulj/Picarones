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

#: Politique CSP par défaut.
#:
#: Sprint 25 a extrait tout le JavaScript de la SPA (~1131 lignes) dans
#: ``picarones/web/static/web-app.js`` — c'est la victoire concrète. Reste
#: dans le HTML environ 30 ``onclick="..."`` inline qui forcent à conserver
#: ``'unsafe-inline'`` dans ``script-src``. Leur migration vers
#: ``addEventListener`` est planifiée (sous-sprint dédié à ne pas mélanger
#: avec l'extraction des templates pour limiter les risques de régression).
#: ``style-src`` reste sur ``'unsafe-inline'`` pour les ``style="..."``
#: sémantiques dans les partials (états vert/rouge/jaune).
DEFAULT_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; "
    "font-src 'self' data:; "
    "connect-src 'self'; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self'"
)


def get_csp_policy() -> str:
    """Retourne la CSP à appliquer (override possible via env)."""
    return os.environ.get("PICARONES_CSP", DEFAULT_CSP)


async def csp_middleware(request, call_next):
    """Middleware FastAPI : ajoute Content-Security-Policy + en-têtes durcis."""
    response = await call_next(request)
    response.headers.setdefault("Content-Security-Policy", get_csp_policy())
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    return response
