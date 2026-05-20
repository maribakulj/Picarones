"""Rate limiting + plafond de concurrence (extrait de ``security.py``).

dégonflage du god-module ``security``.  Réimporté
par ``security`` (API préservée).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque

logger = logging.getLogger(__name__)


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
    # Import différé : ``is_public_mode`` vit dans ``security`` qui
    # réimporte ce module — éviter le cycle à l'import.
    from picarones.interfaces.web.security import is_public_mode

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


__all__ = ["RateLimiter", "get_max_concurrent_jobs", "get_rate_limit_per_hour"]
