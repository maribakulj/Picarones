"""Retry exponentiel partagé par les adapters cloud (OCR + LLM).

Pour une release institutionnelle (BnF, LoC, BL), un benchmark de
N milliers de documents face à un service cloud (Google Vision,
Azure Document Intelligence, Mistral OCR, Anthropic, OpenAI) doit
absorber les erreurs transitoires (429, 5xx, timeout réseau) sans
faire échouer le doc — sinon les résultats partiels ne sont pas
reproductibles d'un run à l'autre.

Ce module fournit la politique commune.  Il vit au top du package
``adapters/`` (et non sous ``llm/`` ou ``ocr/``) parce qu'il est
consommé par les deux familles indistinctement.

API
---
- ``is_retryable(exc)`` : True si l'exception est typique d'un
  problème transitoire.
- ``call_with_retry(callable, max_retries, backoff_base, label)`` :
  exécute le callable, retry exponentiel jusqu'à ``max_retries``
  tentatives.  Lève la dernière exception si épuisé.

Politique
---------
- ``max_retries=3`` (4 tentatives au total : 0 + 1 + 2 + 3 retries).
- ``backoff_base=2.0`` → 2s, 4s, 8s entre les retries (16s cumul max).
- Logs WARNING à chaque retry avec contexte.

Anti-sur-ingénierie
-------------------
- Pas de jitter randomisé : pas indispensable à ce volume ; ajouter
  si un caller en a concrètement besoin.
- Pas de circuit breaker : un caller qui voit 100 % d'échec sur 5000
  documents arrête le run lui-même.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE = 2.0  # secondes : 2, 4, 8

T = TypeVar("T")


def is_retryable(exc: Exception) -> bool:
    """``True`` si l'exception est typique d'un problème transitoire.

    Détection sur trois axes :

    1. Code HTTP exposé par les SDK cloud (``status_code`` ou
       ``http_status``) : 429 (rate limit) et tout 5xx.
    2. Type d'exception réseau : ``TimeoutError``, ``ConnectionError``,
       ``URLError`` (urllib).
    3. Heuristique sur le message (fallback pour les SDK qui ne
       structurent pas) : présence des codes 429/502/503 ou des
       motifs ``rate limit``, ``timeout``, ``connection``.
    """
    status = (
        getattr(exc, "status_code", None)
        or getattr(exc, "http_status", None)
    )
    if status is not None:
        return status == 429 or status >= 500

    exc_name = type(exc).__name__
    if exc_name in ("TimeoutError", "ConnectionError", "URLError"):
        return True

    msg = str(exc).lower()
    if "rate" in msg and "limit" in msg:
        return True
    if "timeout" in msg or "connection" in msg:
        return True
    if "429" in msg or "503" in msg or "502" in msg:
        return True

    return False


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    label: str = "adapter",
) -> T:
    """Exécute ``fn`` avec retry exponentiel sur erreurs retryables.

    Parameters
    ----------
    fn:
        Callable sans argument qui retourne le résultat ou lève.
    max_retries:
        Nombre de retries après la première tentative.  ``0`` =
        une seule tentative (pas de retry).
    backoff_base:
        Base de l'attente exponentielle.  Tentative ``i`` → attente
        ``backoff_base ** (i + 1)`` secondes avant retry.
    label:
        Étiquette du caller pour le logging (typiquement
        ``self.name`` de l'adapter).

    Returns
    -------
    Résultat de ``fn``.

    Raises
    ------
    Exception
        La dernière exception levée si tous les retries sont
        épuisés ou si l'erreur n'est pas retryable.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_retries and is_retryable(exc):
                wait = backoff_base ** (attempt + 1)
                logger.warning(
                    "[%s] erreur retryable (tentative %d/%d, "
                    "attente %.1fs) : %s",
                    label, attempt + 1, max_retries + 1, wait, exc,
                )
                time.sleep(wait)
            else:
                break
    assert last_exc is not None
    raise last_exc


__all__ = [
    "DEFAULT_BACKOFF_BASE",
    "DEFAULT_MAX_RETRIES",
    "call_with_retry",
    "is_retryable",
]
