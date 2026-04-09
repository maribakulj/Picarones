"""Interface abstraite commune à tous les adaptateurs LLM."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Paramètres de retry par défaut
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BACKOFF_BASE = 2.0  # secondes : 2, 4, 8


def _is_retryable(exc: Exception) -> bool:
    """Détermine si une exception est retryable (429, 5xx, timeout réseau)."""
    # HTTP status codes retryables
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if status is not None:
        return status == 429 or status >= 500

    # Erreurs réseau / timeout
    exc_name = type(exc).__name__
    if exc_name in ("TimeoutError", "ConnectionError", "URLError"):
        return True

    # Messages d'erreur courants
    msg = str(exc).lower()
    if "rate" in msg and "limit" in msg:
        return True
    if "timeout" in msg or "connection" in msg:
        return True
    if "429" in msg or "503" in msg or "502" in msg:
        return True

    return False


@dataclass
class LLMResult:
    """Résultat produit par un appel LLM."""

    model_id: str
    text: str
    duration_seconds: float
    tokens_used: Optional[int] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


class BaseLLMAdapter(ABC):
    """Classe de base pour tous les adaptateurs LLM.

    Chaque adaptateur doit implémenter :
    - ``name``         : identifiant du provider (ex : 'openai')
    - ``default_model``: modèle par défaut du provider
    - ``_call()``      : appel API effectif, retourne le texte brut

    Les clés API sont lues depuis les variables d'environnement uniquement.

    Retry automatique
    -----------------
    Les erreurs retryables (HTTP 429, 5xx, timeout réseau) sont automatiquement
    retentées avec backoff exponentiel (2s, 4s, 8s par défaut). Configurable
    via ``config["max_retries"]`` et ``config["retry_backoff"]``.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> None:
        self.config: dict = config or {}
        self.model: str = model or self.default_model

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifiant du provider (ex : 'openai', 'anthropic')."""

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Modèle utilisé si aucun n'est fourni explicitement."""

    @abstractmethod
    def _call(self, prompt: str, image_b64: Optional[str] = None) -> str:
        """Appel LLM effectif.

        Parameters
        ----------
        prompt:
            Texte du prompt final (variables déjà substituées).
        image_b64:
            Image encodée en base64 (sans préfixe data URI).
            None pour les appels texte-uniquement.

        Returns
        -------
        str
            Texte généré par le LLM.
        """

    def complete(
        self,
        prompt: str,
        image_b64: Optional[str] = None,
    ) -> LLMResult:
        """Point d'entrée public : appelle le LLM avec retry automatique."""
        max_retries = int(self.config.get("max_retries", _DEFAULT_MAX_RETRIES))
        backoff_base = float(self.config.get("retry_backoff", _DEFAULT_BACKOFF_BASE))

        start = time.perf_counter()
        last_exc: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                text = self._call(prompt, image_b64)
                duration = time.perf_counter() - start
                return LLMResult(
                    model_id=self.model,
                    text=text,
                    duration_seconds=round(duration, 4),
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < max_retries and _is_retryable(exc):
                    wait = backoff_base ** (attempt + 1)
                    logger.warning(
                        "[%s] erreur retryable (tentative %d/%d, attente %.1fs) : %s",
                        self.name, attempt + 1, max_retries + 1, wait, exc,
                    )
                    time.sleep(wait)
                else:
                    break

        duration = time.perf_counter() - start
        return LLMResult(
            model_id=self.model,
            text="",
            duration_seconds=round(duration, 4),
            error=str(last_exc),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"
