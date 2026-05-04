"""Interface abstraite commune à tous les adaptateurs LLM."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

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


def normalize_llm_content(raw: Any) -> str:
    """Normalise une réponse LLM en chaîne plate.

    Chantier 4 (post-Sprint 97) — propagation du fix Mistral
    Sprint 15 à tous les providers. Le SDK Mistral peut retourner
    une liste de ``ContentChunk`` au lieu d'une chaîne pour certains
    modèles/versions ; le SDK OpenAI peut faire de même quand on
    active des features de structuration. Ce helper applique la même
    discipline pour les 4 adapters :

    - ``str``                          → renvoyée telle quelle (ou ``""``).
    - ``None``                         → ``""``.
    - ``list[ContentChunk]``           → concaténation des ``.text``.
    - ``list[dict]`` avec clé ``text`` → concaténation des ``["text"]``.
    - ``list[str]``                    → concaténation directe.
    - autre objet avec ``.text``       → ``obj.text``.
    - autre                            → ``str(obj)`` (best-effort).

    Le résultat est garanti être une ``str`` ; ``""`` quand la réponse
    est vide. La fonction est idempotente : ``normalize_llm_content(s)
    == s`` pour toute chaîne ``s``.
    """
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for chunk in raw:
            if chunk is None:
                continue
            if isinstance(chunk, str):
                parts.append(chunk)
                continue
            if hasattr(chunk, "text"):
                txt = getattr(chunk, "text", None)
                if isinstance(txt, str):
                    parts.append(txt)
                    continue
            if isinstance(chunk, dict) and isinstance(chunk.get("text"), str):
                parts.append(chunk["text"])
                continue
            # Dernier recours — convertit le chunk en chaîne
            parts.append(str(chunk))
        return "".join(parts)
    if hasattr(raw, "text") and isinstance(getattr(raw, "text", None), str):
        return raw.text  # type: ignore[no-any-return]
    return str(raw)


def log_http_error(
    adapter_name: str,
    model: str,
    exc: Exception,
    *,
    env_var: Optional[str] = None,
) -> None:
    """Log standardisé des erreurs HTTP des SDK LLM.

    Chantier 4 (post-Sprint 97) — propagation du log discriminant
    Mistral/OpenAI à tous les providers. Inspecte ``status_code`` et
    ``http_status`` puis émet un warning ciblé selon le code :

    - 401 : clé API invalide/expirée (mention de la variable
      d'environnement à vérifier si fournie).
    - 429 : rate limit / quota dépassé.
    - 5xx : problème serveur côté provider.
    - autre / pas de status_code : log générique.

    L'exception n'est pas levée — l'appelant doit ``raise``
    explicitement après ce log s'il veut propager (le retry est géré
    par ``BaseLLMAdapter.complete`` selon ``_is_retryable``).
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if status == 401:
        suffix = f" Vérifier {env_var}." if env_var else ""
        logger.warning(
            "[%s] erreur HTTP 401 — clé API invalide ou expirée "
            "(modèle=%s).%s",
            adapter_name, model, suffix,
        )
    elif status == 429:
        logger.warning(
            "[%s] erreur HTTP 429 — quota dépassé ou rate-limit "
            "(modèle=%s). Réessayer plus tard.",
            adapter_name, model,
        )
    elif status is not None and status >= 500:
        logger.warning(
            "[%s] erreur HTTP %d — problème serveur (modèle=%s) : %s",
            adapter_name, status, model, exc,
        )
    else:
        logger.warning(
            "[%s] erreur lors de l'appel API (modèle=%s) : %s",
            adapter_name, model, exc,
        )


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

    Normalisation des réponses (chantier 4)
    ---------------------------------------
    Les sous-classes utilisent :func:`normalize_llm_content` sur la
    réponse SDK avant de la retourner — garantit qu'une réponse de
    type ``list[ContentChunk]`` (Mistral, parfois OpenAI) est
    convertie en ``str`` plate.

    Logging d'erreurs HTTP (chantier 4)
    -----------------------------------
    Les sous-classes utilisent :func:`log_http_error` pour produire
    un log discriminant par ``status_code`` (401 → clé invalide,
    429 → rate limit, 5xx → serveur).  Auparavant ce log était
    dupliqué chez Mistral/OpenAI et absent chez Anthropic.
    """

    # Variable d'environnement portant la clé API.  Sous-classes
    # surchargent (ex. ``"OPENAI_API_KEY"``) ; mention utilisée par
    # :func:`log_http_error` quand un 401 est rencontré.  ``None``
    # pour les providers sans clé (Ollama).
    api_key_env_var: Optional[str] = None

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


__all__ = [
    "BaseLLMAdapter",
    "LLMResult",
    "log_http_error",
    "normalize_llm_content",
]
