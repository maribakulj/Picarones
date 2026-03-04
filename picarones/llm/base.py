"""Interface abstraite commune à tous les adaptateurs LLM."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


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
        """Point d'entrée public : appelle le LLM et mesure la durée."""
        start = time.perf_counter()
        try:
            text = self._call(prompt, image_b64)
            error = None
        except Exception as exc:  # noqa: BLE001
            text = ""
            error = str(exc)
        duration = time.perf_counter() - start
        return LLMResult(
            model_id=self.model,
            text=text,
            duration_seconds=round(duration, 4),
            error=error,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"
