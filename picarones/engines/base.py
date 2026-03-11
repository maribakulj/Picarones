"""Interface abstraite commune à tous les adaptateurs moteurs OCR."""

from __future__ import annotations

import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_base_logger = logging.getLogger(__name__)


@dataclass
class EngineResult:
    """Résultat brut produit par un moteur OCR sur une image."""

    engine_name: str
    image_path: str
    text: str
    duration_seconds: float
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def image_sha256(self) -> str:
        return hashlib.sha256(Path(self.image_path).read_bytes()).hexdigest()


class BaseOCREngine(ABC):
    """Classe de base dont héritent tous les adaptateurs OCR.

    Chaque adaptateur doit implémenter :
    - ``name`` : identifiant unique du moteur
    - ``version()`` : retourne la version du moteur sous forme de chaîne
    - ``_run_ocr(image_path)`` : logique d'exécution OCR, retourne le texte brut

    Attribut de classe
    ------------------
    execution_mode : ``"io"`` (défaut) ou ``"cpu"``
        Indique au runner quel type d'exécuteur utiliser :
        - ``"io"``  → ``ThreadPoolExecutor``  (moteurs API / réseau)
        - ``"cpu"`` → ``ProcessPoolExecutor`` (moteurs CPU-intensifs : Tesseract, Pero, Kraken)
    """

    execution_mode: str = "io"
    """``"io"`` pour ThreadPoolExecutor (défaut), ``"cpu"`` pour ProcessPoolExecutor."""

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config: dict = config or {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifiant unique et stable du moteur."""

    @abstractmethod
    def version(self) -> str:
        """Retourne la version du moteur (ex : '5.3.0')."""

    @abstractmethod
    def _run_ocr(self, image_path: Path) -> str:
        """Exécute l'OCR et retourne le texte brut extrait."""

    def run(self, image_path: str | Path) -> EngineResult:
        """Point d'entrée public : exécute l'OCR et mesure le temps d'exécution."""
        image_path = Path(image_path)
        # ENTRY TRACE — confirme quel moteur/classe est réellement exécuté
        _base_logger.info(
            "[BaseOCREngine-ENTRY] run() — classe=%s, doc=%s",
            self.__class__.__name__, image_path.name,
        )
        start = time.perf_counter()
        try:
            text = self._run_ocr(image_path)
            error = None
        except Exception as exc:  # noqa: BLE001
            text = ""
            error = str(exc)
        duration = time.perf_counter() - start
        return EngineResult(
            engine_name=self.name,
            image_path=str(image_path),
            text=text,
            duration_seconds=round(duration, 4),
            error=error,
            metadata={"engine_version": self._safe_version()},
        )

    def _safe_version(self) -> str:
        try:
            return self.version()
        except Exception:  # noqa: BLE001
            return "unknown"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
