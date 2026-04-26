"""Interface abstraite commune à tous les adaptateurs moteurs OCR."""

from __future__ import annotations

import hashlib
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from picarones.core.modules import ArtifactType, BaseModule


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


class BaseOCREngine(BaseModule):
    """Classe de base dont héritent tous les adaptateurs OCR.

    Sprint 33 — Phase 0.2 : ``BaseOCREngine`` hérite désormais de
    ``BaseModule`` (cf. ``picarones.core.modules``) afin que les moteurs
    OCR existants soient automatiquement utilisables comme nœuds d'une
    pipeline composée (axe B du plan d'évolution).  Aucune sous-classe
    OCR n'est touchée : la méthode ``process`` est implémentée ici et
    délègue à ``run`` puis à ``_run_ocr``.

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

    # Déclaration BaseModule — un OCR consomme une image et produit du texte.
    input_types = (ArtifactType.IMAGE,)
    output_types = (ArtifactType.TEXT,)
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

    # ──────────────────────────────────────────────────────────────────
    # Implémentation BaseModule (Sprint 33)
    # ──────────────────────────────────────────────────────────────────

    def process(self, inputs: dict[ArtifactType, Any]) -> dict[ArtifactType, Any]:
        """Exécute le moteur OCR comme un module générique.

        Wrapper rétrocompatible : extrait le chemin image de ``inputs``,
        appelle ``run()``, et retourne la sortie sous forme de dictionnaire
        ``{ArtifactType.TEXT: text}``.  Les erreurs sont conservées dans
        le résultat (cf. ``EngineResult.error``) plutôt que de lever, comme
        l'implémentation historique de ``run()``.
        """
        self.validate_inputs(inputs)
        result = self.run(inputs[ArtifactType.IMAGE])
        return {ArtifactType.TEXT: result.text}

    def metadata(self) -> dict:
        """Expose la version du moteur dans les métadonnées du module."""
        return {"engine_version": self._safe_version()}

    def run(self, image_path: str | Path) -> EngineResult:
        """Point d'entrée public : exécute l'OCR et mesure le temps d'exécution."""
        image_path = Path(image_path)
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
        # Sprint 30 — log la stacktrace en DEBUG pour aider au diagnostic
        # quand un moteur retourne ``"unknown"`` (utilisateur qui se
        # demande pourquoi). Ne pollue pas l'output normal (INFO+).
        try:
            return self.version()
        except Exception as exc:  # noqa: BLE001
            import logging
            logging.getLogger(__name__).debug(
                "[%s._safe_version] retourne 'unknown' suite à %s: %s",
                self.__class__.__name__, type(exc).__name__, exc,
                exc_info=True,
            )
            return "unknown"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
