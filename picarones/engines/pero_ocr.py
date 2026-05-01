"""Adaptateur Pero OCR.

Pero OCR est un moteur HTR/OCR performant sur les documents historiques,
développé par l'Université technologique de Brno.

Dépendance : pero-ocr  (pip install pero-ocr)
Dépôt      : https://github.com/DCGM/pero-ocr

Configuration YAML :
```yaml
name: pero_ocr
engine: pero_ocr
config: /chemin/vers/config.ini   # fichier de configuration Pero OCR
cuda: false                        # utiliser le GPU si disponible
expose_confidences: true           # défaut ; expose la confidence par ligne
                                   # (transcription_confidence) à tous les
                                   # mots de la ligne, format Sprint 42
```

Sprint 48 — exposition des token_confidences
---------------------------------------------
Pero OCR fournit ``line.transcription_confidence`` (probabilité moyenne CTC
sur la ligne).  L'adapter applique cette confidence à chaque **mot** de la
ligne (granularité disponible la plus fine sans logits CTC).

Refactor du chantier 1 (post-Sprint 97)
---------------------------------------
L'adapter ne surcharge plus ``run()`` — il implémente ``_run_with_native``
et ``_extract_raw_confidences`` (les hooks factorisés dans ``BaseOCREngine``).
Comportement externe et octets de sortie strictement identiques.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from picarones.engines.base import BaseOCREngine

try:
    import numpy as np
    from PIL import Image

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    from pero_ocr.document_ocr.layout import PageLayout
    from pero_ocr.document_ocr.page_parser import PageParser

    _PERO_AVAILABLE = True
except ImportError:
    _PERO_AVAILABLE = False


logger = logging.getLogger(__name__)


class PeroOCREngine(BaseOCREngine):
    """Adaptateur pour Pero OCR.

    Pero OCR produit une sortie structurée (PAGE XML) ; cet adaptateur
    en extrait le texte plat dans l'ordre de lecture naturel et, depuis
    le Sprint 48, les confidences au niveau mot (héritées de la
    confidence ligne ``transcription_confidence``).

    Moteur CPU-bound : utilise ``ProcessPoolExecutor`` dans le runner parallèle.
    """

    execution_mode = "cpu"

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._parser: Optional[object] = None

    @property
    def name(self) -> str:
        return self.config.get("name", "pero_ocr")

    def version(self) -> str:
        if not _PERO_AVAILABLE:
            raise RuntimeError("pero-ocr n'est pas installé.")
        try:
            import pero_ocr

            return getattr(pero_ocr, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "[pero_ocr] version non disponible : %s", exc, exc_info=True,
            )
            return "unknown"

    def _get_parser(self) -> "PageParser":
        """Instancie le PageParser (lazy, une seule fois par moteur)."""
        if self._parser is None:
            if not _PERO_AVAILABLE:
                raise RuntimeError(
                    "pero-ocr n'est pas installé. "
                    "Installez-le avec : pip install pero-ocr"
                )
            config_path = self.config.get("config")
            if not config_path:
                raise ValueError(
                    "La configuration Pero OCR requiert un paramètre 'config' "
                    "pointant vers un fichier .ini Pero OCR valide."
                )
            import configparser

            parser_config = configparser.ConfigParser()
            parser_config.read(config_path)
            self._parser = PageParser(parser_config)
        return self._parser  # type: ignore[return-value]

    def _run_pero_pipeline(self, image_path: Path) -> tuple[str, Any]:
        """Exécute le pipeline Pero OCR et retourne ``(text, page_layout)``."""
        if not _PIL_AVAILABLE:
            raise RuntimeError("Pillow n'est pas installé.")

        parser = self._get_parser()

        image = np.array(Image.open(image_path).convert("RGB"))
        page_layout = PageLayout(id=image_path.stem, page_size=(image.shape[0], image.shape[1]))

        # Exécution du pipeline Pero OCR
        parser.process_page(image, page_layout)

        # Extraction du texte plat dans l'ordre des lignes
        lines = []
        for region in page_layout.regions:
            for line in region.lines:
                if line.transcription:
                    lines.append(line.transcription.strip())

        return "\n".join(lines), page_layout

    def _run_ocr(self, image_path: Path) -> str:
        text, _ = self._run_pero_pipeline(image_path)
        return text

    def _run_with_native(self, image_path: Path) -> tuple[str, Any]:
        """Exécute Pero OCR et retourne ``(text, page_layout)``.

        Un seul passage du pipeline coûteux ; le ``page_layout``
        contient toutes les informations nécessaires à l'extraction
        des confidences (Sprint 48).
        """
        return self._run_pero_pipeline(image_path)

    def _extract_raw_confidences(
        self, native: Any,
    ) -> Optional[list[dict[str, Any]]]:
        """Extrait les confidences au niveau mot depuis ``page_layout``.

        Stratégie : pour chaque ligne, on prend
        ``line.transcription_confidence`` (probabilité CTC moyenne) et
        on l'applique à chaque mot de la ligne.  Granularité minimale
        sans déchiffrer les logits CTC, mais suffisante pour la
        calibration.
        """
        if not self.config.get("expose_confidences", True):
            return None
        if native is None:
            return None
        out: list[dict[str, Any]] = []
        for region in getattr(native, "regions", []) or []:
            for line in getattr(region, "lines", []) or []:
                transcription = getattr(line, "transcription", None)
                if not transcription:
                    continue
                conf = getattr(line, "transcription_confidence", None)
                if conf is None:
                    continue
                for word in transcription.strip().split():
                    if word:
                        out.append({"token": word, "confidence": conf})
        return out or None

    @classmethod
    def from_config(cls, config: Optional[dict] = None) -> "PeroOCREngine":
        return cls(config=config or {})
