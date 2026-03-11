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
```
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

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


class PeroOCREngine(BaseOCREngine):
    """Adaptateur pour Pero OCR.

    Pero OCR produit une sortie structurée (PAGE XML) ; cet adaptateur
    en extrait le texte plat dans l'ordre de lecture naturel.

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
        except Exception:  # noqa: BLE001
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

    def _run_ocr(self, image_path: Path) -> str:
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

        return "\n".join(lines)

    @classmethod
    def from_config(cls, config: Optional[dict] = None) -> "PeroOCREngine":
        return cls(config=config or {})
