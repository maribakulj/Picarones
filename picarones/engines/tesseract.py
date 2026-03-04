"""Adaptateur Tesseract 5 via pytesseract."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from picarones.engines.base import BaseOCREngine

try:
    import pytesseract
    from PIL import Image

    _PYTESSERACT_AVAILABLE = True
except ImportError:
    _PYTESSERACT_AVAILABLE = False


# Correspondance des valeurs PSM acceptées en argument YAML/CLI
_PSM_LABELS = {
    0: "Orientation and script detection only",
    1: "Automatic page segmentation with OSD",
    3: "Fully automatic page segmentation (default)",
    4: "Single column of text",
    5: "Single uniform block of vertically aligned text",
    6: "Single uniform block of text",
    7: "Single text line",
    8: "Single word",
    9: "Single word in a circle",
    10: "Single character",
    11: "Sparse text",
    12: "Sparse text with OSD",
    13: "Raw line",
}


class TesseractEngine(BaseOCREngine):
    """Adaptateur pour Tesseract 5 (via pytesseract).

    Configuration YAML :
    ```yaml
    name: tesseract
    engine: tesseract
    lang: fra          # code langue Tesseract (fra, lat, eng, ...)
    psm: 6             # Page Segmentation Mode (0-13)
    oem: 3             # OCR Engine Mode (0=legacy, 3=LSTM, 3=default)
    tesseract_cmd: tesseract  # chemin vers l'exécutable si non standard
    ```
    """

    @property
    def name(self) -> str:
        return self.config.get("name", "tesseract")

    def version(self) -> str:
        if not _PYTESSERACT_AVAILABLE:
            raise RuntimeError("pytesseract n'est pas installé.")
        return pytesseract.get_tesseract_version().vstring

    def _run_ocr(self, image_path: Path) -> str:
        if not _PYTESSERACT_AVAILABLE:
            raise RuntimeError(
                "pytesseract n'est pas installé. "
                "Installez-le avec : pip install pytesseract"
            )

        # Paramétrage optionnel de l'exécutable
        tesseract_cmd = self.config.get("tesseract_cmd")
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        lang = self.config.get("lang", "fra")
        psm = int(self.config.get("psm", 6))
        oem = int(self.config.get("oem", 3))

        custom_config = f"--oem {oem} --psm {psm}"

        image = Image.open(image_path)
        text: str = pytesseract.image_to_string(image, lang=lang, config=custom_config)
        return text.strip()

    @classmethod
    def from_config(cls, config: Optional[dict] = None) -> "TesseractEngine":
        return cls(config=config or {})
