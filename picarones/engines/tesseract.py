"""Adaptateur Tesseract 5 via pytesseract."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from picarones.engines.base import BaseOCREngine

try:
    import pytesseract
    from PIL import Image

    _PYTESSERACT_AVAILABLE = True
except ImportError:
    _PYTESSERACT_AVAILABLE = False


logger = logging.getLogger(__name__)


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

    Moteur CPU-bound : utilise ``ProcessPoolExecutor`` dans le runner parallèle.

    Configuration YAML :
    ```yaml
    name: tesseract
    engine: tesseract
    lang: fra          # code langue Tesseract (fra, lat, eng, ...)
    psm: 6             # Page Segmentation Mode (0-13)
    oem: 3             # OCR Engine Mode (0=legacy, 3=LSTM, 3=default)
    tesseract_cmd: tesseract  # chemin vers l'exécutable si non standard
    expose_confidences: true  # défaut ; mettre à false pour économiser
                              # un appel image_to_data par document
    ```

    Sprint 47 — exposition des token_confidences
    --------------------------------------------
    L'adapter appelle ``image_to_data`` en parallèle de
    ``image_to_string`` pour produire ``EngineResult.token_confidences``
    (liste de ``{"token": str, "confidence": float}``).  Le runner
    Sprint 42 calcule alors automatiquement la calibration ECE/MCE.

    Le texte ``EngineResult.text`` reste **strictement identique** à
    celui produit par ``image_to_string`` (pas de reconstruction depuis
    ``image_to_data``) — rétrocompatibilité octet par octet.

    Le coût supplémentaire est d'un second appel Tesseract par image.
    Pour le désactiver : ``expose_confidences: false`` dans la config.

    Refactor du chantier 1 (post-Sprint 97)
    ---------------------------------------
    L'adapter ne surcharge plus ``run()`` — il implémente
    ``_run_with_native`` et ``_extract_raw_confidences`` (les hooks
    factorisés dans ``BaseOCREngine``).  Comportement externe et
    octets de sortie strictement identiques aux versions Sprint 47+.
    """

    execution_mode = "cpu"

    @property
    def name(self) -> str:
        return self.config.get("name", "tesseract")

    def version(self) -> str:
        if not _PYTESSERACT_AVAILABLE:
            raise RuntimeError("pytesseract n'est pas installé.")
        return pytesseract.get_tesseract_version().vstring

    def _tesseract_args(self) -> tuple[str, str]:
        """Retourne ``(lang, custom_config)`` selon la config courante.

        Centralisé pour rester cohérent entre ``_run_ocr`` et
        ``_run_with_native``.
        """
        lang = self.config.get("lang", "fra")
        psm = int(self.config.get("psm", 6))
        oem = int(self.config.get("oem", 3))
        return lang, f"--oem {oem} --psm {psm}"

    def _apply_tesseract_cmd(self) -> None:
        """Applique le chemin Tesseract custom si la config en fournit un."""
        tesseract_cmd = self.config.get("tesseract_cmd")
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def _run_ocr(self, image_path: Path) -> str:
        if not _PYTESSERACT_AVAILABLE:
            raise RuntimeError(
                "pytesseract n'est pas installé. "
                "Installez-le avec : pip install pytesseract"
            )

        self._apply_tesseract_cmd()
        lang, custom_config = self._tesseract_args()
        image = Image.open(image_path)
        text: str = pytesseract.image_to_string(image, lang=lang, config=custom_config)
        return text.strip()

    def _run_with_native(self, image_path: Path) -> tuple[str, Optional[dict]]:
        """Appelle ``image_to_string`` puis ``image_to_data``.

        Retourne ``(text, image_to_data_dict)`` — la deuxième valeur
        peut être ``None`` si ``expose_confidences`` est à ``False``
        ou si l'appel ``image_to_data`` échoue (best-effort).

        Le texte reste **identique** à celui produit par
        ``_run_ocr`` (rétrocompat octet par octet — Sprint 47).
        """
        text = self._run_ocr(image_path)
        if not self.config.get("expose_confidences", True):
            return text, None
        try:
            self._apply_tesseract_cmd()
            lang, custom_config = self._tesseract_args()
            image = Image.open(image_path)
            data = pytesseract.image_to_data(
                image,
                lang=lang,
                config=custom_config,
                output_type=pytesseract.Output.DICT,
            )
            return text, data
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[tesseract] extraction des token_confidences "
                "(image_to_data) indisponible : %s — calibration "
                "sautée pour ce document",
                exc,
            )
            return text, None

    def _extract_raw_confidences(
        self, native: Any,
    ) -> Optional[list[dict[str, Any]]]:
        """Parse le ``image_to_data`` dict de Tesseract.

        Format Tesseract : dict ``{"text": [...], "conf": [...], ...}``
        avec confidences ∈ [0, 100] et ``-1`` pour les segments
        non-mots — ces derniers sont écartés par
        ``_normalize_token_confidences`` (filtre les conf < 0).
        """
        if not isinstance(native, dict):
            return None
        texts = native.get("text") or []
        confs = native.get("conf") or []
        if not texts or len(texts) != len(confs):
            return None
        out: list[dict[str, Any]] = []
        for tok_text, conf in zip(texts, confs):
            out.append({"token": tok_text, "confidence": conf})
        return out or None

    def _extract_token_confidences(
        self, image_path: Path,
    ) -> Optional[list[dict[str, Any]]]:
        """Alias rétrocompat (Sprint 47) — extrait les confidences depuis ``image_path``.

        Pipeline interne du chantier 1 : ``_run_with_native`` → ``_extract_raw_confidences``
        → ``_normalize_token_confidences``. Retourne ``None`` si pytesseract est
        absent ou si l'extraction échoue (signal au runner de sauter la calibration).
        """
        if not _PYTESSERACT_AVAILABLE:
            return None
        try:
            _text, native = self._run_with_native(Path(image_path))
            raw = self._extract_raw_confidences(native)
            return self._normalize_token_confidences(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[tesseract] extraction des token_confidences indisponible : %s",
                exc,
            )
            return None

    @classmethod
    def from_config(cls, config: Optional[dict] = None) -> "TesseractEngine":
        return cls(config=config or {})
