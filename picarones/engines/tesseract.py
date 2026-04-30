"""Adaptateur Tesseract 5 via pytesseract."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

from picarones.engines.base import BaseOCREngine, EngineResult

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
        ``_extract_token_confidences``.
        """
        lang = self.config.get("lang", "fra")
        psm = int(self.config.get("psm", 6))
        oem = int(self.config.get("oem", 3))
        return lang, f"--oem {oem} --psm {psm}"

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

        lang, custom_config = self._tesseract_args()

        image = Image.open(image_path)
        text: str = pytesseract.image_to_string(image, lang=lang, config=custom_config)
        return text.strip()

    def _extract_token_confidences(
        self, image_path: Path,
    ) -> Optional[list[dict[str, Any]]]:
        """Extrait les confidences mot par mot via ``image_to_data``.

        Retourne ``None`` quand pytesseract n'est pas disponible OU si
        l'extraction échoue (best-effort — on ne casse pas l'OCR si
        seule la calibration est indisponible).

        Format de sortie compatible Sprint 42 : liste de dicts
        ``{"token": str, "confidence": float}`` avec confidence ∈
        [0, 100] (Tesseract).  Les non-mots (conf = -1) et tokens
        vides sont ignorés.
        """
        if not _PYTESSERACT_AVAILABLE:
            return None
        if not self.config.get("expose_confidences", True):
            return None

        try:
            tesseract_cmd = self.config.get("tesseract_cmd")
            if tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

            lang, custom_config = self._tesseract_args()
            image = Image.open(image_path)
            data = pytesseract.image_to_data(
                image,
                lang=lang,
                config=custom_config,
                output_type=pytesseract.Output.DICT,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[tesseract] extraction des token_confidences dégradée : %s",
                exc,
            )
            return None

        texts = data.get("text") or []
        confs = data.get("conf") or []
        if not texts or len(texts) != len(confs):
            return None

        out: list[dict[str, Any]] = []
        for tok_text, conf in zip(texts, confs):
            tok_text = (tok_text or "").strip()
            if not tok_text:
                continue
            try:
                conf_val = float(conf)
            except (TypeError, ValueError):
                continue
            # Tesseract met -1 pour les segments non-mots ; le runner
            # Sprint 42 les filtre aussi mais on les écarte ici pour
            # éviter le bruit dans les diagnostics.
            if conf_val < 0:
                continue
            out.append({"token": tok_text, "confidence": conf_val})
        return out or None

    def run(self, image_path: str | Path) -> EngineResult:
        """Exécute Tesseract et expose les ``token_confidences`` natifs
        (via ``image_to_data``) en plus du texte.

        Surcharge du ``BaseOCREngine.run()`` (Sprint 33) qui ne
        mettait pas de confidences.  On garde la mesure du temps et la
        gestion des erreurs.  Si l'extraction des confidences échoue,
        on retourne quand même le texte avec ``token_confidences =
        None`` — le runner saute simplement le calcul de calibration
        sur ce document.
        """
        image_path = Path(image_path)
        start = time.perf_counter()
        text = ""
        error: Optional[str] = None
        token_confidences: Optional[list[dict[str, Any]]] = None
        try:
            text = self._run_ocr(image_path)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        else:
            # On n'extrait les confidences que si l'OCR de base a réussi
            token_confidences = self._extract_token_confidences(image_path)
        duration = time.perf_counter() - start
        return EngineResult(
            engine_name=self.name,
            image_path=str(image_path),
            text=text,
            duration_seconds=round(duration, 4),
            error=error,
            metadata={"engine_version": self._safe_version()},
            token_confidences=token_confidences,
        )

    @classmethod
    def from_config(cls, config: Optional[dict] = None) -> "TesseractEngine":
        return cls(config=config or {})
