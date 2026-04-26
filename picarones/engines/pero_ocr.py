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
``run()`` est surchargé : un seul appel ``parser.process_page`` produit
à la fois le texte et les confidences.  Pero OCR fournit
``line.transcription_confidence`` (probabilité moyenne CTC sur la
ligne).  L'adapter applique cette confidence à chaque **mot** de la
ligne (granularité disponible la plus fine sans logits CTC).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

from picarones.engines.base import BaseOCREngine, EngineResult

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

    def _run_pero_pipeline(self, image_path: Path) -> tuple[str, Any]:
        """Exécute le pipeline Pero OCR et retourne ``(text, page_layout)``.

        Centralisé pour que ``_run_ocr`` (rétrocompat) et ``run``
        (Sprint 48 — extraction des confidences) partagent un seul
        passage du pipeline coûteux.
        """
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

    def _extract_token_confidences_from_layout(
        self, page_layout: Any,
    ) -> Optional[list[dict[str, Any]]]:
        """Extrait les confidences au niveau mot depuis le ``page_layout``.

        Stratégie : pour chaque ligne, on prend
        ``line.transcription_confidence`` (probabilité CTC moyenne) et
        on l'applique à chaque mot de la ligne.  Granularité minimale
        sans déchiffrer les logits CTC, mais suffisante pour la
        calibration.

        Retourne ``None`` si :

        - ``page_layout`` est ``None`` (cas dégénéré),
        - aucune ligne n'a de ``transcription_confidence`` non-None,
        - aucun mot non-vide n'a été émis.

        Les exceptions sont absorbées en warning : la calibration
        d'un document peut échouer sans casser le benchmark.
        """
        if not self.config.get("expose_confidences", True):
            return None
        if page_layout is None:
            return None
        try:
            out: list[dict[str, Any]] = []
            for region in getattr(page_layout, "regions", []) or []:
                for line in getattr(region, "lines", []) or []:
                    transcription = getattr(line, "transcription", None)
                    if not transcription:
                        continue
                    conf = getattr(line, "transcription_confidence", None)
                    if conf is None:
                        continue
                    try:
                        conf_val = float(conf)
                    except (TypeError, ValueError):
                        continue
                    if conf_val < 0:
                        continue
                    for word in transcription.strip().split():
                        if not word:
                            continue
                        out.append({"token": word, "confidence": conf_val})
            return out or None
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[pero_ocr] extraction des token_confidences dégradée : %s",
                exc,
            )
            return None

    def run(self, image_path: str | Path) -> EngineResult:
        """Exécute Pero OCR et expose les ``token_confidences`` natifs.

        Surcharge du ``BaseOCREngine.run()`` (Sprint 33) qui ne
        propageait pas les confidences.  Le pipeline Pero est appelé
        **une seule fois** ; le texte et les confidences sont
        extraits depuis le même ``page_layout`` (zéro coût
        supplémentaire vs l'implémentation historique).
        """
        image_path = Path(image_path)
        start = time.perf_counter()
        text = ""
        error: Optional[str] = None
        token_confidences: Optional[list[dict[str, Any]]] = None
        try:
            text, page_layout = self._run_pero_pipeline(image_path)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        else:
            token_confidences = self._extract_token_confidences_from_layout(
                page_layout,
            )
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
    def from_config(cls, config: Optional[dict] = None) -> "PeroOCREngine":
        return cls(config=config or {})
