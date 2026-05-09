"""Factory legacy : instancier un ``BaseOCREngine`` à partir de son nom court.

Phase 7.A — module relocalisé depuis ``picarones.engines.factory``
vers ``picarones.adapters.legacy_engines.factory``.

Sprint H.2.b du plan v2.0 — équivalent canonique disponible :
``picarones.adapters.ocr.factory.ocr_adapter_from_name`` retourne
des ``BaseOCRAdapter`` (StepExecutor Protocol) directement
consommables par ``PipelineExecutor`` sans ``LegacyOCREngineExecutor``.
Les nouveaux callers doivent utiliser la factory canonique.  Cette
factory ne sera supprimée qu'avec ``BaseOCREngine`` lui-même
(H.2.d).

Discipline : ne pas importer ``click`` ici, sous peine de remonter une
dépendance interfaces dans la couche adapters.
"""

from __future__ import annotations

from picarones.adapters.legacy_engines.base import BaseOCREngine


def engine_from_name(engine_name: str, lang: str = "fra", psm: int = 6) -> BaseOCREngine:
    """Instancie un moteur OCR par son nom court.

    Parameters
    ----------
    engine_name:
        Identifiant court (``"tesseract"``/``"tess"``, ``"pero_ocr"``/``"pero"``).
    lang:
        Code langue propagé au moteur quand il en consomme un (Tesseract).
    psm:
        Mode de segmentation Tesseract (ignoré par les autres moteurs).

    Returns
    -------
    BaseOCREngine
        Instance prête à exécuter ``run(image_path)``.

    Raises
    ------
    ValueError
        Si le nom est inconnu ou si le moteur est indisponible (par
        exemple Pero OCR non installé). Le message inclut la liste des
        moteurs effectivement disponibles dans l'environnement courant.
    """
    from picarones.adapters.legacy_engines.tesseract import TesseractEngine

    if engine_name in {"tesseract", "tess"}:
        return TesseractEngine(config={"lang": lang, "psm": psm})

    try:
        from picarones.adapters.legacy_engines.pero_ocr import PeroOCREngine

        if engine_name in {"pero_ocr", "pero"}:
            return PeroOCREngine(config={"name": "pero_ocr"})
    except ImportError:
        pass

    raise ValueError(
        f"Moteur inconnu ou non disponible : '{engine_name}'. "
        "Moteurs supportés : tesseract, pero_ocr"
    )


__all__ = ["engine_from_name"]
