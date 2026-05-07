"""Factory : instancier un moteur OCR à partir de son nom court.

Vit en cercle 2 (``picarones.engines``) parce que c'est de la logique de
catalogue OCR — le CLI (cercle 3) et l'API web (cercle 3) la consomment
tous les deux. Auparavant ce helper était défini dans
``picarones.cli`` puis importé par ``picarones.web.benchmark_utils`` —
violation de la règle d'imports inward-only.

Cette factory ne dépend d'aucune brique cercle 3 (pas de ``click``,
pas de FastAPI). Les erreurs sont signalées via ``ValueError``, le CLI
les retraduit en ``click.BadParameter`` et l'API web les convertit en
warning utilisateur.

Discipline : ne pas importer ``click`` ici, sous peine de remonter une
dépendance cercle 3 dans cercle 2.
"""

from __future__ import annotations

from picarones.evaluation.engines.base import BaseOCREngine


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
    from picarones.engines.tesseract import TesseractEngine

    if engine_name in {"tesseract", "tess"}:
        return TesseractEngine(config={"lang": lang, "psm": psm})

    try:
        from picarones.engines.pero_ocr import PeroOCREngine

        if engine_name in {"pero_ocr", "pero"}:
            return PeroOCREngine(config={"name": "pero_ocr"})
    except ImportError:
        pass

    raise ValueError(
        f"Moteur inconnu ou non disponible : '{engine_name}'. "
        "Moteurs supportés : tesseract, pero_ocr"
    )


__all__ = ["engine_from_name"]
