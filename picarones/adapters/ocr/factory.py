"""Factory canonique : instancier un ``BaseOCRAdapter`` par nom court.

Sprint H.2.b du plan v2.0 — équivalent canonique de
``picarones.adapters.legacy_engines.factory.engine_from_name`` qui
retournait des ``BaseOCREngine`` (legacy, ``run(image_path) →
EngineResult``).  Cette factory retourne des ``BaseOCRAdapter``
(rewrite, ``StepExecutor`` Protocol, ``execute(inputs, params,
context) → dict[ArtifactType, Artifact]``).

Pourquoi ici
------------
Vit en couche 5 (``picarones.adapters.ocr``) plutôt qu'en
``app/`` parce que c'est de la logique de catalogue OCR — la CLI
(couche 8) et la web API (couche 8) la consomment toutes les deux.
Cette factory ne dépend d'aucune brique de couche supérieure
(pas de ``click``, pas de FastAPI).

Migration depuis le legacy
--------------------------
Code legacy ::

    from picarones.adapters.legacy_engines.factory import engine_from_name
    engine = engine_from_name("tesseract", lang="fra", psm=6)
    # engine est un BaseOCREngine, à wrapper via LegacyOCREngineExecutor
    # avant de pouvoir être consommé par PipelineExecutor.

Code canonique équivalent ::

    from picarones.adapters.ocr.factory import ocr_adapter_from_name
    adapter = ocr_adapter_from_name("tesseract", lang="fra", psm=6)
    # adapter est un BaseOCRAdapter — déjà un StepExecutor, peut
    # être directement enregistré dans un adapter_resolver et
    # consommé par PipelineExecutor sans wrapping.

Alias supportés
---------------
- ``tesseract`` / ``tess``
- ``pero_ocr`` / ``pero``
- ``mistral_ocr`` / ``mistral``
- ``google_vision`` / ``google`` / ``gv``
- ``azure_doc_intel`` / ``azure`` / ``adi``
- ``precomputed``
"""

from __future__ import annotations

from typing import Any

from picarones.adapters.ocr.base import BaseOCRAdapter

#: Mapping ``alias → nom canonique`` pour les noms abrégés.
_ALIASES: dict[str, str] = {
    "tess": "tesseract",
    "pero": "pero_ocr",
    "mistral": "mistral_ocr",
    "google": "google_vision",
    "gv": "google_vision",
    "azure": "azure_doc_intel",
    "adi": "azure_doc_intel",
}

#: Liste des noms canoniques supportés pour les messages d'erreur.
_SUPPORTED: tuple[str, ...] = (
    "tesseract",
    "pero_ocr",
    "mistral_ocr",
    "google_vision",
    "azure_doc_intel",
    "precomputed",
)


def ocr_adapter_from_name(
    name: str, **kwargs: Any,
) -> BaseOCRAdapter:
    """Instancie un ``BaseOCRAdapter`` canonique par son nom court.

    Parameters
    ----------
    name:
        Identifiant court du moteur (cf. liste des alias dans le
        docstring du module).  Insensible à la casse.
    **kwargs:
        Arguments propagés au constructeur de l'adapter cible.
        Les kwargs non reconnus par le constructeur lèveront un
        ``TypeError`` — c'est intentionnel, on ne masque pas les
        fautes de frappe.

    Returns
    -------
    BaseOCRAdapter
        Instance prête à être enregistrée dans un
        ``adapter_resolver`` et consommée par ``PipelineExecutor``.

    Raises
    ------
    ValueError
        Si ``name`` est inconnu, ou si l'adapter cible nécessite
        une dépendance optionnelle non installée (ex : Pero OCR
        sans ``pero-ocr``).  Le message d'erreur inclut la liste
        des moteurs effectivement supportés.

    Examples
    --------
    >>> adapter = ocr_adapter_from_name("tesseract", lang="fra")
    >>> adapter.name
    'tesseract'

    >>> adapter = ocr_adapter_from_name("tess")  # alias
    >>> adapter.name
    'tesseract'

    >>> adapter = ocr_adapter_from_name(
    ...     "precomputed", source_label="bnf_jean_zay",
    ... )
    >>> adapter.name
    'precomputed:bnf_jean_zay'
    """
    canonical = _ALIASES.get(name.lower(), name.lower())

    if canonical == "tesseract":
        from picarones.adapters.ocr.tesseract import TesseractAdapter
        return TesseractAdapter(**kwargs)

    if canonical == "pero_ocr":
        try:
            from picarones.adapters.ocr.pero_ocr import PeroOCRAdapter
        except ImportError as exc:
            raise ValueError(
                f"Adapter 'pero_ocr' indisponible : {exc}.  "
                "Installer la dépendance optionnelle ``pero-ocr``."
            ) from exc
        return PeroOCRAdapter(**kwargs)

    if canonical == "mistral_ocr":
        try:
            from picarones.adapters.ocr.mistral_ocr import MistralOCRAdapter
        except ImportError as exc:
            raise ValueError(
                f"Adapter 'mistral_ocr' indisponible : {exc}.  "
                "Installer la dépendance optionnelle ``mistralai``."
            ) from exc
        return MistralOCRAdapter(**kwargs)

    if canonical == "google_vision":
        try:
            from picarones.adapters.ocr.google_vision import (
                GoogleVisionAdapter,
            )
        except ImportError as exc:
            raise ValueError(
                f"Adapter 'google_vision' indisponible : {exc}.  "
                "Installer la dépendance optionnelle "
                "``google-cloud-vision``."
            ) from exc
        return GoogleVisionAdapter(**kwargs)

    if canonical == "azure_doc_intel":
        try:
            from picarones.adapters.ocr.azure_doc_intel import (
                AzureDocIntelAdapter,
            )
        except ImportError as exc:
            raise ValueError(
                f"Adapter 'azure_doc_intel' indisponible : {exc}.  "
                "Installer la dépendance optionnelle "
                "``azure-ai-formrecognizer``."
            ) from exc
        return AzureDocIntelAdapter(**kwargs)

    if canonical == "precomputed":
        from picarones.adapters.ocr.precomputed import (
            PrecomputedTextAdapter,
        )
        return PrecomputedTextAdapter(**kwargs)

    raise ValueError(
        f"Moteur OCR inconnu : {name!r}.  Valeurs supportées : "
        f"{', '.join(_SUPPORTED)} (alias : "
        f"{', '.join(sorted(_ALIASES))}).",
    )


__all__ = ["ocr_adapter_from_name"]
