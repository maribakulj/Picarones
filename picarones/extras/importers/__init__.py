"""Importeurs de corpus depuis sources distantes (Cercle 3).

Importeurs livrés
-----------------
- :mod:`_http`         — helpers HTTP partagés (validate_http_url, download_url)
- :mod:`iiif`          — manifestes IIIF v2/v3 (Bodleian, BnF, Vatican…)
- :mod:`htr_united`    — datasets HTR-United (CC0, GitHub)
- :mod:`gallica`       — BnF Gallica (SRU + IIIF + OCR brut)
- :mod:`huggingface`   — datasets HuggingFace ⚠ **expérimental**
- :mod:`escriptorium`  — projets eScriptorium ⚠ **expérimental**

Modules expérimentaux
---------------------
``huggingface`` et ``escriptorium`` émettent un ``UserWarning`` à
l'import. Ils sont fonctionnellement présents mais leur usage en
production n'est pas garanti — l'API HuggingFace Datasets évolue
fréquemment et eScriptorium n'a qu'un test isolé.
"""

from picarones.extras.importers.iiif import IIIFImporter, import_iiif_manifest
from picarones.extras.importers.gallica import (
    GallicaClient,
    GallicaRecord,
    search_gallica,
    import_gallica_document,
)
from picarones.extras.importers.escriptorium import (
    EScriptoriumClient,
    EScriptoriumProject,
    EScriptoriumDocument,
    connect_escriptorium,
)
from picarones.extras.importers._fallback_log import (
    consume_fallback_log,
    peek_fallback_log,
    record_fallback,
    reset_fallback_log,
)

__all__ = [
    "IIIFImporter",
    "import_iiif_manifest",
    "GallicaClient",
    "GallicaRecord",
    "search_gallica",
    "import_gallica_document",
    "EScriptoriumClient",
    "EScriptoriumProject",
    "EScriptoriumDocument",
    "connect_escriptorium",
    # Sprint A3 (B-3) — journal des fallbacks d'importer
    "record_fallback",
    "consume_fallback_log",
    "peek_fallback_log",
    "reset_fallback_log",
]
