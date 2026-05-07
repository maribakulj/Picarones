"""Importeurs de corpus depuis sources distantes.

Importeurs livrés ici (legacy, en cours de retrait) :

- :mod:`_http`         — helpers HTTP partagés (validate_http_url, download_url)
- :mod:`iiif`          — manifestes IIIF v2/v3 (Bodleian, BnF, Vatican…)
- :mod:`gallica`       — BnF Gallica (SRU + IIIF + OCR brut)
- :mod:`escriptorium`  — projets eScriptorium ⚠ **expérimental**

Importeurs migrés vers :mod:`picarones.adapters.corpus` (Lot I) :

- ``htr_united``        → :mod:`picarones.adapters.corpus.htr_united`
- ``huggingface``       → :mod:`picarones.adapters.corpus.huggingface`
  ⚠ **expérimental**
- ``_fallback_log``     → :mod:`picarones.adapters.corpus._fallback_log`

L'API publique de ce package re-expose ces modules canoniques pour
préserver la rétrocompat (``from picarones.extras.importers import
HuggingFaceDataset, HTRUnitedEntry, …``).
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
from picarones.adapters.corpus._fallback_log import (
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
