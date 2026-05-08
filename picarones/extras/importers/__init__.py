"""Shim de compatibilité — importeurs de corpus depuis sources distantes.

Phase 8 (mai 2026) — les implémentations canoniques vivent désormais
dans :mod:`picarones.adapters.corpus`.  Ce package re-expose les
symboles publics pour préserver la rétrocompat le temps que les
callers externes migrent ; il sera supprimé en 2.0.

Mapping appliqué :

- ``iiif``          → :mod:`picarones.adapters.corpus.iiif`
- ``gallica``       → :mod:`picarones.adapters.corpus.gallica`
- ``escriptorium``  → :mod:`picarones.adapters.corpus.escriptorium`
- ``_http``         → :mod:`picarones.adapters.corpus._http`
- ``htr_united``    → :mod:`picarones.adapters.corpus.htr_united` (déjà migré au Lot I)
- ``huggingface``   → :mod:`picarones.adapters.corpus.huggingface` (déjà migré au Lot I)
- ``_fallback_log`` → :mod:`picarones.adapters.corpus._fallback_log` (déjà migré au Lot I)
"""

from picarones.adapters.corpus.iiif import IIIFImporter, import_iiif_manifest
from picarones.adapters.corpus.gallica import (
    GallicaClient,
    GallicaRecord,
    search_gallica,
    import_gallica_document,
)
from picarones.adapters.corpus.escriptorium import (
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
    "record_fallback",
    "consume_fallback_log",
    "peek_fallback_log",
    "reset_fallback_log",
]
