"""Importeurs de corpus depuis des sources distantes (IIIF, HuggingFace, HTR-United, Gallica, eScriptorium…)."""

from picarones.importers.iiif import IIIFImporter, import_iiif_manifest
from picarones.importers.gallica import GallicaClient, GallicaRecord, search_gallica, import_gallica_document
from picarones.importers.escriptorium import EScriptoriumClient, EScriptoriumProject, EScriptoriumDocument, connect_escriptorium

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
]
