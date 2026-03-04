"""Importeurs de corpus depuis des sources distantes (IIIF, HuggingFace, HTR-United…)."""

from picarones.importers.iiif import IIIFImporter, import_iiif_manifest

__all__ = ["IIIFImporter", "import_iiif_manifest"]
