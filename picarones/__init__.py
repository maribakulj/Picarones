"""
Picarones — Plateforme de comparaison de moteurs OCR pour documents patrimoniaux.

Licence Apache 2.0.
"""

try:
    from importlib.metadata import version as _get_version
    __version__ = _get_version("picarones")
except Exception:
    __version__ = "1.0.0"

__author__ = "Picarones contributors"
