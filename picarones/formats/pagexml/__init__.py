"""Format PAGE XML (PRIMA / Transkribus).

Sprint A14-S9 livre :

- ``types.py`` — ``PageDocument``, ``PagePage``, ``PageTextRegion``,
  ``PageTextLine``.  Frozen pydantic.
- ``parser.py`` — ``parse_pagexml(xml_bytes)`` tolérant aux versions
  de namespace PRIMA.  Sécurité ``defusedxml``.
- ``projector.py`` — ``page_document_to_text(doc)`` + ``PageToText``.

Writer reporté post-livraison (les outils PAGE produisent
typiquement le format à partir d'un éditeur — le besoin de re-sortir
est plus rare que pour ALTO).
"""

from __future__ import annotations

from picarones.formats.pagexml.parser import PageParseError, parse_pagexml
from picarones.formats.pagexml.projector import PageToText, page_document_to_text
from picarones.formats.pagexml.types import (
    PageDocument,
    PagePage,
    PageTextLine,
    PageTextRegion,
)

__all__ = [
    "PageTextLine",
    "PageTextRegion",
    "PagePage",
    "PageDocument",
    "parse_pagexml",
    "PageParseError",
    "page_document_to_text",
    "PageToText",
]
