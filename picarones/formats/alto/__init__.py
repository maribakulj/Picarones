"""Format ALTO XML 4.x (et v2/v3 tolérés).

Sprint A14-S9 livre :

- ``types.py`` — ``AltoDocument``, ``AltoPage``, ``AltoTextBlock``,
  ``AltoLine``, ``AltoString``, ``AltoBBox``.  Frozen pydantic.
- ``parser.py`` — ``parse_alto(xml_bytes)`` détection auto v2/v3/v4
  via le namespace du root.  Sécurité ``defusedxml``.
- ``writer.py`` — ``write_alto(doc, version="v4", pretty=False)``
  sortie déterministe (round-trip byte-stable avec ``parser``).
- ``projector.py`` — ``alto_document_to_text(doc)`` (helper) +
  ``AltoToText`` (projecteur conforme au protocole S5).  Gestion
  césure ``HypPart1``/``HypPart2``.

Anti-sur-ingénierie
-------------------
- Validator XSD reporté quand un caller en aura concrètement besoin
  (la plupart des outils consommateurs acceptent un ALTO bien formé
  sans validation stricte).
- ``Illustration``, ``ComposedBlock``, ``GraphicalElement``,
  ``StyleRefs``, ``ProcessingStep`` : non préservés au round-trip
  pour S9.
"""

from __future__ import annotations

from picarones.formats.alto.parser import AltoParseError, parse_alto
from picarones.formats.alto.types import (
    AltoBBox,
    AltoDocument,
    AltoLine,
    AltoPage,
    AltoString,
    AltoTextBlock,
)
from picarones.formats.alto.writer import write_alto

# S13 — les projecteurs ``alto_document_to_text`` et ``AltoToText``
# vivent désormais dans ``picarones.evaluation.projectors.alto``
# (la projection est conceptuellement un composant d'évaluation,
# pas un format).  Importer depuis le nouveau chemin :
#
#     from picarones.evaluation.projectors import (
#         AltoToText, alto_document_to_text,
#     )

__all__ = [
    # Types
    "AltoBBox",
    "AltoString",
    "AltoLine",
    "AltoTextBlock",
    "AltoPage",
    "AltoDocument",
    # Parser / Writer
    "parse_alto",
    "AltoParseError",
    "write_alto",
]
