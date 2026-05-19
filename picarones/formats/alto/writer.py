"""Writer ALTO XML déterministe — Sprint A14-S9.

Sérialise un ``AltoDocument`` en bytes ALTO XML.  Sortie
déterministe : même document → mêmes octets exacts (utile pour le
cache d'artefacts du S7 et les tests de round-trip).

Format de sortie
----------------
Par défaut, le writer sort un ALTO **v4** (le plus récent et le
plus expressif), même si le document a été parsé depuis v2/v3.  Le
caller peut forcer une version cible avec ``write_alto(doc,
version="v3")``.

Anti-sur-ingénierie
-------------------
- Pas de support des ``StyleRefs``, ``ProcessingStep``, ``OCRProcessing``,
  ``Description`` pour S9.  Le writer sort une structure minimale
  (``alto > Layout > Page > PrintSpace > TextBlock > TextLine > String``)
  qui passe la validation des outils consommateurs courants
  (Mirador, IIIF Universal Viewer, Aletheia).
- Pas d'XSL preprocessing.  L'utilisateur qui veut un ALTO
  enrichi écrira un wrapper.
"""

from __future__ import annotations

from typing import cast
from xml.etree import ElementTree as ET

from picarones.formats.alto.types import (
    AltoBBox,
    AltoDocument,
    AltoLine,
    AltoPage,
    AltoString,
    AltoTextBlock,
)


_NAMESPACE_BY_VERSION: dict[str, str] = {
    "v2": "http://www.loc.gov/standards/alto/ns-v2#",
    "v3": "http://www.loc.gov/standards/alto/ns-v3#",
    "v4": "http://www.loc.gov/standards/alto/ns-v4#",
}


def _set_bbox_attrs(elem: ET.Element, bbox: AltoBBox | None) -> None:
    if bbox is None:
        return
    elem.set("HPOS", str(bbox.hpos))
    elem.set("VPOS", str(bbox.vpos))
    elem.set("WIDTH", str(bbox.width))
    elem.set("HEIGHT", str(bbox.height))


def _set_optional(elem: ET.Element, name: str, value: str | None) -> None:
    if value is not None:
        elem.set(name, value)


def _build_string(parent: ET.Element, ns: str, s: AltoString) -> None:
    elem = ET.SubElement(parent, f"{{{ns}}}String" if ns else "String")
    elem.set("CONTENT", s.content)
    _set_optional(elem, "ID", s.id)
    _set_bbox_attrs(elem, s.bbox)
    _set_optional(elem, "SUBS_TYPE", s.subs_type)
    _set_optional(elem, "SUBS_CONTENT", s.subs_content)


def _build_line(parent: ET.Element, ns: str, line: AltoLine) -> None:
    elem = ET.SubElement(parent, f"{{{ns}}}TextLine" if ns else "TextLine")
    _set_optional(elem, "ID", line.id)
    _set_bbox_attrs(elem, line.bbox)
    for s in line.strings:
        _build_string(elem, ns, s)


def _build_block(parent: ET.Element, ns: str, block: AltoTextBlock) -> None:
    elem = ET.SubElement(parent, f"{{{ns}}}TextBlock" if ns else "TextBlock")
    _set_optional(elem, "ID", block.id)
    _set_bbox_attrs(elem, block.bbox)
    for line in block.lines:
        _build_line(elem, ns, line)


def _build_page(parent: ET.Element, ns: str, page: AltoPage) -> None:
    elem = ET.SubElement(parent, f"{{{ns}}}Page" if ns else "Page")
    _set_optional(elem, "ID", page.id)
    if page.width is not None:
        elem.set("WIDTH", str(page.width))
    if page.height is not None:
        elem.set("HEIGHT", str(page.height))
    print_space = ET.SubElement(
        elem, f"{{{ns}}}PrintSpace" if ns else "PrintSpace",
    )
    for block in page.blocks:
        _build_block(print_space, ns, block)


def write_alto(
    document: AltoDocument,
    *,
    version: str = "v4",
    pretty: bool = False,
) -> bytes:
    """Sérialise un ``AltoDocument`` en bytes ALTO XML.

    Parameters
    ----------
    document:
        Document à sérialiser.
    version:
        Version ALTO cible.  ``"v2"`` / ``"v3"`` / ``"v4"`` ou
        ``"none"`` (sans namespace).  Défaut : ``"v4"``.
    pretty:
        Si ``True``, indente la sortie pour la lisibilité humaine.
        ``False`` (défaut) produit une sortie compacte byte-déterministe.

    Returns
    -------
    bytes
        XML encodé en UTF-8 avec déclaration XML.
    """
    if version not in (*_NAMESPACE_BY_VERSION, "none"):
        from picarones.domain.errors import PicaronesError
        raise PicaronesError(
            f"version ALTO invalide : {version!r}.  "
            f"Acceptées : {sorted(_NAMESPACE_BY_VERSION)} + 'none'."
        )
    ns = _NAMESPACE_BY_VERSION.get(version, "")
    if ns:
        ET.register_namespace("", ns)
        root = ET.Element(f"{{{ns}}}alto")
    else:
        root = ET.Element("alto")

    layout = ET.SubElement(root, f"{{{ns}}}Layout" if ns else "Layout")
    for page in document.pages:
        _build_page(layout, ns, page)

    if pretty:
        ET.indent(root, space="  ")

    body = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return cast(bytes, body)


__all__ = ["write_alto"]
