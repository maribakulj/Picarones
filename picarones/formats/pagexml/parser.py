"""Parser PAGE XML tolérant — Sprint A14-S9.

Détection auto du namespace PRIMA (plusieurs versions co-existent
dans la nature : ``2010-03-19``, ``2013-07-15``, ``2017-07-15``,
``2019-07-15``).  Utilise ``defusedxml`` pour la sécurité XXE.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import defusedxml.ElementTree as _SafeET

from picarones.domain.errors import PicaronesError
from picarones.formats.pagexml.types import (
    PageDocument,
    PagePage,
    PageTextLine,
    PageTextRegion,
)

logger = logging.getLogger(__name__)


class PageParseError(PicaronesError):
    """PAGE XML non parsable."""


_NS_RE = re.compile(r"^\{([^}]*)\}")
_LOCAL_NAME_RE = re.compile(r"\{[^}]*\}")


def _local(tag: str) -> str:
    return _LOCAL_NAME_RE.sub("", tag)


def _detect_namespace(root_tag: str) -> str | None:
    m = _NS_RE.match(root_tag)
    return m.group(1) if m else None


def _extract_unicode(elem: Any) -> str:
    """Cherche le premier ``<Unicode>`` descendant et retourne son
    texte, ou ``""`` si absent.

    PAGE XML stocke le texte dans ``<TextEquiv><Unicode>...</Unicode></TextEquiv>``.
    Plusieurs ``TextEquiv`` peuvent coexister (variantes d'OCR) —
    on prend la première.
    """
    for child in elem.iter():
        if _local(child.tag) == "Unicode":
            return (child.text or "").strip()
    return ""


def _parse_coords(elem: Any) -> str | None:
    """Cherche le premier ``<Coords points="...">`` enfant direct."""
    for child in elem:
        if _local(child.tag) == "Coords":
            return child.attrib.get("points")
    return None


def _parse_baseline(elem: Any) -> str | None:
    for child in elem:
        if _local(child.tag) == "Baseline":
            return child.attrib.get("points")
    return None


def _parse_text_line(elem: Any) -> PageTextLine:
    return PageTextLine(
        id=elem.attrib.get("id"),
        coords=_parse_coords(elem),
        baseline=_parse_baseline(elem),
        text=_extract_unicode(elem),
    )


def _parse_text_region(elem: Any) -> PageTextRegion:
    lines: list[PageTextLine] = []
    for child in elem:
        if _local(child.tag) == "TextLine":
            lines.append(_parse_text_line(child))
    return PageTextRegion(
        id=elem.attrib.get("id"),
        coords=_parse_coords(elem),
        region_type=elem.attrib.get("type"),
        text_lines=tuple(lines),
    )


def _parse_int_attr(elem: Any, name: str) -> int | None:
    raw = elem.attrib.get(name)
    if raw is None:
        return None
    try:
        return int(float(raw))
    except (ValueError, TypeError):
        return None


def _parse_page(elem: Any) -> PagePage:
    regions: list[PageTextRegion] = []
    for child in elem:
        if _local(child.tag) == "TextRegion":
            regions.append(_parse_text_region(child))
    return PagePage(
        image_filename=elem.attrib.get("imageFilename"),
        image_width=_parse_int_attr(elem, "imageWidth"),
        image_height=_parse_int_attr(elem, "imageHeight"),
        text_regions=tuple(regions),
    )


def parse_pagexml(xml: bytes | str) -> PageDocument:
    """Parse un document PAGE XML et retourne la structure interne.

    Raises
    ------
    PageParseError
        XML mal formé, défense XXE, ou root absent.
    """
    if isinstance(xml, str):
        xml_bytes = xml.encode("utf-8")
    else:
        xml_bytes = xml
    if not xml_bytes.strip():
        raise PageParseError("PAGE XML vide.")
    try:
        root = _SafeET.fromstring(xml_bytes)
    except Exception as exc:  # noqa: BLE001
        raise PageParseError(f"XML invalide ou XXE bloqué : {exc}") from exc

    if root is None:
        raise PageParseError("PAGE sans root element.")

    ns = _detect_namespace(root.tag)
    pages: list[PagePage] = []
    for elem in root.iter():
        if _local(elem.tag) == "Page":
            pages.append(_parse_page(elem))

    return PageDocument(pages=tuple(pages), source_namespace=ns)


__all__ = ["parse_pagexml", "PageParseError"]
