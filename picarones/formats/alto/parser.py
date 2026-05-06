"""Parser ALTO XML tolérant aux namespaces — Sprint A14-S9.

Détection auto de la version ALTO (v2/v3/v4) via le namespace du
root element.  Tolérant aux variantes : un ALTO sans namespace est
accepté ; un ALTO avec déclaration partielle (``<alto>`` sans xmlns)
aussi.

Sécurité
--------
Utilise ``defusedxml.ElementTree`` pour bloquer XXE, Billion Laughs,
DTD retrieval — un ALTO peut venir d'un module tiers ou d'un
utilisateur web non authentifié.

Anti-sur-ingénierie
-------------------
- Pas de validation de schéma XSD pour S9 (le ``validator.py`` du
  plan est reporté quand un caller en aura concrètement besoin —
  la plupart des outils accepteront un ALTO bien formé même sans
  validation stricte).
- Les éléments non reconnus (``Illustration``, ``ComposedBlock``,
  ``GraphicalElement``) sont silencieusement ignorés par le parser.
- ``HypPart1`` / ``HypPart2`` sont préservés au niveau ``AltoString``
  (le projecteur les utilise pour la césure).
"""

from __future__ import annotations

import logging
import re
from typing import Any

import defusedxml.ElementTree as _SafeET

from picarones.domain.errors import PicaronesError
from picarones.formats.alto.types import (
    AltoBBox,
    AltoDocument,
    AltoLine,
    AltoPage,
    AltoString,
    AltoTextBlock,
)

logger = logging.getLogger(__name__)


class AltoParseError(PicaronesError):
    """ALTO non parsable (XML invalide, XXE bloqué, root absent)."""


_NS_RE = re.compile(r"^\{([^}]*)\}")
_LOCAL_NAME_RE = re.compile(r"\{[^}]*\}")


def _local(tag: str) -> str:
    """Retire le préfixe namespace pour ne garder que le nom local."""
    return _LOCAL_NAME_RE.sub("", tag)


def _detect_version(root_tag: str) -> str | None:
    """Détecte la version ALTO depuis le tag du root.

    - Pas de namespace → ``"none"``.
    - ``http://www.loc.gov/standards/alto/ns-v2#`` → ``"v2"``.
    - ``http://www.loc.gov/standards/alto/ns-v3#`` → ``"v3"``.
    - ``http://www.loc.gov/standards/alto/ns-v4#`` → ``"v4"``.
    - Autre namespace → ``None`` (inconnu).
    """
    m = _NS_RE.match(root_tag)
    if m is None:
        return "none"
    ns = m.group(1)
    if "ns-v2" in ns:
        return "v2"
    if "ns-v3" in ns:
        return "v3"
    if "ns-v4" in ns:
        return "v4"
    return None


def _parse_int_attr(elem: Any, name: str) -> int | None:
    """Parse un attribut entier optionnel.  Retourne ``None`` si
    absent ou invalide (au lieu de lever)."""
    raw = elem.attrib.get(name)
    if raw is None:
        return None
    try:
        # ALTO accepte des floats dans certains attributs (HPOS), on
        # tronque vers int.
        return int(float(raw))
    except (ValueError, TypeError):
        return None


def _parse_bbox(elem: Any) -> AltoBBox | None:
    """Construit un ``AltoBBox`` si les 4 attributs sont présents."""
    h = _parse_int_attr(elem, "HPOS")
    v = _parse_int_attr(elem, "VPOS")
    w = _parse_int_attr(elem, "WIDTH")
    height = _parse_int_attr(elem, "HEIGHT")
    if any(x is None for x in (h, v, w, height)):
        return None
    # Coordonnées négatives → certains ALTO mal formés ; on clip à 0.
    return AltoBBox(
        hpos=max(0, h or 0),
        vpos=max(0, v or 0),
        width=max(0, w or 0),
        height=max(0, height or 0),
    )


def _parse_string(elem: Any) -> AltoString:
    """Convertit un élément ``<String>`` en ``AltoString``."""
    return AltoString(
        content=elem.attrib.get("CONTENT", ""),
        id=elem.attrib.get("ID"),
        bbox=_parse_bbox(elem),
        subs_type=elem.attrib.get("SUBS_TYPE"),
        subs_content=elem.attrib.get("SUBS_CONTENT"),
    )


def _parse_line(elem: Any) -> AltoLine:
    """Convertit un élément ``<TextLine>`` en ``AltoLine``."""
    strings: list[AltoString] = []
    for child in elem:
        if _local(child.tag) == "String":
            strings.append(_parse_string(child))
    return AltoLine(
        id=elem.attrib.get("ID"),
        bbox=_parse_bbox(elem),
        strings=tuple(strings),
    )


def _parse_block(elem: Any) -> AltoTextBlock:
    """Convertit un élément ``<TextBlock>`` en ``AltoTextBlock``."""
    lines: list[AltoLine] = []
    for child in elem.iter():
        if _local(child.tag) == "TextLine":
            lines.append(_parse_line(child))
    return AltoTextBlock(
        id=elem.attrib.get("ID"),
        bbox=_parse_bbox(elem),
        lines=tuple(lines),
    )


def _parse_page(elem: Any) -> AltoPage:
    """Convertit un élément ``<Page>`` en ``AltoPage``."""
    blocks: list[AltoTextBlock] = []
    seen_block_ids: set[int] = set()
    for child in elem.iter():
        if _local(child.tag) != "TextBlock":
            continue
        # Évite la duplication quand un TextBlock est imbriqué dans un
        # ComposedBlock — on retourne le bloc une seule fois (par id python).
        marker = id(child)
        if marker in seen_block_ids:
            continue
        seen_block_ids.add(marker)
        blocks.append(_parse_block(child))
    return AltoPage(
        id=elem.attrib.get("ID"),
        width=_parse_int_attr(elem, "WIDTH"),
        height=_parse_int_attr(elem, "HEIGHT"),
        blocks=tuple(blocks),
    )


def parse_alto(xml: bytes | str) -> AltoDocument:
    """Parse un document ALTO et retourne sa structure interne.

    Parameters
    ----------
    xml:
        Bytes ou string XML.  Encodage détecté automatiquement par
        ``defusedxml`` (à partir de la déclaration ``<?xml encoding="..."?>``
        ou du BOM).

    Returns
    -------
    AltoDocument
        Document avec ``source_version`` indiquant la version
        détectée et ``pages`` contenant la hiérarchie complète.

    Raises
    ------
    AltoParseError
        XML mal formé, défense XXE déclenchée, ou root absent.
    """
    if isinstance(xml, str):
        xml_bytes = xml.encode("utf-8")
    else:
        xml_bytes = xml
    if not xml_bytes.strip():
        raise AltoParseError("ALTO vide.")
    try:
        root = _SafeET.fromstring(xml_bytes)
    except Exception as exc:  # noqa: BLE001
        raise AltoParseError(f"XML invalide ou XXE bloqué : {exc}") from exc

    if root is None:
        raise AltoParseError("ALTO sans root element.")

    version = _detect_version(root.tag)
    if _local(root.tag) != "alto":
        # Tolérant : on cherche un éventuel <alto> imbriqué (cas d'un
        # METS qui embarque l'ALTO dans un mdRef).  Sinon on prend le
        # root tel quel — peut-être qu'un caller passe directement
        # un fragment <Page>.
        for elem in root.iter():
            if _local(elem.tag) == "alto":
                root = elem
                version = _detect_version(elem.tag)
                break

    pages: list[AltoPage] = []
    for elem in root.iter():
        if _local(elem.tag) == "Page":
            pages.append(_parse_page(elem))

    return AltoDocument(pages=tuple(pages), source_version=version)


__all__ = ["parse_alto", "AltoParseError"]
