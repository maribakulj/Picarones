"""Structures internes PAGE XML — Sprint A14-S9.

Représentation typée et immuable d'un document PAGE XML (PRIMA /
Transkribus / eScriptorium).  Symétrique de ``formats.alto.types``
mais avec les conventions PAGE :

- ``Coords`` au lieu de ``HPOS/VPOS/WIDTH/HEIGHT`` — chaîne de points
  ``"x1,y1 x2,y2 ..."`` représentant un polygone.
- ``Baseline`` (optionnel) — ligne médiane horizontale typique des
  manuscrits.
- ``TextEquiv > Unicode`` au lieu de ``CONTENT`` ALTO.

Anti-sur-ingénierie
-------------------
- Pas de support des ``Word``/``Glyph`` PAGE (granularité plus fine
  que la ligne) pour S9 — la plupart des outils PAGE patrimoniaux
  utilisent la granularité ``TextLine``.  Un ``Word`` séparé peut
  être ajouté quand un caller en aura besoin.
- Coordonnées stockées en string brut (``points``).  Le caller qui
  veut une bbox calculée appelle ``points_to_bbox()`` du parser.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PageTextLine(BaseModel):
    """Une ligne PAGE (élément ``<TextLine>``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str | None = Field(default=None, max_length=128)
    coords: str | None = Field(default=None, max_length=4096)
    """Polygone en format PAGE : ``"x1,y1 x2,y2 x3,y3 ..."``."""
    baseline: str | None = Field(default=None, max_length=2048)
    """Polyline baseline (optionnelle, typique HTR)."""
    text: str = ""
    """Texte de la ligne extrait de ``TextEquiv > Unicode``."""


class PageTextRegion(BaseModel):
    """Région de texte PAGE (élément ``<TextRegion>``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str | None = Field(default=None, max_length=128)
    coords: str | None = Field(default=None, max_length=4096)
    region_type: str | None = Field(default=None, max_length=64)
    """Type sémantique PAGE : ``"paragraph"``, ``"heading"``,
    ``"caption"``, ``"footnote"``, etc.  Préservé tel quel sans
    enum (les valeurs PRIMA peuvent être étendues)."""
    text_lines: tuple[PageTextLine, ...] = Field(default_factory=tuple)


class PagePage(BaseModel):
    """Une page PAGE (élément ``<Page>``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    image_filename: str | None = Field(default=None, max_length=512)
    image_width: int | None = Field(default=None, ge=0)
    image_height: int | None = Field(default=None, ge=0)
    text_regions: tuple[PageTextRegion, ...] = Field(default_factory=tuple)


class PageDocument(BaseModel):
    """Document PAGE XML complet (peut contenir une seule page)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pages: tuple[PagePage, ...] = Field(default_factory=tuple)
    source_namespace: str | None = Field(default=None, max_length=256)
    """Namespace détecté au parsing (ex ``2019-07-15``, ``2013-07-15``)."""


__all__ = [
    "PageTextLine",
    "PageTextRegion",
    "PagePage",
    "PageDocument",
]
