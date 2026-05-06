"""Structures internes ALTO — Sprint A14-S9.

Représentation **typée et immuable** d'un document ALTO XML pour
manipulation, projection, et round-trip parser/writer.  Indépendante
du namespace source (v2/v3/v4) — le parser normalise.

Hiérarchie ALTO simplifiée :

::

    AltoDocument
      └─ AltoPage  (1..N)
           └─ AltoTextBlock  (0..N)
                └─ AltoLine  (0..N)
                     └─ AltoString  (0..N)

Les coordonnées (HPOS, VPOS, WIDTH, HEIGHT) sont **optionnelles**.
Un ALTO produit par certains VLM peut omettre les bbox (texte sans
coordonnées) — on accepte au parsing et le projecteur ALTO→texte
fonctionne quand même.

Anti-sur-ingénierie
-------------------
Pas de support des éléments rares pour S9 :
- ``Composed Block`` (regroupement de blocks) — projeté en blocks plats.
- ``Illustration`` / ``GraphicalElement`` — ignorés à l'extraction texte.
- ``StyleRefs`` / typographie — non préservés par le writer.
- ``Hyphenation`` côté ``HypPart1`` / ``HypPart2`` est par contre
  géré par le projector (cf. ``projector.py``).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AltoBBox(BaseModel):
    """Boîte englobante optionnelle (coordonnées en pixels)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hpos: int = Field(ge=0)
    vpos: int = Field(ge=0)
    width: int = Field(ge=0)
    height: int = Field(ge=0)


class AltoString(BaseModel):
    """Un mot ALTO (élément ``<String>``).

    Attributs ALTO mappés :
    - ``CONTENT`` → ``content``
    - ``ID`` → ``id``
    - ``HPOS``/``VPOS``/``WIDTH``/``HEIGHT`` → ``bbox``
    - ``SUBS_TYPE`` → ``subs_type`` (``"HypPart1"`` / ``"HypPart2"``).
      Le projecteur l'utilise pour gérer la césure de fin de ligne.
    - ``SUBS_CONTENT`` → ``subs_content`` (mot complet quand césuré).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    content: str
    id: str | None = Field(default=None, max_length=128)
    bbox: AltoBBox | None = None
    subs_type: str | None = Field(default=None, pattern=r"^(HypPart1|HypPart2)$")
    subs_content: str | None = None


class AltoLine(BaseModel):
    """Une ligne ALTO (élément ``<TextLine>``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str | None = Field(default=None, max_length=128)
    bbox: AltoBBox | None = None
    strings: tuple[AltoString, ...] = Field(default_factory=tuple)
    """Mots de la ligne, ordre de lecture naturel (gauche → droite)."""


class AltoTextBlock(BaseModel):
    """Un bloc de texte ALTO (élément ``<TextBlock>``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str | None = Field(default=None, max_length=128)
    bbox: AltoBBox | None = None
    lines: tuple[AltoLine, ...] = Field(default_factory=tuple)


class AltoPage(BaseModel):
    """Une page ALTO (élément ``<Page>``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str | None = Field(default=None, max_length=128)
    width: int | None = Field(default=None, ge=0)
    """Largeur physique en pixels (``WIDTH``)."""
    height: int | None = Field(default=None, ge=0)
    """Hauteur physique en pixels (``HEIGHT``)."""
    blocks: tuple[AltoTextBlock, ...] = Field(default_factory=tuple)


class AltoDocument(BaseModel):
    """Document ALTO complet.

    Conserve la version source au parsing pour permettre au writer
    de re-sortir dans le même namespace si demandé.  Par défaut,
    le writer sort en v4 (le plus récent et le plus expressif).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pages: tuple[AltoPage, ...] = Field(default_factory=tuple)
    source_version: str | None = Field(default=None, max_length=8)
    """Version détectée au parsing : ``"v2"`` / ``"v3"`` / ``"v4"`` /
    ``"none"`` (sans namespace) / ``None`` (inconnue)."""


__all__ = [
    "AltoBBox",
    "AltoString",
    "AltoLine",
    "AltoTextBlock",
    "AltoPage",
    "AltoDocument",
]
