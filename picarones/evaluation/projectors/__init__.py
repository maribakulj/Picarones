"""Projecteurs — transformations entre types d'artefacts.

Un projecteur convertit un artefact d'un type vers un autre, en
documentant explicitement ce qu'il **perd** au passage via un
``ProjectionReport``.

Exemples (à venir Sprint S14) :

- ``AltoToText`` — extraction du texte par ordre de lecture.
  Pertes : coordonnées, blocs, IDs de ligne, hiérarchie.
- ``PageToText`` — équivalent pour PAGE XML.
- ``CanonicalDocumentToText`` — ``markdown`` ou JSON canonique
  vers texte brut.
- ``MarkdownToText`` — supprime les balises markdown.

Règle d'or : un projecteur est **non-symétrique** par défaut.  On
peut projeter ALTO → texte (perte), pas l'inverse.  La
reconstruction inverse (texte → ALTO) est un module de pipeline,
pas un projecteur.
"""

from __future__ import annotations

from picarones.evaluation.projectors.alto import (
    AltoToText,
    alto_document_to_text,
)
from picarones.evaluation.projectors.base import ProjectionReport, Projector
from picarones.evaluation.projectors.canonical import (
    CanonicalToText,
    canonical_payload_to_text,
    markdown_to_text,
)
from picarones.evaluation.projectors.pagexml import (
    PageToText,
    page_document_to_text,
)
from picarones.evaluation.projectors.registry import (
    ProjectorNotFoundError,
    ProjectorRegistrationError,
    ProjectorRegistry,
)

__all__ = [
    # Protocol + report
    "Projector",
    "ProjectionReport",
    # Registry
    "ProjectorRegistry",
    "ProjectorRegistrationError",
    "ProjectorNotFoundError",
    # Concrete projectors (S13)
    "AltoToText",
    "alto_document_to_text",
    "PageToText",
    "page_document_to_text",
    # Canonical (S14)
    "CanonicalToText",
    "canonical_payload_to_text",
    "markdown_to_text",
]
