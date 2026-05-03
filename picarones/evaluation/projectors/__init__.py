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

from picarones.evaluation.projectors.base import ProjectionReport, Projector

__all__ = ["Projector", "ProjectionReport"]
