"""Cercle 2 — Formats documentaires.

Parsers, writers et validateurs pour les formats d'entrée/sortie
patrimoniaux.  Tout le code XML / namespaces / parsing vit ici, à
l'écart du domain (qui ne connaît que des ``ArtifactType``) et de
``evaluation/`` (qui consomme des structures de données déjà
parsées).

Sous-packages :

- ``alto/`` — ALTO XML 4.x (Sprint S9).  Parser tolérant aux 3
  versions de namespace, writer déterministe, validator schéma.
- ``pagexml/`` — PAGE XML (PRIMA, transkribus).
- ``text/`` — normalisation texte (NFC, casefold, profils
  diplomatiques, exclusion de caractères).  Cible du déplacement
  de ``picarones.measurements.normalization`` au Sprint S9.

Règle d'import : ces modules peuvent importer ``lxml`` et
``defusedxml``.  Ils ne doivent **jamais** importer un moteur OCR
ou un calcul de métrique — ils opèrent sur des bytes / des chaînes,
pas sur des résultats d'OCR.
"""

from __future__ import annotations

__all__: list[str] = []
