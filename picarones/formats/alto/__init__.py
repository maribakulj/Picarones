"""Format ALTO XML 4.x.

Cible Sprint S9 :

- ``parser.py`` — détection auto namespace (v2/v3/v4), parsing
  tolérant.  Retourne une structure interne (lignes, mots,
  coordonnées, IDs).
- ``writer.py`` — structure interne → XML déterministe (même
  entrée, même bytes).
- ``validator.py`` — conformité au schéma XSD ALTO.
- ``projector.py`` — extraction texte par ordre de lecture,
  extraction lignes, extraction mots avec coordonnées.

Règle de sécurité : tout parsing XML passe par ``defusedxml`` (pas
``lxml`` direct sur du XML utilisateur), pour bloquer XXE et
Billion Laughs.
"""

from __future__ import annotations

__all__: list[str] = []
