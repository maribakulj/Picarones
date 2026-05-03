"""Normalisation et manipulation de texte.

Cible Sprint S9 — déplacement de ``picarones.measurements.normalization``
sans modification de logique.

Modules cibles :

- ``normalization.py`` — 11 profils (nfc, caseless, minimal,
  medieval_french, early_modern_french, medieval_latin,
  medieval_english, early_modern_english, secretary_hand,
  sans_ponctuation, sans_apostrophes).  Tables diplomatiques.
  Exclusion de caractères.

Règle : ce module ne fait **pas** d'extraction depuis ALTO/PAGE
(c'est le rôle des projecteurs).  Il prend une chaîne en entrée,
applique un profil, retourne une chaîne.
"""

from __future__ import annotations

__all__: list[str] = []
