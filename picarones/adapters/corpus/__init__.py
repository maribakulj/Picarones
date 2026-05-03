"""Adaptateurs corpus — Sprint S11.

Cible : déplacement de ``picarones.extras.importers.{iiif,gallica,
htr_united,huggingface,escriptorium}``.  Un corpus adapter charge
un corpus depuis une source distante (manifeste IIIF, dataset HF,
catalogue HTR-United, eScriptorium, ZIP utilisateur) et retourne
un ``CorpusSpec`` (références aux images + GT par niveau).

Règle : pas de pré-calcul.  Pas d'OCR.  Le corpus adapter ne sait
que **nommer et localiser** les paires (image, GT).  L'exécution
des moteurs est faite plus tard par le pipeline executor.
"""

from __future__ import annotations

__all__: list[str] = []
