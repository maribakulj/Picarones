"""Adaptateurs corpus — Sprint S11 + Phase 8.

Charge un corpus depuis une source distante (manifeste IIIF, dataset HF,
catalogue HTR-United, eScriptorium, Gallica) et retourne un objet
``Corpus`` (références aux images + GT par niveau).

Règle : pas de pré-calcul.  Pas d'OCR.  Le corpus adapter ne sait
que **nommer et localiser** les paires (image, GT).  L'exécution
des moteurs est faite plus tard par le pipeline executor.

Modules disponibles
-------------------
- :mod:`iiif`           — manifestes IIIF v2/v3 (Bodleian, BnF, Vatican…)
- :mod:`gallica`        — BnF Gallica (SRU + IIIF + OCR brut)
- :mod:`escriptorium`   — projets eScriptorium ⚠ **expérimental**
- :mod:`htr_united`     — catalogue HTR-United
- :mod:`huggingface`    — datasets HuggingFace ⚠ **expérimental**
- :mod:`_http`          — helpers HTTP partagés (validate/download)
- :mod:`_fallback_log`  — journal des fallbacks d'importer
"""

from __future__ import annotations

__all__: list[str] = []
