"""Normalisation et manipulation de texte.

Sprint A14-S9 livre ``normalization.py``, déplacé depuis
``picarones/measurements/normalization.py`` sans modification de
logique.  L'ancien emplacement reste un re-export pour ne pas
casser les ~50 consommateurs (sera retiré au S22).

11 profils intégrés : ``nfc``, ``caseless``, ``minimal``,
``medieval_french``, ``early_modern_french``, ``medieval_latin``,
``medieval_english``, ``early_modern_english``, ``secretary_hand``,
``sans_ponctuation``, ``sans_apostrophes``.

Règle architecturale : ce module ne fait **pas** d'extraction depuis
ALTO/PAGE (c'est le rôle des projecteurs dans
``picarones.evaluation.projectors``).  Il prend une chaîne en entrée,
applique un profil, retourne une chaîne.
"""

from __future__ import annotations

from picarones.formats.text.normalization import (
    DEFAULT_DIPLOMATIC_PROFILE,
    DIPLOMATIC_EN_EARLY_MODERN,
    DIPLOMATIC_EN_MEDIEVAL,
    DIPLOMATIC_EN_SECRETARY,
    DIPLOMATIC_FR_EARLY_MODERN,
    DIPLOMATIC_FR_MEDIEVAL,
    DIPLOMATIC_LATIN_MEDIEVAL,
    DIPLOMATIC_MINIMAL,
    NORMALIZATION_PROFILES,
    NormalizationProfile,
    get_builtin_profile,
)

__all__ = [
    "NormalizationProfile",
    "NORMALIZATION_PROFILES",
    "DEFAULT_DIPLOMATIC_PROFILE",
    "get_builtin_profile",
    "DIPLOMATIC_FR_MEDIEVAL",
    "DIPLOMATIC_FR_EARLY_MODERN",
    "DIPLOMATIC_LATIN_MEDIEVAL",
    "DIPLOMATIC_MINIMAL",
    "DIPLOMATIC_EN_EARLY_MODERN",
    "DIPLOMATIC_EN_MEDIEVAL",
    "DIPLOMATIC_EN_SECRETARY",
]
