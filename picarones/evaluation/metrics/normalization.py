"""Re-export depuis ``picarones.formats.text.normalization`` — Sprint A14-S9.

Le contenu canonique de ce module a été déplacé vers
``picarones/formats/text/normalization.py`` au Sprint S9 du
rewrite ciblé (cf. ``docs/roadmap/rewrite-2026.md``).

Ce fichier est conservé comme re-export pour ne **rien casser**
chez les ~50 consommateurs qui font ``from
picarones.formats.text.normalization import X``.  Les symboles
publics ET privés utilisés downstream (``_parse_exclude_chars``,
``_apply_diplomatic_table``) sont ré-exposés explicitement.

Plan de migration
-----------------
Au S22, les consommateurs qui importent encore depuis cet
emplacement seront migrés vers ``picarones.formats.text.normalization``
et ce re-export disparaîtra.

Règle architecturale
--------------------
``measurements/`` (ancien code legacy) est autorisé à importer
``formats/`` (nouveau code) pendant la phase de migration.
L'inverse est interdit (vérifié par ``test_layer_dependencies``).
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
    _apply_diplomatic_table,
    _parse_exclude_chars,
    get_builtin_profile,
)

__all__ = [
    "NormalizationProfile",
    "DIPLOMATIC_FR_MEDIEVAL",
    "DIPLOMATIC_FR_EARLY_MODERN",
    "DIPLOMATIC_LATIN_MEDIEVAL",
    "DIPLOMATIC_MINIMAL",
    "DIPLOMATIC_EN_EARLY_MODERN",
    "DIPLOMATIC_EN_MEDIEVAL",
    "DIPLOMATIC_EN_SECRETARY",
    "NORMALIZATION_PROFILES",
    "DEFAULT_DIPLOMATIC_PROFILE",
    "get_builtin_profile",
    "_parse_exclude_chars",
    "_apply_diplomatic_table",
]
