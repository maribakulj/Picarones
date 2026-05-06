"""``picarones.measurements.normalization`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.evaluation.metrics.normalization`.
"""

from __future__ import annotations

import warnings

from picarones.evaluation.metrics.normalization import (  # noqa: F401
    NormalizationProfile,
    DIPLOMATIC_FR_MEDIEVAL,
    DIPLOMATIC_FR_EARLY_MODERN,
    DIPLOMATIC_LATIN_MEDIEVAL,
    DIPLOMATIC_MINIMAL,
    DIPLOMATIC_EN_EARLY_MODERN,
    DIPLOMATIC_EN_MEDIEVAL,
    DIPLOMATIC_EN_SECRETARY,
    NORMALIZATION_PROFILES,
    DEFAULT_DIPLOMATIC_PROFILE,
    get_builtin_profile,
    _parse_exclude_chars,
    _apply_diplomatic_table,
)

warnings.warn(
    "picarones.measurements.normalization is deprecated and will be removed in 2.0.  "
    "Import from picarones.evaluation.metrics.normalization instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ['NormalizationProfile', 'DIPLOMATIC_FR_MEDIEVAL', 'DIPLOMATIC_FR_EARLY_MODERN', 'DIPLOMATIC_LATIN_MEDIEVAL', 'DIPLOMATIC_MINIMAL', 'DIPLOMATIC_EN_EARLY_MODERN', 'DIPLOMATIC_EN_MEDIEVAL', 'DIPLOMATIC_EN_SECRETARY', 'NORMALIZATION_PROFILES', 'DEFAULT_DIPLOMATIC_PROFILE', 'get_builtin_profile', '_parse_exclude_chars', '_apply_diplomatic_table']
