"""Shim de compatibilité — détection de sur-normalisation LLM.

Phase 6 (mai 2026) — l'implémentation canonique vit désormais dans
``picarones.evaluation.metrics.over_normalization``.  Ce shim ré-exporte
l'API publique avec un ``DeprecationWarning`` et sera supprimé en 2.0.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "picarones.pipelines.over_normalization est obsolète et sera supprimé en 2.0. "
    "Utiliser picarones.evaluation.metrics.over_normalization à la place.",
    DeprecationWarning,
    stacklevel=2,
)

from picarones.evaluation.metrics.over_normalization import *  # noqa: F401, F403, E402
from picarones.evaluation.metrics.over_normalization import (  # noqa: E402
    OverNormalizationResult,
    aggregate_over_normalization,
    detect_over_normalization,
)

__all__ = [
    "OverNormalizationResult",
    "aggregate_over_normalization",
    "detect_over_normalization",
]
