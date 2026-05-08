"""Shim de compatibilité — métrique relocalisée.

Sprint E.1 du plan v2.0 (mai 2026) — module migré depuis
``picarones.measurements.modern_archives`` vers
``picarones.evaluation.metrics.modern_archives`` (couche canonique).
Ce shim re-exporte l'API publique avec un ``DeprecationWarning``
et sera supprimé en 2.0.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "picarones.measurements.modern_archives est obsolète et sera supprimé en 2.0.  "
    "Utiliser picarones.evaluation.metrics.modern_archives à la place.",
    DeprecationWarning,
    stacklevel=2,
)

from picarones.evaluation.metrics.modern_archives import *  # noqa: F401, F403, E402
