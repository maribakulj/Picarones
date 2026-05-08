"""Shim de compatibilité — métrique relocalisée.

Sprint E.2 du plan v2.0 (mai 2026) — module migré depuis
``picarones.measurements.reading_order`` vers
``picarones.evaluation.metrics.reading_order`` (couche canonique).
Ce shim re-exporte l'API publique avec un ``DeprecationWarning``
et sera supprimé en 2.0.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "picarones.measurements.reading_order est obsolète et sera supprimé en 2.0.  "
    "Utiliser picarones.evaluation.metrics.reading_order à la place.",
    DeprecationWarning,
    stacklevel=2,
)

from picarones.evaluation.metrics.reading_order import *  # noqa: F401, F403, E402
