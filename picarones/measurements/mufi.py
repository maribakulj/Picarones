"""Shim de compatibilité — métrique relocalisée.

Sprint E.1 du plan v2.0 (mai 2026) — module migré depuis
``picarones.measurements.mufi`` vers
``picarones.evaluation.metrics.mufi`` (couche canonique).
Ce shim re-exporte l'API publique avec un ``DeprecationWarning``
et sera supprimé en 2.0.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "picarones.measurements.mufi est obsolète et sera supprimé en 2.0.  "
    "Utiliser picarones.evaluation.metrics.mufi à la place.",
    DeprecationWarning,
    stacklevel=2,
)

from picarones.evaluation.metrics.mufi import *  # noqa: F401, F403, E402
