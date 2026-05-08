"""Shim de compatibilité — métrique relocalisée.

Sprint E.4 du plan v2.0 (mai 2026) — module migré depuis
``picarones.measurements.builtin_metrics`` vers
``picarones.evaluation.metrics.builtin_metrics`` (couche canonique).
Ce shim re-exporte l'API publique avec un ``DeprecationWarning``
et sera supprimé en 2.0.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "picarones.measurements.builtin_metrics est obsolète et sera supprimé en 2.0.  "
    "Utiliser picarones.evaluation.metrics.builtin_metrics à la place.",
    DeprecationWarning,
    stacklevel=2,
)

from picarones.evaluation.metrics.builtin_metrics import *  # noqa: F401, F403, E402
