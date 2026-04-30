"""Alias rétrocompat — package déplacé dans :mod:`picarones.measurements.narrative.detectors`.

Phase E du chantier de refonte. Les 18 détecteurs en 6 familles
(ranking, pareto, stratum, quality, history, ensemble) vivent
désormais dans ``picarones.measurements.narrative.detectors/``.
"""

from picarones.measurements.narrative.detectors import *  # noqa: F401, F403

import picarones.measurements.narrative.detectors as _module
__all__ = getattr(_module, "__all__", [
    nm for nm in dir(_module) if not nm.startswith("_")
])
