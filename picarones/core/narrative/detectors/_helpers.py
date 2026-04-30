"""Alias rétrocompat — module déplacé dans :mod:`picarones.measurements.narrative.detectors._helpers`.

Phase E du chantier de refonte en 3 cercles. Le moteur narratif
(Cercle 2 — measurements/) a quitté ``picarones.core.narrative``.
Cet alias maintient la rétrocompat des imports historiques.
"""

from picarones.measurements.narrative.detectors._helpers import *  # noqa: F401, F403

import picarones.measurements.narrative.detectors._helpers as _module
__all__ = getattr(_module, "__all__", [
    nm for nm in dir(_module) if not nm.startswith("_")
])
