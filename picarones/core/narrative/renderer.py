"""Alias rétrocompat — module déplacé dans :mod:`picarones.measurements.narrative.renderer`.

Phase E du chantier de refonte en 3 cercles. Le moteur narratif
(Cercle 2 — measurements/) a quitté ``picarones.core.narrative``.
Cet alias maintient la rétrocompat des imports historiques.
"""

from picarones.measurements.narrative.renderer import *  # noqa: F401, F403

import picarones.measurements.narrative.renderer as _module
__all__ = getattr(_module, "__all__", [
    nm for nm in dir(_module) if not nm.startswith("_")
])
