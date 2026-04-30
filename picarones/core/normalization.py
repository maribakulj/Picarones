"""Alias rétrocompat — module déplacé dans :mod:`picarones.measurements.normalization`.

Phase E du chantier de refonte en 3 cercles. Cette mesure (Cercle 2)
n'est plus dans ``picarones.core/`` ; elle vit dans
``picarones.measurements/``. L'alias ici permet aux imports
historiques (``from picarones.core.normalization import ...``) de continuer
à fonctionner sans modification.

Voir :doc:`docs/architecture-cercles.md` pour la cartographie des
3 cercles. Le ``core/`` strict ne contient plus que les abstractions
du domaine et l'orchestration (Cercle 1).
"""

from picarones.measurements.normalization import *  # noqa: F401, F403

import picarones.measurements.normalization as _module
__all__ = getattr(_module, "__all__", [
    nm for nm in dir(_module) if not nm.startswith("_")
])
