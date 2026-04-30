"""Alias rétrocompat — module déplacé dans :mod:`picarones.extras.importers.huggingface`.

Phase C du chantier de refonte en 3 cercles (architecture-cercles.md).
Cet importeur est désormais en Cercle 3 (``extras/importers/``). L'alias
ici permet aux imports historiques (``from picarones.importers.huggingface
import ...``) de continuer à fonctionner.

Voir :doc:`docs/architecture-cercles.md` et l'extra
``picarones[importers]`` du ``pyproject.toml``.
"""

from picarones.extras.importers.huggingface import *  # noqa: F401, F403

import picarones.extras.importers.huggingface as _module
__all__ = getattr(_module, "__all__", [
    name for name in dir(_module) if not name.startswith("_")
])
