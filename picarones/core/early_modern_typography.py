"""Alias rétrocompat — module déplacé dans :mod:`picarones.extras.historical.early_modern_typography`.

Phase B du chantier de refonte en 3 cercles (architecture-cercles.md).
Ce module philologique est désormais en Cercle 3 (``extras/``). L'alias
ici permet aux imports historiques (``from picarones.core.early_modern_typography
import ...``) de continuer à fonctionner sans modification.

Voir :doc:`docs/architecture-cercles.md` et l'extra
``picarones[historical]`` du ``pyproject.toml``.
"""

from picarones.extras.historical.early_modern_typography import *  # noqa: F401, F403

import picarones.extras.historical.early_modern_typography as _module
__all__ = getattr(_module, "__all__", [
    name for name in dir(_module) if not name.startswith("_")
])
