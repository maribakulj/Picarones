"""Alias rétrocompat — package déplacé dans :mod:`picarones.measurements.narrative`.

Phase E du chantier de refonte en 3 cercles. Le moteur narratif
(Cercle 2) vit désormais dans ``picarones.measurements.narrative``.
Cet alias maintient la rétrocompat des imports historiques :
``from picarones.core.narrative import build_synthesis``,
``from picarones.core.narrative.facts import Fact``, etc.
"""

from picarones.measurements.narrative import *  # noqa: F401, F403

import picarones.measurements.narrative as _module
# Réexport explicite des noms privés (préfixe ``_``) que ``import *``
# ne propage pas — rétrocompat des tests Sprints qui importent
# directement ``_DEFAULT_REGISTRY`` (test_sprint19_narrative_engine).
for _shim_name in dir(_module):
    if _shim_name == "__builtins__":
        continue
    if _shim_name not in globals():
        globals()[_shim_name] = getattr(_module, _shim_name)
del _shim_name
__all__ = getattr(_module, "__all__", [
    nm for nm in dir(_module) if not nm.startswith("_")
])
