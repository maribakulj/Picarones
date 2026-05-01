"""Alias rétrocompat — module déplacé dans :mod:`picarones.measurements.narrative.facts`.

Le contenu vit désormais dans son cercle d'origine. Cet alias permet
aux imports historiques (y compris les noms privés ``_*``) de
continuer à fonctionner sans modification.

Voir :doc:`docs/architecture-cercles.md` pour la cartographie.
"""

from picarones.measurements.narrative.facts import *  # noqa: F401, F403

# Réexport explicite de TOUS les noms (privés inclus) pour la
# rétrocompatibilité des tests Sprints qui importent ``_helper``,
# ``_compute_X``, ``_SCIPY_AVAILABLE``, etc. Sans cette boucle, ``import *``
# ne propage que les noms publics et casse les imports historiques.
import picarones.measurements.narrative.facts as _shim_module
for _shim_name in dir(_shim_module):
    if _shim_name == "__builtins__":
        continue
    if _shim_name not in globals():
        globals()[_shim_name] = getattr(_shim_module, _shim_name)
del _shim_module, _shim_name

__all__ = [
    _n for _n in dir() if not _n.startswith("__")
]
