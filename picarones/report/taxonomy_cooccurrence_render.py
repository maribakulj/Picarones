"""Alias rétrocompat — module déplacé dans :mod:`picarones.extras.render.taxonomy_cooccurrence_render`.

Phase A du chantier de refonte en 3 cercles (architecture-cercles.md).
Le contenu vit désormais dans son cercle 3 ``extras/``. Cet alias
permet aux imports historiques (``from picarones.report.taxonomy_cooccurrence_render
import ...``) de continuer à fonctionner sans modification.

Voir :doc:`docs/architecture-cercles.md` pour la justification du
classement de ce module au Cercle 3.
"""

from picarones.extras.render.taxonomy_cooccurrence_render import *  # noqa: F401, F403

# Réexport explicite des éventuels noms privés ou modules accédés
# directement par leur attribut (rare mais possible). Pour la plupart
# des modules, l'``import *`` ci-dessus suffit.
import picarones.extras.render.taxonomy_cooccurrence_render as _module
__all__ = getattr(_module, "__all__", [
    name for name in dir(_module) if not name.startswith("_")
])
