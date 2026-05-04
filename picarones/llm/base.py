"""Re-export — Sprint A14-S11. Le contenu canonique vit dans
``picarones.adapters.llm.base``.

L'ancien chemin ``picarones.llm.base`` est conservé pour ne casser
aucun consommateur.  Au S22, ce re-export disparaîtra.
"""

from __future__ import annotations

from picarones.adapters.llm.base import *  # noqa: F401,F403
