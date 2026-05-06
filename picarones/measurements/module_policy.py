"""Re-export — Sprint A14-S10. Le contenu canonique vit dans
``picarones.evaluation.metrics.module_policy``.

L'ancien chemin ``picarones.measurements.module_policy`` est conservé pour
ne casser aucun consommateur.  Au S22, ce re-export disparaîtra.
"""

from __future__ import annotations

from picarones.evaluation.metrics.module_policy import *  # noqa: F401,F403
