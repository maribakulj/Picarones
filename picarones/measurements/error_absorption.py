"""Re-export — Sprint A14-S10. Le contenu canonique vit dans
``picarones.evaluation.metrics.error_absorption``.

L'ancien chemin ``picarones.measurements.error_absorption`` est conservé pour
ne casser aucun consommateur.  Au S22, ce re-export disparaîtra.
"""

from __future__ import annotations

from picarones.evaluation.metrics.error_absorption import *  # noqa: F401,F403
