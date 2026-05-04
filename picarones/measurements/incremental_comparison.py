"""Re-export — Sprint A14-S10. Le contenu canonique vit dans
``picarones.evaluation.metrics.incremental_comparison``.

L'ancien chemin ``picarones.measurements.incremental_comparison`` est conservé pour
ne casser aucun consommateur.  Au S22, ce re-export disparaîtra.
"""

from __future__ import annotations

from picarones.evaluation.metrics.incremental_comparison import *  # noqa: F401,F403
