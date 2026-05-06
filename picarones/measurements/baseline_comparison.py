"""Re-export — Sprint A14-S10. Le contenu canonique vit dans
``picarones.evaluation.metrics.baseline_comparison``.

L'ancien chemin ``picarones.measurements.baseline_comparison`` est conservé pour
ne casser aucun consommateur.  Au S22, ce re-export disparaîtra.
"""

from __future__ import annotations

from picarones.evaluation.metrics.baseline_comparison import *  # noqa: F401,F403
