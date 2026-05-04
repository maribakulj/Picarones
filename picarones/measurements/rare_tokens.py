"""Re-export — Sprint A14-S10. Le contenu canonique vit dans
``picarones.evaluation.metrics.rare_tokens``.

L'ancien chemin ``picarones.measurements.rare_tokens`` est conservé pour
ne casser aucun consommateur.  Au S22, ce re-export disparaîtra.
"""

from __future__ import annotations

from picarones.evaluation.metrics.rare_tokens import *  # noqa: F401,F403
