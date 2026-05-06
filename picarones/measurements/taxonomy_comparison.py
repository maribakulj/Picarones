"""Re-export — Sprint A14-S10. Le contenu canonique vit dans
``picarones.evaluation.metrics.taxonomy_comparison``.

L'ancien chemin ``picarones.measurements.taxonomy_comparison`` est conservé pour
ne casser aucun consommateur.  Au S22, ce re-export disparaîtra.
"""

from __future__ import annotations

from picarones.evaluation.metrics.taxonomy_comparison import *  # noqa: F401,F403
