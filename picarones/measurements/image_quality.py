"""Re-export — Sprint A14-S10. Le contenu canonique vit dans
``picarones.evaluation.metrics.image_quality``.

L'ancien chemin ``picarones.measurements.image_quality`` est conservé
pour ne casser aucun consommateur.  Au S22, ce re-export disparaîtra.

Ré-expose explicitement ``_global_quality_score`` (symbole privé
utilisé downstream).
"""

from __future__ import annotations

from picarones.evaluation.metrics.image_quality import *  # noqa: F401,F403
from picarones.evaluation.metrics.image_quality import _global_quality_score  # noqa: F401
