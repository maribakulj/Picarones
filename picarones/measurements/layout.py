"""Re-export — Sprint A14-S10. Le contenu canonique vit dans
``picarones.evaluation.metrics.layout``.

L'ancien chemin ``picarones.measurements.layout`` est conservé pour
ne casser aucun consommateur.  Au S22, ce re-export disparaîtra.

Ré-expose explicitement le symbole privé ``_iou_bbox`` qu'au moins
un test importe directement.
"""

from __future__ import annotations

from picarones.evaluation.metrics.layout import *  # noqa: F401,F403
from picarones.evaluation.metrics.layout import _iou_bbox  # noqa: F401
