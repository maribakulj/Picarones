"""Re-export — Sprint A14-S10. Le contenu canonique vit dans
``picarones.evaluation.metrics.pricing``.

L'ancien chemin ``picarones.measurements.pricing`` est conservé pour
ne casser aucun consommateur.  Au S22, ce re-export disparaîtra.

Ce module ré-expose **explicitement** le symbole privé
``_DEFAULT_PRICING_PATH`` qu'au moins un consommateur importe
directement (cf. tests).
"""

from __future__ import annotations

from picarones.evaluation.metrics.pricing import *  # noqa: F401,F403
from picarones.evaluation.metrics.pricing import _DEFAULT_PRICING_PATH  # noqa: F401
