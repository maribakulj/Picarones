"""Re-export — Sprint A14-S10. Le contenu canonique vit dans
``picarones.evaluation.metrics.robustness_projection``.

L'ancien chemin ``picarones.measurements.robustness_projection`` est
conservé pour ne casser aucun consommateur.  Au S22, ce re-export
disparaîtra.

Ré-expose explicitement ``_extract_quality_value`` et
``_interpolate_cer`` (symboles privés utilisés downstream).
"""

from __future__ import annotations

from picarones.evaluation.metrics.robustness_projection import *  # noqa: F401,F403
from picarones.evaluation.metrics.robustness_projection import (  # noqa: F401
    _extract_quality_value,
    _interpolate_cer,
)
