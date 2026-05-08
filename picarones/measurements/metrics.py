"""Shim de compatibilité — métriques texte CER/WER.

Sprint E.3 du plan v2.0 (mai 2026) — module migré depuis
``picarones.measurements.metrics`` vers
``picarones.evaluation.metrics.text_metrics`` (couche canonique).
Ce shim re-exporte l'API publique avec un ``DeprecationWarning``
et sera supprimé en 2.0.

Note de renommage
-----------------
Le module a été renommé ``metrics`` → ``text_metrics`` au moment du
déplacement, parce que ``picarones.evaluation.metrics`` est déjà le
package (impossible de mettre un sous-module ``metrics.py`` dedans).
``text_metrics`` est descriptif — CER/WER/MER/WIL pour texte plat.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "picarones.measurements.metrics est obsolète et sera supprimé en 2.0.  "
    "Utiliser picarones.evaluation.metrics.text_metrics à la place.",
    DeprecationWarning,
    stacklevel=2,
)

from picarones.evaluation.metrics.text_metrics import *  # noqa: F401, F403, E402
