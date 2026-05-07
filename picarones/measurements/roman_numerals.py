"""``picarones.measurements.roman_numerals`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.evaluation.metrics.roman_numerals`.
Phase 5.C.batch7 du retrait du legacy.
"""

from __future__ import annotations

import warnings

from picarones.evaluation.metrics.roman_numerals import *  # noqa: F401, F403

warnings.warn(
    "picarones.measurements.roman_numerals is deprecated and will be removed in 2.0.  "
    "Import from picarones.evaluation.metrics.roman_numerals instead.",
    DeprecationWarning,
    stacklevel=2,
)
