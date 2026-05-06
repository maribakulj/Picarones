"""``picarones.measurements.ner_backends`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.evaluation.metrics.ner_backends`.
"""

from __future__ import annotations

import warnings

from picarones.evaluation.metrics.ner_backends import (  # noqa: F401
    EntityExtractor,
    SpacyEntityExtractor,
    SPACY_PROFILES,
    get_extractor,
    is_spacy_available,
)

warnings.warn(
    "picarones.measurements.ner_backends is deprecated and will be removed in 2.0.  "
    "Import from picarones.evaluation.metrics.ner_backends instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ['EntityExtractor', 'SpacyEntityExtractor', 'SPACY_PROFILES', 'get_extractor', 'is_spacy_available']
