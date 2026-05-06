"""``picarones.measurements.taxonomy`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.evaluation.metrics.taxonomy`.
"""

from __future__ import annotations

import warnings

from picarones.evaluation.metrics.taxonomy import (  # noqa: F401
    VISUAL_CONFUSIONS,
    TaxonomyResult,
    ERROR_CLASSES,
    classify_errors,
    aggregate_taxonomy,
    _VISUAL_PAIRS,
    _LATIN_BASIC,
    _classify_word_error,
    _is_ligature_error,
    _is_abbreviation_error,
    _is_diacritic_error,
    _is_visual_confusion,
    _is_oov_word,
)

warnings.warn(
    "picarones.measurements.taxonomy is deprecated and will be removed in 2.0.  "
    "Import from picarones.evaluation.metrics.taxonomy instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ['VISUAL_CONFUSIONS', 'TaxonomyResult', 'ERROR_CLASSES', 'classify_errors', 'aggregate_taxonomy']
