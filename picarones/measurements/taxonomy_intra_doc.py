"""``picarones.measurements.taxonomy_intra_doc`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.evaluation.metrics.taxonomy_intra_doc`.
"""

from __future__ import annotations

import warnings

from picarones.evaluation.metrics.taxonomy_intra_doc import (  # noqa: F401
    compute_taxonomy_position_heatmap,
    _classify_word_pair,
    _bin_for_position,
)

warnings.warn(
    "picarones.measurements.taxonomy_intra_doc is deprecated and will be removed in 2.0.  "
    "Import from picarones.evaluation.metrics.taxonomy_intra_doc instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ['compute_taxonomy_position_heatmap']
