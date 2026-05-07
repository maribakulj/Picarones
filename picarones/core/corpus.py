"""``picarones.core.corpus`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.evaluation.corpus`.  Phase 4-quater
du retrait du legacy.
"""

from __future__ import annotations

import warnings

from picarones.evaluation.corpus import *  # noqa: F401, F403
from picarones.evaluation.corpus import (  # noqa: F401
    GT_SUFFIXES,
    IMAGE_EXTENSIONS,
    AltoGT,
    Corpus,
    Document,
    EntitiesGT,
    GTLevel,
    GTPayload,
    PageGT,
    ReadingOrderGT,
    TextGT,
    _load_extra_gt_levels,
    load_corpus_from_directory,
)

warnings.warn(
    "picarones.core.corpus is deprecated and will be removed in 2.0.  "
    "Import from picarones.evaluation.corpus instead.",
    DeprecationWarning,
    stacklevel=2,
)
